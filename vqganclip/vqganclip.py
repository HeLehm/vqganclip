from torchvision.transforms import functional as TF
from torch.nn import functional as F
from torchvision import transforms
import torch
from torch import optim
from PIL import Image
import argparse
from tqdm import tqdm
from IPython import display
import gc

#submodules
import sys 
import os
_clip_path = os.path.join(
    os.path.dirname(
        __file__
    ),
    "submodules",
    "CLIP"
)
sys.path.append(_clip_path)
import clip

#helper.py
from .helper import load_vqgan_model, MakeCutouts, fetch, Prompt, parse_prompt, resize_image, vector_quantize, clamp_with_grad


class VQGANCLIP:
    def __init__(
        self,
        prompts,
        vqgan_config_path,
        vqgan_checkpoint_path,
        image_prompts=[],
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size = [512,512],
        init_image_path = None,
        init_weight=0.,
        clip_model='ViT-B/32',
        step_size=0.05,
        cutn=64,
        cut_pow=1.,
        seed=0,
        ) -> None:

        self.args = argparse.Namespace(
            prompts=prompts,
            image_prompts=image_prompts,
            noise_prompt_seeds=noise_prompt_seeds,
            noise_prompt_weights=noise_prompt_weights,
            size=size,
            init_image=init_image_path,
            init_weight=init_weight,
            clip_model=clip_model,
            vqgan_config=vqgan_config_path,
            vqgan_checkpoint=vqgan_checkpoint_path,
            step_size=step_size,
            cutn=cutn,
            cut_pow=cut_pow,
            seed=seed
        )
        #will all be set later
        self.z_orig = None
        self.opt = None
        self.z = None
        self.pMs = None
    
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.model = load_vqgan_model(self.args.vqgan_config, self.args.vqgan_checkpoint).to(self.device)
        self.perceptor = clip.load(self.args.clip_model, jit=False)[0].eval().requires_grad_(False).to(self.device)
        

        cut_size = self.perceptor.visual.input_resolution
        self.e_dim = self.model.quantize.e_dim
        f = 2**(self.model.decoder.num_resolutions - 1)
        self.make_cutouts = MakeCutouts(cut_size, self.args.cutn, cut_pow=self.args.cut_pow)
        self.n_toks = self.model.quantize.n_e
        self.toksX, self.toksY = self.args.size[0] // f, self.args.size[1] // f
        self.sideX, self.sideY = self.toksX * f, self.toksY * f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)

        init_img = None
        if self.args.init_image is not None:
            init_img = Image.open(fetch(self.args.init_image)).convert('RGB')
        self.set_image(init_img)

        self.set_prompts(self.args.prompts, self.args.image_prompts, self.args.noise_prompt_seeds, self.args.noise_prompt_weights)
        
        self.last_losses = None
        self._i = 0

    def set_prompts(self,prompts = None,image_prompts = None,noise_prompt_seeds = None,noise_prompt_weights = None):
        """
        will reset the prompts in arg and set them up properly
        if a parameter is None it  will be ignored
        """
        pMs = []
        if prompts is not None:
            self.args.prompts = prompts
        if image_prompts:
            self.args.image_prompts = image_prompts
        if noise_prompt_seeds is not None and noise_prompt_weights is not None:
            self.args.noise_prompt_seeds, self.args.noise_prompt_weights = noise_prompt_seeds, noise_prompt_weights

        for prompt in self.args.prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(self.device)).float()
            pMs.append(Prompt(embed, weight, stop).to(self.device))

        for prompt in self.args.image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(fetch(path)).convert('RGB'), (self.sideX, self.sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            pMs.append(Prompt(embed, weight, stop).to(self.device))

        for seed, weight in zip(self.args.noise_prompt_seeds, self.args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            pMs.append(Prompt(embed, weight).to(self.device))
        self.pMs = pMs

    def set_image(self,pil_image):
        z = None
        if pil_image is not None:
            pil_image = pil_image.resize((self.sideX, self.sideY), Image.LANCZOS)
            z, *_ = self.model.encode(TF.to_tensor(pil_image).to(self.device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(self.n_toks, [self.toksY * self.toksX], device=self.device), self.n_toks).float()
            z = one_hot @ self.model.quantize.embedding.weight
            z = z.view([-1, self.toksY, self.toksX, self.e_dim]).permute(0, 3, 1, 2)
        self.z_orig = z.clone()
        z.requires_grad_(True)
        self.opt = optim.Adam([z], lr=self.args.step_size)
        self.z = z

    def set_step_size(self,step_size):
        self.args.step_size = step_size
        self.opt = optim.Adam([self.z], lr=self.args.step_size)

    @torch.no_grad()
    def get_img(self):
        """
        Returns
        -------
        current image as pil image
        (can be saved with .save(path...))
        """
        out = self.synth(self.z)
        return TF.to_pil_image(out[0].cpu())

    def synth(self,z):
        z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def show_current_img(self):
        if self.last_losses is not None:
            losses_str = ', '.join(f'{loss.item():g}' for loss in self.last_losses)
            tqdm.write(f'i: {self._i} loss: {sum(self.last_losses).item():g}, losses: {losses_str}')
        out = self.synth(self.z)
        display.display(TF.to_pil_image(out[0].cpu()))

    def get_save_path(self,i):
      return self.args.save_dir + self.args.save_file_name_prefix + str(i) + ".png"

    def ascend_txt(self):
        out = self.synth(self.z)
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []

        if self.args.init_weight:
            result.append(F.mse_loss(self.z, self.z_orig) * self.args.init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))

        return result

    def train_step(self):
        self.opt.zero_grad()
        lossAll = self.ascend_txt()
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))
        self._i += 1

    def get_args(self):
        return self.args
    
    
    def __del__(self):
        """
        cleanup
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VQGANCLIP instance deleted")
