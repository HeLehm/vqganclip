import os
from vqganclip.vqganclip import VQGANCLIP
from vqganclip.helper import download_vqgan_model

yaml_url = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
yaml_path = os.path.join(
    os.path.dirname(__file__),
    'vqgan_imagenet_f16_16384.yaml'
)

ckpt_url = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
ckpt_path = os.path.join(
    os.path.dirname(__file__),
    'vqgan_imagenet_f16_16384.ckpt'
)

download_vqgan_model(yaml_url,yaml_path)
download_vqgan_model(ckpt_url,ckpt_path)

ai = VQGANCLIP(
    prompts=[''],
    vqgan_config_path=yaml_path,
    vqgan_checkpoint_path=ckpt_path
)

ai.set_prompts(['anything you want','unreal engine'])
for _ in range(200):
    ai.train_step()

ai.get_img().save(
    os.path.join(
        os.path.dirname(__file__),
        'anything you want.png'
    )
)




