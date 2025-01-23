import sys
import os
sys.path.append(os.getcwd())
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from PIL import Image
from deepfloyd_if.pipelines import style_transfer

# If the default space is insufficient, set a custom cache location for the model
#config_path = '/root/autodl-tmp/transformers-cache/'
config_path = None

device = 'cuda:0'

# Initialize the model
device = 'cuda:0'
if_I = IFStageI('IF-I-XL-v1.0', device=device, cache_dir=config_path)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=config_path)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device, cache_dir=config_path)
t5 = T5Embedder(device="cpu", cache_dir=config_path)

# Load the original image
raw_pil_image = Image.open('raw-image/raw-image-05.png').convert('RGB')


count = 4
prompt = 'white cat'

result = style_transfer(
    t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
    support_pil_img=raw_pil_image,
    prompt=[prompt]*count,
    style_prompt=[
        f'in style lego',
        f'in style zombie',
        f'in style origami',
        f'in style anime',
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 10.0,
        "sample_timestep_respacing": "10,10,10,10,10,0,0,0,0,0",
        'support_noise_less_qsample_steps': 5,
        'positive_mixer': 0.8,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": 'smart50',
        "support_noise_less_qsample_steps": 5,
        'positive_mixer': 1.0,
    },
)
#if_I.show(result['III'], 2, 14)

#print(result)

# save the generated images as png files
for stage, images in result.items():
    for i, img in enumerate(images):
        img.save(f'generate-imgs/06.style-transfer-double-prompt_{stage}_{i}.png')



    