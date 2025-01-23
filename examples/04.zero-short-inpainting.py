import sys
import os
sys.path.append(os.getcwd())
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import inpainting
from PIL import Image
import numpy as np
import torch

# If the default space is insufficient, set a custom cache location for the model
#config_path = '/root/autodl-tmp/transformers-cache/'
config_path = None

device = 'cuda:0'

# Initialize the model
if_I = IFStageI('IF-I-XL-v1.0', device=device, cache_dir=config_path)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=config_path)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device, cache_dir=config_path)
t5 = T5Embedder(device="cpu", cache_dir=config_path)

# Load the original image
raw_pil_image = Image.open('raw-image/raw-image-03.png').convert('RGB')

img = np.array(raw_pil_image)
img = img.astype(np.float32) / 127.5 - 1
img = np.transpose(img, [2, 0, 1])
img = torch.from_numpy(img).unsqueeze(0)

# 创建 inpainting_mask
inpainting_mask = torch.zeros_like(img[0], device='cpu')
inpainting_mask[:, 0:210, 210:660] = 1
inpainting_mask = inpainting_mask.unsqueeze(0)

if_I.to_images((1-inpainting_mask)*img)[0].save(f'generate-imgs/04.oil-man_inpainting-mask-4.png')
print("inpainting_mask.shape:", inpainting_mask.shape)

result = inpainting(
    t5=t5, if_I=if_I,
    if_II=if_II,
    if_III=if_III,
    support_pil_img=raw_pil_image,
    inpainting_mask=inpainting_mask,
    prompt=[
        'detailed picture, 4k dslr, best quality, a man in a black hat',
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "10,10,10,10,10,0,0,0,0,0",
        'support_noise_less_qsample_steps': 0,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        'aug_level': 0.0,
        "sample_timestep_respacing": '100',
    },
    if_III_kwargs={
        "guidance_scale": 9.0,
        "noise_level": 20,
        "sample_timestep_respacing": "75",
    },
)

# save the generated images as png files
for stage, images in result.items():
    for i, img in enumerate(images):
        img.save(f'generate-imgs/04.zero-short-inpainting_{stage}_{i}-4.png')