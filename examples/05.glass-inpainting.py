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
raw_pil_image = Image.open('raw-image/raw-image-04.png').convert('RGB').resize((1024, 1024))

pil_image = raw_pil_image.resize(
    (64, 64), resample=Image.Resampling.BICUBIC, reducing_gap=None
)
img = np.array(pil_image)
img = img.astype(np.float32) / 127.5 - 1
img = np.transpose(img, [2, 0, 1])
img = torch.from_numpy(img).unsqueeze(0)

if_I.to_images(img)[0].save(f'generate-imgs/05.glass_inpainting_64*64.png')

inpainting_mask = torch.zeros_like(img[0], device='cpu')
inpainting_mask[:, 26:36, 24:34] = 1
#inpainting_mask[:, 29:33, 34:36] = 1
#inpainting_mask[:, 26:36, 36:44] = 1
inpainting_mask = inpainting_mask.unsqueeze(0)

if_I.to_images((1-inpainting_mask)*img)[0].save(f'generate-imgs/05.glass_inpainting_64*64-mask-1.png')
print("inpainting_mask.shape:", inpainting_mask.shape)

# 将 inpainting_mask 转换为 NumPy 数组
#inpainting_mask_np = inpainting_mask.cpu().numpy()

result = inpainting(
    t5=t5, if_I=if_I, 
    if_II=if_II,
    if_III=if_III,
    support_pil_img=raw_pil_image,
    inpainting_mask=inpainting_mask,
    prompt=[
        'blue sunglasses',
        'yellow sunglasses',
        'red sunglasses',
        'green sunglasses',
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
)

# save the generated images as png files
for stage, images in result.items():
    for i, img in enumerate(images):
        img.save(f'generate-imgs/05.glass_inpainting_{stage}_{i}.png')