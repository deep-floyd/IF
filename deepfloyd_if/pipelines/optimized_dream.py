import gc
import numpy as np

import torch.cuda
from PIL import Image


def run_garbage_collection():
    gc.collect()
    torch.cuda.empty_cache()


def to_pil_images(images: torch.Tensor) -> list[Image]:
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = np.round(images * 255).astype(np.uint8)
    return [Image.fromarray(image) for image in images]


def run_stage1(
        model,
        t5_embs,
        negative_t5_embs,
        seed: int = 0,
        num_images: int = 1,
        guidance_scale_1: float = 7.0,
        custom_timesteps_1: str = 'smart100',
        num_inference_steps_1: int = 100,
        aspect_ratio='1:1',
        img_size=(64, 64)
):
    run_garbage_collection()

    if custom_timesteps_1 == "none":
        custom_timesteps_1 = str(num_inference_steps_1)

    ret_images1, ret_images2 = [], []
    for _ in range(num_images):
        images, _ = model.embeddings_to_image(t5_embs=t5_embs,
                                              negative_t5_embs=negative_t5_embs,
                                              guidance_scale=guidance_scale_1, img_size=img_size,
                                              sample_timestep_respacing=custom_timesteps_1,
                                              seed=seed, aspect_ratio=aspect_ratio, force_size=True
                                              )
        pil_images_I = model.to_images(images, disable_watermark=True)
        ret_images1.append(pil_images_I[0])
        ret_images2.append(images[0])

    return ret_images2, ret_images1


def run_stage2(
        model,
        t5_embs,
        negative_t5_embs,
        images,
        seed: int = 0,
        guidance_scale: float = 4.0,
        custom_timesteps_2: str = 'smart50',
        num_inference_steps_2: int = 50,
        disable_watermark: bool = True,
        device=None
) -> Image:
    run_garbage_collection()

    if custom_timesteps_2 == "none":
        custom_timesteps_2 = str(num_inference_steps_2)
    stageII_generations, _ = model.embeddings_to_image(low_res=images,
                                                       t5_embs=t5_embs,
                                                       negative_t5_embs=negative_t5_embs,
                                                       guidance_scale=guidance_scale,
                                                       sample_timestep_respacing=custom_timesteps_2,
                                                       seed=seed, device=device)
    pil_images_II = model.to_images(stageII_generations, disable_watermark=disable_watermark)
    return stageII_generations, pil_images_II


def run_stage3(
        model,
        prompt,
        negative_t5_embs,
        images,
        seed: int = 0,
        guidance_scale: float = 4.0,
        custom_timesteps_2: str = 'smart50',
        num_inference_steps_2: int = 50,
        disable_watermark: bool = True,
        device=None
) -> Image:
    run_garbage_collection()
    stageII_generations, _ = model.embeddings_to_image(low_res=images,
                                                       prompt=prompt,
                                                       negative_t5_embs=negative_t5_embs,
                                                       guidance_scale=guidance_scale,
                                                       sample_timestep_respacing=num_inference_steps_2,
                                                       num_images_per_prompt=1,
                                                       noise_level=20,
                                                       seed=seed, device=device)
    pil_images_III = model.to_images(stageII_generations, disable_watermark=disable_watermark)
    return pil_images_III
