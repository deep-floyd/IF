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
):
    run_garbage_collection()

    images, _ = model.embeddings_to_image(t5_embs=t5_embs,
                                          negative_t5_embs=negative_t5_embs,
                                          num_images_per_prompt=num_images,
                                          guidance_scale=guidance_scale_1,
                                          sample_timestep_respacing=custom_timesteps_1,
                                          seed=seed
                                          ).images
    pil_images_I = model.to_images(images, disable_watermark=True)

    return pil_images_I


def run_stage2(
        model,
        stage1_result,
        stage2_index: int,
        seed_2: int = 0,
        guidance_scale_2: float = 4.0,
        custom_timesteps_2: str = 'smart50',
        num_inference_steps_2: int = 50,
        disable_watermark: bool = True,
) -> Image:
    run_garbage_collection()

    prompt_embeds = stage1_result['prompt_embeds']
    negative_embeds = stage1_result['negative_embeds']
    images = stage1_result['images']
    images = images[[stage2_index]]

    stageII_generations, _ = model.embeddings_to_image(low_res=images,
                                                       t5_embs=prompt_embeds,
                                                       negative_t5_embs=negative_embeds,
                                                       guidance_scale=guidance_scale_2,
                                                       sample_timestep_respacing=custom_timesteps_2,
                                                       seed=seed_2)
    pil_images_II = model.to_images(stageII_generations, disable_watermark=disable_watermark)

    return pil_images_II


def run_stage3(
        model,
        image: Image,
        t5_embs,
        negative_t5_embs,
        seed_3: int = 0,
        guidance_scale_3: float = 9.0,
        sample_timestep_respacing='super40',
        disable_watermark=True
) -> Image:
    run_garbage_collection()

    _stageIII_generations, _ = model.embeddings_to_image(image=image,
                                                         t5_embs=t5_embs,
                                                         negative_t5_embs=negative_t5_embs,
                                                         num_images_per_prompt=1,
                                                         guidance_scale=guidance_scale_3,
                                                         noise_level=100,
                                                         sample_timestep_respacing=sample_timestep_respacing,
                                                         seed=seed_3)
    pil_image_III = model.to_images(_stageIII_generations, disable_watermark=disable_watermark)

    return pil_image_III
