# -*- coding: utf-8 -*-
from datetime import datetime

import PIL
import torch

from .utils import _prepare_pil_image


def inpainting(
    t5,
    if_I,
    if_II=None,
    if_III=None,
    *,
    support_pil_img,
    prompt,
    inpainting_mask,
    negative_prompt=None,
    seed=None,
    if_I_kwargs=None,
    if_II_kwargs=None,
    if_III_kwargs=None,
    progress=True,
    return_tensors=False,
    disable_watermark=False,
):
    from skimage.transform import resize  # noqa
    from skimage import img_as_bool  # noqa
    assert isinstance(support_pil_img, PIL.Image.Image)

    if seed is None:
        seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))

    t5_embs = t5.get_text_embeddings(prompt)

    if negative_prompt is not None:
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        negative_t5_embs = t5.get_text_embeddings(negative_prompt)
    else:
        negative_t5_embs = None

    low_res = _prepare_pil_image(support_pil_img, 64)
    mid_res = _prepare_pil_image(support_pil_img, 256)
    high_res = _prepare_pil_image(support_pil_img, 1024)

    result = {}

    _, _, image_h, image_w = low_res.shape
    if_I_kwargs = if_I_kwargs or {}
    if_I_kwargs['seed'] = seed
    if_I_kwargs['progress'] = progress
    if_I_kwargs['aspect_ratio'] = f'{image_w}:{image_h}'

    if_I_kwargs['t5_embs'] = t5_embs
    if_I_kwargs['negative_t5_embs'] = negative_t5_embs

    if_I_kwargs['support_noise'] = low_res

    inpainting_mask_I = img_as_bool(resize(inpainting_mask[0].cpu(), (3, image_h, image_w)))
    inpainting_mask_I = torch.from_numpy(inpainting_mask_I).unsqueeze(0).to(if_I.device)

    if_I_kwargs['inpainting_mask'] = inpainting_mask_I

    stageI_generations, _ = if_I.embeddings_to_image(**if_I_kwargs)
    pil_images_I = if_I.to_images(stageI_generations, disable_watermark=disable_watermark)

    result['I'] = pil_images_I

    if if_II is not None:
        _, _, image_h, image_w = mid_res.shape

        if_II_kwargs = if_II_kwargs or {}
        if_II_kwargs['low_res'] = stageI_generations
        if_II_kwargs['seed'] = seed
        if_II_kwargs['t5_embs'] = t5_embs
        if_II_kwargs['negative_t5_embs'] = negative_t5_embs
        if_II_kwargs['progress'] = progress

        if_II_kwargs['support_noise'] = mid_res

        if 'inpainting_mask' not in if_II_kwargs:
            inpainting_mask_II = img_as_bool(resize(inpainting_mask[0].cpu(), (3, image_h, image_w)))
            inpainting_mask_II = torch.from_numpy(inpainting_mask_II).unsqueeze(0).to(if_II.device)
            if_II_kwargs['inpainting_mask'] = inpainting_mask_II

        stageII_generations, _meta = if_II.embeddings_to_image(**if_II_kwargs)
        pil_images_II = if_II.to_images(stageII_generations, disable_watermark=disable_watermark)

        result['II'] = pil_images_II
    else:
        stageII_generations = None

    if if_II is not None and if_III is not None:
        _, _, image_h, image_w = high_res.shape
        if_III_kwargs = if_III_kwargs or {}

        stageIII_generations = []
        for idx in range(len(stageII_generations)):
            if_III_kwargs['low_res'] = stageII_generations[idx:idx+1]
            if_III_kwargs['seed'] = seed
            if_III_kwargs['t5_embs'] = t5_embs[idx:idx+1]
            if negative_t5_embs is not None:
                if_III_kwargs['negative_t5_embs'] = negative_t5_embs[idx:idx+1]
            if_III_kwargs['progress'] = progress
            if_III_kwargs['support_noise'] = high_res

            if 'inpainting_mask' not in if_III_kwargs:
                inpainting_mask_III = img_as_bool(resize(inpainting_mask[0].cpu(), (3, image_h, image_w)))
                inpainting_mask_III = torch.from_numpy(inpainting_mask_III).unsqueeze(0).to(if_III.device)
                if_III_kwargs['inpainting_mask'] = inpainting_mask_III

            _stageIII_generations, _meta = if_III.embeddings_to_image(**if_III_kwargs)
            stageIII_generations.append(_stageIII_generations)

        stageIII_generations = torch.cat(stageIII_generations, 0)
        pil_images_III = if_III.to_images(stageIII_generations, disable_watermark=disable_watermark)

        result['III'] = pil_images_III
    else:
        stageIII_generations = None

    if return_tensors:
        return result, (stageI_generations, stageII_generations, stageIII_generations)
    else:
        return result
