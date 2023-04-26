# -*- coding: utf-8 -*-

from datetime import datetime

import PIL
import torch

from .utils import _prepare_pil_image


def style_transfer(
    t5,
    if_I,
    if_II,
    if_III=None,
    *,
    support_pil_img,
    style_prompt,
    prompt=None,
    negative_prompt=None,
    seed=None,
    if_I_kwargs=None,
    if_II_kwargs=None,
    if_III_kwargs=None,
    progress=True,
    return_tensors=False,
    disable_watermark=False,
):
    assert isinstance(support_pil_img, PIL.Image.Image)

    bs = len(style_prompt)

    if seed is None:
        seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))

    if prompt is not None:
        t5_embs = t5.get_text_embeddings(prompt)
    else:
        t5_embs = t5.get_text_embeddings(style_prompt)

    style_t5_embs = t5.get_text_embeddings(style_prompt)

    if negative_prompt is not None:
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        negative_t5_embs = t5.get_text_embeddings(negative_prompt)
    else:
        negative_t5_embs = None

    low_res = _prepare_pil_image(support_pil_img, 64)
    mid_res = _prepare_pil_image(support_pil_img, 256)
    # high_res = _prepare_pil_image(support_pil_img, 1024)

    result = {}
    if if_I is not None:
        _, _, image_h, image_w = low_res.shape
        if_I_kwargs = if_I_kwargs or {'sample_timestep_respacing': '20,20,20,20,10,0,0,0,0,0'}
        if_I_kwargs['seed'] = seed
        if_I_kwargs['progress'] = progress
        if_I_kwargs['aspect_ratio'] = f'{image_w}:{image_h}'

        if_I_kwargs['t5_embs'] = t5_embs
        if_I_kwargs['style_t5_embs'] = style_t5_embs
        if_I_kwargs['positive_t5_embs'] = style_t5_embs
        if_I_kwargs['negative_t5_embs'] = negative_t5_embs

        if_I_kwargs['support_noise'] = low_res

        stageI_generations, _ = if_I.embeddings_to_image(**if_I_kwargs)
        pil_images_I = if_I.to_images(stageI_generations, disable_watermark=disable_watermark)

        result['I'] = pil_images_I
    else:
        stageI_generations = None

    if if_II is not None:
        if stageI_generations is None:
            stageI_generations = low_res.repeat(bs, 1, 1, 1)

        if_II_kwargs = if_II_kwargs or {}
        if_II_kwargs['low_res'] = stageI_generations
        if_II_kwargs['seed'] = seed
        if_II_kwargs['t5_embs'] = t5_embs
        if_II_kwargs['style_t5_embs'] = style_t5_embs
        if_II_kwargs['positive_t5_embs'] = style_t5_embs
        if_II_kwargs['negative_t5_embs'] = negative_t5_embs
        if_II_kwargs['progress'] = progress

        if_II_kwargs['support_noise'] = mid_res

        stageII_generations, _meta = if_II.embeddings_to_image(**if_II_kwargs)
        pil_images_II = if_II.to_images(stageII_generations, disable_watermark=disable_watermark)

        result['II'] = pil_images_II
    else:
        stageII_generations = None

    if if_II is not None and if_III is not None:
        if_III_kwargs = if_III_kwargs or {}

        stageIII_generations = []
        for idx in range(len(stageII_generations)):
            if if_III.use_diffusers:
                if_III_kwargs["prompt"] = prompt[idx: idx+1] if prompt is not None else style_prompt[idx: idx+1]

            if_III_kwargs['low_res'] = stageII_generations[idx:idx+1]
            if_III_kwargs['seed'] = seed
            if_III_kwargs['t5_embs'] = t5_embs[idx:idx+1]
            if_III_kwargs['progress'] = progress
            style_t5_embs = if_I_kwargs.get('style_t5_embs')
            if style_t5_embs is not None:
                style_t5_embs = style_t5_embs[idx:idx+1]
            positive_t5_embs = if_I_kwargs.get('positive_t5_embs')
            if positive_t5_embs is not None:
                positive_t5_embs = positive_t5_embs[idx:idx+1]
            if_III_kwargs['style_t5_embs'] = style_t5_embs
            if_III_kwargs['positive_t5_embs'] = positive_t5_embs

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
