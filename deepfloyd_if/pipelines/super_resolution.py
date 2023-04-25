# -*- coding: utf-8 -*-

from datetime import datetime

import PIL
from .utils import _prepare_pil_image


def super_resolution(
    t5,
    if_III=None,
    *,
    support_pil_img,
    prompt=None,
    negative_prompt=None,
    seed=None,
    if_III_kwargs=None,
    progress=True,
    img_size=256,
    img_scale=4.0,
    return_tensors=False,
    disable_watermark=False,
):
    assert isinstance(support_pil_img, PIL.Image.Image)
    assert img_size % 8 == 0

    if seed is None:
        seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))

    if prompt is not None:
        t5_embs = t5.get_text_embeddings(prompt)
    else:
        t5_embs = t5.get_text_embeddings('')

    if negative_prompt is not None:
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        negative_t5_embs = t5.get_text_embeddings(negative_prompt)
    else:
        negative_t5_embs = None

    low_res = _prepare_pil_image(support_pil_img, img_size)

    result = {}

    bs = 1
    if_III_kwargs = if_III_kwargs or {}

    if_III_kwargs["prompt"] = prompt
    if_III_kwargs['low_res'] = low_res.repeat(bs, 1, 1, 1)
    if_III_kwargs['seed'] = seed
    if_III_kwargs['noise_level'] = if_III_kwargs.get("noise_level", 20)

    stageIII_generations, _ = if_III.embeddings_to_image(**if_III_kwargs)
    pil_images_III = if_III.to_images(stageIII_generations, disable_watermark=disable_watermark)
    result['III'] = pil_images_III

    if return_tensors:
        return result, (stageIII_generations,)
    else:
        return result
