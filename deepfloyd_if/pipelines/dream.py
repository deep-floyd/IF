# -*- coding: utf-8 -*-
from datetime import datetime

import torch


def dream(
    t5,
    if_I,
    if_II=None,
    if_III=None,
    *,
    prompt,
    style_prompt=None,
    negative_prompt=None,
    seed=None,
    aspect_ratio='1:1',
    if_I_kwargs=None,
    if_II_kwargs=None,
    if_III_kwargs=None,
    progress=True,
    return_tensors=False,
):
    """
    Generate pictures using text description!

    :param optional dict if_I_kwargs:
        "dynamic_thresholding_p": 0.95, [0.5, 1.0] it controls color saturation on high cfg values
        "dynamic_thresholding_c": 1.5, [1.0, 15.0] clips the limiter to avoid greyish images on high limiter values
        "guidance_scale": 7.0, [1.0, 20.0] control the level of text understanding
        "positive_mixer": 0.25, [0.0, 1.0] contribution of the second positive prompt, 0.0 - minimum, 1.0 - maximum
        "sample_timestep_respacing": "150", see available modes IFBaseModule.respacing_modes or use custom

    :param optional dict if_II_kwargs:
        "dynamic_thresholding_p": 0.95, [0.5, 1.0] it controls color saturation on high cfg values
        "dynamic_thresholding_c": 1.0, [1.0, 15.0] clips the limiter to avoid greyish images on high limiter values
        "guidance_scale": 4.0, [1.0, 20.0] control the amount of texture and details in the final image
        "aug_level": 0.25, [0.0, 1.0] adds additional augmentation to generate more realistic images
        "positive_mixer": 0.5, [0.0, 1.0] contribution of the second positive prompt, 0.0 - minimum, 1.0 - maximum
        "sample_timestep_respacing": "smart50", see available modes IFBaseModule.respacing_modes or use custom

    :param deepfloyd_if.modules.IFStageI if_I: obj
    :param deepfloyd_if.modules.IFStageII if_II: obj
    :param deepfloyd_if.modules.IFStageIII if_III: obj
    :param deepfloyd_if.modules.T5Embedder t5: obj

    :param int seed: int, in case None will use random value
    :param aspect_ratio:
    :param str prompt: text hint/description
    :param str style_prompt: text hint/description for style
    :param str negative_prompt: text hint/description for negative prompt, will use it as unconditional emb
    :param progress:
    :return:
    """
    if seed is None:
        seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))
    if_I.seed_everything(seed)

    if isinstance(prompt, str):
        prompt = [prompt]

    t5_embs = t5.get_text_embeddings(prompt)

    if_I_kwargs = if_I_kwargs or {}
    if_I_kwargs['seed'] = seed
    if_I_kwargs['t5_embs'] = t5_embs
    if_I_kwargs['aspect_ratio'] = aspect_ratio
    if_I_kwargs['progress'] = progress

    if style_prompt is not None:
        if isinstance(style_prompt, str):
            style_prompt = [style_prompt]
        style_t5_embs = t5.get_text_embeddings(style_prompt)
        if_I_kwargs['style_t5_embs'] = style_t5_embs
        if_I_kwargs['positive_t5_embs'] = style_t5_embs

    if negative_prompt is not None:
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        negative_t5_embs = t5.get_text_embeddings(negative_prompt)
        if_I_kwargs['negative_t5_embs'] = negative_t5_embs

    stageI_generations, _ = if_I.embeddings_to_image(**if_I_kwargs)
    pil_images_I = if_I.to_images(stageI_generations)

    result = {'I': pil_images_I}

    if if_II is not None:
        if_II_kwargs = if_II_kwargs or {}
        if_II_kwargs['low_res'] = stageI_generations
        if_II_kwargs['seed'] = seed
        if_II_kwargs['t5_embs'] = t5_embs
        if_II_kwargs['progress'] = progress
        if_II_kwargs['style_t5_embs'] = if_I_kwargs.get('style_t5_embs')
        if_II_kwargs['positive_t5_embs'] = if_I_kwargs.get('positive_t5_embs')

        stageII_generations, _meta = if_II.embeddings_to_image(**if_II_kwargs)
        pil_images_II = if_II.to_images(stageII_generations)

        result['II'] = pil_images_II
    else:
        stageII_generations = None

    if if_II is not None and if_III is not None:
        if_III_kwargs = if_III_kwargs or {}

        stageIII_generations = []
        for idx in range(len(stageII_generations)):
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
        pil_images_III = if_III.to_images(stageIII_generations)

        result['III'] = pil_images_III
    else:
        stageIII_generations = None

    if return_tensors:
        return result, (stageI_generations, stageII_generations, stageIII_generations)
    else:
        return result
