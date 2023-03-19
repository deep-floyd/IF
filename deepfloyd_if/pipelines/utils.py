# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image


def _prepare_pil_image(raw_pil_img, img_size):
    raw_pil_img = raw_pil_img.convert('RGB')
    w, h = raw_pil_img.size
    coef = w / h
    image_h, image_w = img_size, img_size
    if coef >= 1:
        image_w = int(round(img_size / 8 * coef) * 8)
    else:
        image_h = int(round(img_size / 8 / coef) * 8)

    pil_img = raw_pil_img.resize(
        (image_w, image_h), resample=getattr(Image, 'Resampling', Image).BICUBIC, reducing_gap=None
    )
    img = np.array(pil_img)
    img = img.astype(np.float32) / 127.5 - 1
    img = np.transpose(img, [2, 0, 1])
    img = torch.from_numpy(img).unsqueeze(0)
    return img
