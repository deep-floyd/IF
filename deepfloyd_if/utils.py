# -*- coding: utf-8 -*-
from os.path import abspath, dirname, join

import torch
import numpy as np
from PIL import Image, ImageFilter

RESOURCES_ROOT = join(abspath(dirname(__file__)), 'resources')


def drop_shadow(image, offset=(5, 5), background=0xffffff, shadow=0x444444, border=8, iterations=3):
    """
    Drop shadows with PIL.
    Author: Kevin Schluff
    License: Python license
    https://code.activestate.com/recipes/474116/

    Add a gaussian blur drop shadow to an image.

    image       - The image to overlay on top of the shadow.
    offset      - Offset of the shadow from the image as an (x,y) tuple.  Can be
                  positive or negative.
    background  - Background colour behind the image.
    shadow      - Shadow colour (darkness).
    border      - Width of the border around the image.  This must be wide
                  enough to account for the blurring of the shadow.
    iterations  - Number of times to apply the filter.  More iterations
                  produce a more blurred shadow, but increase processing time.
    """

    # Create the backdrop image -- a box in the background colour with a
    # shadow on it.
    total_width = image.size[0] + abs(offset[0]) + 2 * border
    total_height = image.size[1] + abs(offset[1]) + 2 * border
    back = Image.new(image.mode, (total_width, total_height), background)

    # Place the shadow, taking into account the offset from the image
    shadow_left = border + max(offset[0], 0)
    shadow_top = border + max(offset[1], 0)
    back.paste(shadow, [shadow_left, shadow_top, shadow_left + image.size[0], shadow_top + image.size[1]])

    # Apply the filter to blur the edges of the shadow.  Since a small kernel
    # is used, the filter must be applied repeatedly to get a decent blur.
    n = 0
    while n < iterations:
        back = back.filter(ImageFilter.BLUR)
        n += 1

    # Paste the input image onto the shadow backdrop
    image_left = border - min(offset[0], 0)
    image_top = border - min(offset[1], 0)
    back.paste(image, (image_left, image_top))
    return back


def pil_list_to_torch_tensors(pil_images):
    result = []
    for pil_image in pil_images:
        image = np.array(pil_image, dtype=np.uint8)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).unsqueeze(0)
        result.append(image)
    return torch.cat(result, dim=0)
