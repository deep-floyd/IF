# -*- coding: utf-8 -*-
from diffusers import DiffusionPipeline, DDPMScheduler
import torch
import warnings
import packaging.version as pv
import xformers
import os

from .base import IFBaseModule


class IFStageIII(IFBaseModule):

    available_models = ['stable-diffusion-x4-upscaler']
    use_diffusers = True

    def __init__(self, *args, model_kwargs=None, pil_img_size=1024, precision="16", **kwargs):
        super().__init__(*args, pil_img_size=pil_img_size, **kwargs)
        model_params = model_kwargs or {}

        if self.dir_or_name in self.available_models and self.use_diffusers:
            model_id = os.path.join("stabilityai", self.dir_or_name)
        else:
            model_id = self.dir_or_name

        precision = str(precision)
        if precision == '16':
            torch_dtype = torch.float16
        elif precision == 'bf16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        self.model = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, token=self.hf_token)
        self.model.to(self.device)

        # make sure to use xformers if version is smaller than 2.0.0
        if bool(os.environ.get("FORCE_MEM_EFFICIENT_ATTN")) and pv.parse(torch.__version__) < pv.parse('2.0.0'):
            if pv.parse(xformers.__version__) < pv.parse("0.0.18"):
                warnings.warn(
                    "`xformers` 0.0.18 seems to produce NaN values for large inputs."
                    " If you experience unexpected errors, make sure to upgrade `xformers` as described here:"
                    "https://github.com/facebookresearch/xformers/issues/722."
                )

            self.model.enable_xformers_memory_efficient_attention()


    def embeddings_to_image(
            self, low_res, prompt, batch_repeat=1,
            noise_level=20, blur_sigma=None,
            sample_loop='ddpm', sample_timestep_respacing='75', guidance_scale=9.0, img_scale=4.0,
            progress=True, seed=None, sample_fn=None, **kwargs):

        if sample_loop == "ddpm":
            self.model.scheduler = DDPMScheduler.from_config(self.model.scheduler.config)
        else:
            raise ValueError(f"For now only the 'ddpm' sample loop type is supported, but you passed {sample_loop}")

        num_inference_steps = int(sample_timestep_respacing)

        self.model.set_progress_bar_config(disable=not progress)

        generator = torch.manual_seed(seed)
        prompt = sum([batch_repeat * [p] for p in prompt], [])
        low_res = low_res.repeat(batch_repeat, 1, 1, 1)

        metadata = {
            "image": low_res,
            "prompt": prompt,
            "noise_level": noise_level,
            "generator": generator,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "output_type": "np",
        }

        output = self.model(**metadata)

        images = torch.tensor(output.images, device=low_res.device, dtype=low_res.dtype)
        images = 2 * (images - 0.5)
        images = images.permute(0, 3, 1, 2)

        sample = self._IFBaseModule__validate_generations(images)

        return sample, metadata

