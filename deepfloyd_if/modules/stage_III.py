# -*- coding: utf-8 -*-
from diffusers import DiffusionPipeline, DDPMScheduler
import torch
import accelerate
import os

from .base import IFBaseModule


class IFStageIII(IFBaseModule):

    available_models = ['IF-III-L-v1.0', 'stable-diffusion-x4-upscaler']

    def __init__(self, dir_or_name, device, model_kwargs=None, pil_img_size=1024, **kwargs):
        super().__init__(dir_or_name, device, pil_img_size=pil_img_size, **kwargs)
        if not self.use_diffusers:
            model_params = dict(self.conf.params)
            model_params.update(model_kwargs or {})
            with accelerate.init_empty_weights():
                self.model = SuperResUNetModel(low_res_diffusion=self.get_diffusion('1000'), **model_params)
            self.model = self.load_checkpoint(self.model, self.dir_or_name)
            self.model.eval().to(self.device)
        else:
            model_id = os.path.join("stabilityai", self.dir_or_name)

            model_kwargs = model_kwargs or {}
            precision = str(model_kwargs.get("precision", "16"))
            if precision == '16':
                torch_dtype = torch.float16
            elif precision == 'bf16':
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            self.model = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, token=self.hf_token)
            self.model.to(self.device)

            # make sure to use xformers if version is smaller than 2.0.0
            if bool(os.environ.get("FORCE_MEM_EFFICIENT_ATTN")):
                self.model.enable_xformers_memory_efficient_attention()

    @property
    def use_diffusers(self):
        if self.dir_or_name == self.available_models[-1]:
            return True
        elif os.path.isdir(self.dir_or_name) and os.path.isfile(os.path.join(self.dir_or_name, "model_index.json")):
            return True
        return False

    def embeddings_to_image(self, *args, **kwargs):
        if not self.use_diffusers:
            return self._embeddings_to_image(*args, **kwargs)
        else:
            return self._embeddings_to_image_diffusers(*args, **kwargs)

    def _embeddings_to_image(
        self, low_res, t5_embs, style_t5_embs=None, positive_t5_embs=None, negative_t5_embs=None, batch_repeat=1,
        aug_level=0.0, blur_sigma=None, dynamic_thresholding_p=0.95, dynamic_thresholding_c=1.0, positive_mixer=0.5,
        sample_loop='ddpm', sample_timestep_respacing='super40', guidance_scale=4.0, img_scale=4.0,
        progress=True, seed=None, sample_fn=None, **kwargs):
        return super().embeddings_to_image(
            t5_embs=t5_embs,
            low_res=low_res,
            style_t5_embs=style_t5_embs,
            positive_t5_embs=positive_t5_embs,
            negative_t5_embs=negative_t5_embs,
            batch_repeat=batch_repeat,
            aug_level=aug_level,
            blur_sigma=blur_sigma,
            dynamic_thresholding_p=dynamic_thresholding_p,
            dynamic_thresholding_c=dynamic_thresholding_c,
            sample_loop=sample_loop,
            sample_timestep_respacing=sample_timestep_respacing,
            guidance_scale=guidance_scale,
            positive_mixer=positive_mixer,
            img_size=1024,
            img_scale=img_scale,
            progress=progress,
            seed=seed,
            sample_fn=sample_fn,
            **kwargs
        )

    def _embeddings_to_image_diffusers(
        self, low_res, t5_embs, style_t5_embs=None, positive_t5_embs=None, negative_t5_embs=None, batch_repeat=1,
        aug_level=0.0, blur_sigma=None, dynamic_thresholding_p=0.95, dynamic_thresholding_c=1.0, positive_mixer=0.5,
        sample_loop='ddpm', sample_timestep_respacing='75', guidance_scale=4.0, img_scale=4.0,
        progress=True, seed=None, sample_fn=None, **kwargs):

        prompt = kwargs.pop("prompt")
        noise_level = kwargs.pop("noise_level", 20)

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
            "output_type": "pt",
        }

        images = self.model(**metadata).images

        sample = self._IFBaseModule__validate_generations(images)

        return sample, metadata

