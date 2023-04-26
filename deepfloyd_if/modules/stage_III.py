# -*- coding: utf-8 -*-
import accelerate

from .base import IFBaseModule
from ..model import SuperResUNetModel


class IFStageIII(IFBaseModule):

    available_models = ['IF-III-L-v1.0']

    def __init__(self, *args, model_kwargs=None, pil_img_size=1024, **kwargs):
        super().__init__(*args, pil_img_size=pil_img_size, **kwargs)
        model_params = dict(self.conf.params)
        model_params.update(model_kwargs or {})
        with accelerate.init_empty_weights():
            self.model = SuperResUNetModel(low_res_diffusion=self.get_diffusion('1000'), **model_params)
        self.model = self.load_checkpoint(self.model, self.dir_or_name)
        self.model.eval().to(self.device)

    def embeddings_to_image(
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
