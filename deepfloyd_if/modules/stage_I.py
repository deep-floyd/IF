# -*- coding: utf-8 -*-
from .base import IFBaseModule
from ..model import UNetModel


class IFStageI(IFBaseModule):
    stage = 'I'
    available_models = ['IF-I-M-v0.1rc', 'IF-I-L-v0.1rc', 'IF-I-L-v0.2rc', 'IF-I-IF-v1.0']

    def __init__(self, *args, model_kwargs=None, pil_img_size=256, **kwargs):
        """
        :param conf_or_path:
        :param device:
        :param cache_dir:
        :param use_auth_token:
        """
        super().__init__(*args, pil_img_size=pil_img_size, **kwargs)
        model_params = dict(self.conf.params)
        model_params.update(model_kwargs or {})
        self.model = UNetModel(**model_params).eval().to(self.device)
        self.model = self.load_checkpoint(self.model, self.dir_or_name)

    def embeddings_to_image(self, t5_embs, style_t5_embs=None, positive_t5_embs=None, negative_t5_embs=None,
                            batch_repeat=1, dynamic_thresholding_p=0.95, sample_loop='ddpm', positive_mixer=0.25,
                            sample_timestep_respacing='150', dynamic_thresholding_c=1.5, guidance_scale=7.0,
                            aspect_ratio='1:1', progress=True, seed=None, sample_fn=None, **kwargs):

        return super().embeddings_to_image(
            t5_embs=t5_embs,
            style_t5_embs=style_t5_embs,
            positive_t5_embs=positive_t5_embs,
            negative_t5_embs=negative_t5_embs,
            batch_repeat=batch_repeat,
            dynamic_thresholding_p=dynamic_thresholding_p,
            dynamic_thresholding_c=dynamic_thresholding_c,
            sample_loop=sample_loop,
            sample_timestep_respacing=sample_timestep_respacing,
            guidance_scale=guidance_scale,
            img_size=64,
            aspect_ratio=aspect_ratio,
            progress=progress,
            seed=seed,
            sample_fn=sample_fn,
            positive_mixer=positive_mixer,
            **kwargs
        )
