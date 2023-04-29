# -*- coding: utf-8 -*-
import os
import random
import platform
from datetime import datetime

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from accelerate.utils import set_module_tensor_to_device

from .. import utils
from ..model.respace import create_gaussian_diffusion
from .utils import load_model_weights, predict_proba, clip_process_generations


class IFBaseModule:
    stage = '-'

    available_models = []
    cpu_zero_emb = np.load(os.path.join(utils.RESOURCES_ROOT, 'zero_t5-v1_1-xxl_vector.npy'))
    cpu_zero_emb = torch.from_numpy(cpu_zero_emb)

    respacing_modes = {
        'fast27': '10,10,3,2,2',
        'smart27': '7,4,2,1,2,4,7',
        'smart50': '10,6,4,3,2,2,3,4,6,10',
        'smart100': '1,1,1,1,2,2,2,2,2,2,3,3,4,4,5,5,6,7,7,8,9,10,13',
        'smart185': '1,1,2,2,2,3,3,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20',
        'super27': '1,1,1,1,1,1,1,2,5,13',  # for III super-res
        'super40': '2,2,2,2,2,2,3,4,6,15',  # for III super-res
        'super100': '4,4,6,6,8,8,10,10,14,30',  # for III super-res
    }

    wm_pil_img = Image.open(os.path.join(utils.RESOURCES_ROOT, 'wm.png'))

    try:
        import clip  # noqa
    except ModuleNotFoundError:
        print('Warning! You should install CLIP: "pip install git+https://github.com/openai/CLIP.git --no-deps"')
        raise

    clip_model, clip_preprocess = clip.load('ViT-L/14', device='cpu')
    clip_model.eval()

    cpu_w_weights, cpu_w_biases = load_model_weights(os.path.join(utils.RESOURCES_ROOT, 'w_head_v1.npz'))
    cpu_p_weights, cpu_p_biases = load_model_weights(os.path.join(utils.RESOURCES_ROOT, 'p_head_v1.npz'))
    w_threshold, p_threshold = 0.5, 0.5

    def __init__(self, dir_or_name, device, pil_img_size=256, cache_dir=None, hf_token=None):
        self.hf_token = hf_token
        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/IF_')
        self.dir_or_name = dir_or_name
        self.conf = self.load_conf(dir_or_name) if not self.use_diffusers else None
        self.device = torch.device(device)
        self.zero_emb = self.cpu_zero_emb.clone().to(self.device)
        self.pil_img_size = pil_img_size

    @property
    def use_diffusers(self):
        return False

    def embeddings_to_image(
            self, t5_embs, low_res=None, *,
            style_t5_embs=None,
            positive_t5_embs=None,
            negative_t5_embs=None,
            batch_repeat=1,
            dynamic_thresholding_p=0.95,
            sample_loop='ddpm',
            sample_timestep_respacing='smart185',
            dynamic_thresholding_c=1.5,
            guidance_scale=7.0,
            aug_level=0.25,
            positive_mixer=0.15,
            blur_sigma=None,
            img_size=None,
            img_scale=4.0,
            aspect_ratio='1:1',
            progress=True,
            seed=None,
            sample_fn=None,
            support_noise=None,
            support_noise_less_qsample_steps=0,
            inpainting_mask=None,
            device=None,
            **kwargs,
    ):
        if device is None:
            device = self.model.primary_device
        self._clear_cache()
        image_w, image_h = self._get_image_sizes(low_res, img_size, aspect_ratio, img_scale)
        diffusion = self.get_diffusion(sample_timestep_respacing)

        bs_scale = 2 if positive_t5_embs is None else 3

        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // bs_scale]
            combined = torch.cat([half] * bs_scale, dim=0).to(device)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            if bs_scale == 3:
                cond_eps, pos_cond_eps, uncond_eps = torch.split(eps, len(eps) // bs_scale, dim=0)
                half_eps = uncond_eps + guidance_scale * (
                        cond_eps * (1 - positive_mixer) + pos_cond_eps * positive_mixer - uncond_eps)
                pos_half_eps = uncond_eps + guidance_scale * (pos_cond_eps - uncond_eps)
                eps = torch.cat([half_eps, pos_half_eps, half_eps], dim=0)
            else:
                cond_eps, uncond_eps = torch.split(eps, len(eps) // bs_scale, dim=0)
                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        seed = self.seed_everything(seed)

        text_emb = t5_embs.to(self.device, dtype=self.model.dtype).repeat(batch_repeat, 1, 1)
        batch_size = text_emb.shape[0] * batch_repeat

        if positive_t5_embs is not None:
            positive_t5_embs = positive_t5_embs.to(self.device, dtype=self.model.dtype).repeat(batch_repeat, 1, 1)

        if negative_t5_embs is not None:
            negative_t5_embs = negative_t5_embs.to(self.device, dtype=self.model.dtype).repeat(batch_repeat, 1, 1)

        timestep_text_emb = None
        if style_t5_embs is not None:
            list_timestep_text_emb = [
                style_t5_embs.to(self.device, dtype=self.model.dtype).repeat(batch_repeat, 1, 1),
            ]
            if positive_t5_embs is not None:
                list_timestep_text_emb.append(positive_t5_embs)
            if negative_t5_embs is not None:
                list_timestep_text_emb.append(negative_t5_embs)
            else:
                list_timestep_text_emb.append(
                    self.zero_emb.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device, dtype=self.model.dtype))
            timestep_text_emb = torch.cat(list_timestep_text_emb, dim=0).to(self.device, dtype=self.model.dtype)

        metadata = {
            'seed': seed,
            'guidance_scale': guidance_scale,
            'dynamic_thresholding_p': dynamic_thresholding_p,
            'dynamic_thresholding_c': dynamic_thresholding_c,
            'batch_size': batch_size,
            'device_name': self.device_name,
            'img_size': [image_w, image_h],
            'sample_loop': sample_loop,
            'sample_timestep_respacing': sample_timestep_respacing,
            'stage': self.stage,
        }

        list_text_emb = [t5_embs.to(self.device)]
        if positive_t5_embs is not None:
            list_text_emb.append(positive_t5_embs.to(self.device))
        if negative_t5_embs is not None:
            list_text_emb.append(negative_t5_embs.to(self.device))
        else:
            list_text_emb.append(
                self.zero_emb.unsqueeze(0).repeat(batch_size, 1, 1).to(self.device, dtype=self.model.dtype))

        model_kwargs = dict(
            text_emb=torch.cat(list_text_emb, dim=0).to(self.device, dtype=self.model.dtype),
            timestep_text_emb=timestep_text_emb,
            use_cache=True,
        )
        if low_res is not None:
            if blur_sigma is not None:
                low_res = T.GaussianBlur(3, sigma=(blur_sigma, blur_sigma))(low_res)
            model_kwargs['low_res'] = torch.cat([low_res] * bs_scale, dim=0).to(self.device)
            model_kwargs['aug_level'] = aug_level

        if support_noise is None:
            noise = torch.randn(
                (batch_size * bs_scale, 3, image_h, image_w), device=self.device, dtype=self.model.dtype)
        else:
            assert support_noise_less_qsample_steps < len(diffusion.timestep_map) - 1
            assert support_noise.shape == (1, 3, image_h, image_w)
            q_sample_steps = torch.tensor([int(len(diffusion.timestep_map) - 1 - support_noise_less_qsample_steps)])
            support_noise = support_noise.cpu()
            noise = support_noise.clone()
            noise[inpainting_mask.cpu().bool() if inpainting_mask is not None else ...] = diffusion.q_sample(
                support_noise[inpainting_mask.cpu().bool() if inpainting_mask is not None else ...],
                q_sample_steps,
            )
            noise = noise.repeat(batch_size * bs_scale, 1, 1, 1).to(device=self.device, dtype=self.model.dtype)

        if inpainting_mask is not None:
            inpainting_mask = inpainting_mask.to(device=self.device, dtype=torch.long)

        if sample_loop == 'ddpm':
            with torch.no_grad():
                sample = diffusion.p_sample_loop(
                    model_fn,
                    (batch_size * bs_scale, 3, image_h, image_w),
                    noise=noise,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    dynamic_thresholding_p=dynamic_thresholding_p,
                    dynamic_thresholding_c=dynamic_thresholding_c,
                    inpainting_mask=inpainting_mask,
                    device=device,
                    progress=progress,
                    sample_fn=sample_fn,
                )[:batch_size]
        elif sample_loop == 'ddim':
            with torch.no_grad():
                sample = diffusion.ddim_sample_loop(
                    model_fn,
                    (batch_size * bs_scale, 3, image_h, image_w),
                    noise=noise,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    dynamic_thresholding_p=dynamic_thresholding_p,
                    dynamic_thresholding_c=dynamic_thresholding_c,
                    device=device,
                    progress=progress,
                    sample_fn=sample_fn,
                )[:batch_size]
        else:
            raise ValueError(f'Sample loop "{sample_loop}" doesnt support')

        sample = self.__validate_generations(sample)
        self._clear_cache()

        return sample, metadata

    def load_conf(self, dir_or_name, filename='config.yml'):
        path = self._get_path_or_download_file_from_hf(dir_or_name, filename)
        conf = OmegaConf.load(path)
        return conf

    def load_checkpoint(self, model, dir_or_name, filename='pytorch_model.bin'):
        path = self._get_path_or_download_file_from_hf(dir_or_name, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            param_device = 'cpu'
            for param_name, param in checkpoint.items():
                set_module_tensor_to_device(model, param_name, param_device, value=param)
        else:
            print(f'Warning! In directory "{dir_or_name}" filename "pytorch_model.bin" is not found.')
        return model

    def _get_path_or_download_file_from_hf(self, dir_or_name, filename):
        if dir_or_name in self.available_models:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            hf_hub_download(repo_id=f'DeepFloyd/{dir_or_name}', filename=filename, cache_dir=cache_dir,
                            force_filename=filename, token=self.hf_token)
            return os.path.join(cache_dir, filename)
        else:
            return os.path.join(dir_or_name, filename)

    def get_diffusion(self, timestep_respacing):
        timestep_respacing = self.respacing_modes.get(timestep_respacing, timestep_respacing)
        diffusion = create_gaussian_diffusion(
            steps=1000,
            learn_sigma=True,
            sigma_small=False,
            noise_schedule='cosine',
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
            timestep_respacing=timestep_respacing,
        )
        return diffusion

    @staticmethod
    def seed_everything(seed=None):
        if seed is None:
            seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        return seed

    def device_name(self):
        if self.device.type == 'cpu':
            return 'cpu_' + str(platform.processor())
        if self.device.type == 'cuda':
            return torch.cuda.get_device_name(self.device)
        return '-'

    def to_images(self, generations, disable_watermark=False):
        bs, c, h, w = generations.shape
        coef = min(h / self.pil_img_size, w / self.pil_img_size)
        img_h, img_w = (int(h / coef), int(w / coef)) if coef < 1 else (h, w)

        S1, S2 = 1024 ** 2, img_w * img_h
        K = (S2 / S1) ** 0.5
        wm_size, wm_x, wm_y = int(K * 62), img_w - int(14 * K), img_h - int(14 * K)

        wm_img = self.wm_pil_img.resize(
            (wm_size, wm_size), getattr(Image, 'Resampling', Image).BICUBIC, reducing_gap=None)

        pil_images = []
        for image in ((generations + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu():
            pil_img = torchvision.transforms.functional.to_pil_image(image).convert('RGB')
            pil_img = pil_img.resize((img_w, img_h), getattr(Image, 'Resampling', Image).NEAREST)
            if not disable_watermark:
                pil_img.paste(wm_img, box=(wm_x - wm_size, wm_y - wm_size, wm_x, wm_y), mask=wm_img.split()[-1])
            pil_images.append(pil_img)
        return pil_images

    def show(self, pil_images, nrow=None, size=10):
        if nrow is None:
            nrow = round(len(pil_images) ** 0.5)

        imgs = torchvision.utils.make_grid(utils.pil_list_to_torch_tensors(pil_images), nrow=nrow)
        if not isinstance(imgs, list):
            imgs = [imgs.cpu()]

        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(size, size))
        for i, img in enumerate(imgs):
            img = img.detach()
            img = torchvision.transforms.functional.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        fix.show()
        plt.show()

    def _clear_cache(self):
        self.model.cache = None

    def _get_image_sizes(self, low_res, img_size, aspect_ratio, img_scale):
        if low_res is not None:
            bs, c, h, w = low_res.shape
            image_h, image_w = int((h * img_scale) // 32) * 32, int((w * img_scale // 32)) * 32
        else:
            scale_w, scale_h = aspect_ratio.split(':')
            scale_w, scale_h = int(scale_w), int(scale_h)
            coef = scale_w / scale_h
            image_h, image_w = img_size, img_size
            if coef >= 1:
                image_w = int(round(img_size / 8 * coef) * 8)
            else:
                image_h = int(round(img_size / 8 / coef) * 8)

        assert image_h % 8 == 0
        assert image_w % 8 == 0

        return image_w, image_h

    def __validate_generations(self, generations):
        with torch.no_grad():
            imgs = clip_process_generations(generations)
            image_features = self.clip_model.encode_image(imgs.to('cpu'))
            image_features = image_features.detach().cpu().numpy().astype(np.float16)
            p_pred = predict_proba(image_features, self.cpu_p_weights, self.cpu_p_biases)
            w_pred = predict_proba(image_features, self.cpu_w_weights, self.cpu_w_biases)
            query = p_pred > self.p_threshold
            if query.sum() > 0:
                generations[query] = T.GaussianBlur(99, sigma=(100.0, 100.0))(generations[query])
            query = w_pred > self.w_threshold
            if query.sum() > 0:
                generations[query] = T.GaussianBlur(99, sigma=(100.0, 100.0))(generations[query])
        return generations
