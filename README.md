[![License](https://img.shields.io/badge/License-GNU_GPL-blue.svg)](LICENSE)

### DeepFloyd-IF (Imagen Free)
___


![](./pics/main.jpg)

## Minimum requirements to use all IF models:

- 40GB vRAM/RAM (or 16GB, but `cascade-III` will not be available)
- install xformers and set env variable `FORCE_MEM_EFFICIENT_ATTN=1`


## Quick Start

soon:
```shell
pip install deepfloyd_if==0.0.1
```


### I. Dream

```python
prompt = 'ultra close-up color photo portrait of rainbow owl with deer horns in the woods'
count = 4

result = dream(
    t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
    prompt=[prompt]*count,
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "smart100",
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "smart50",
    },
    if_III_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": "super40",
    },
)
if_I.show(result['I'], size=3)
if_I.show(result['II'], size=6)
if_I.show(result['III'], size=9)
```

![](./pics/dream-I.jpg)

![](./pics/dream-II.jpg)

![](./pics/dream-III.jpg)


## II. Style Transfer

```python
result = style_transfer(
    t5=t5, if_I=if_I, if_II=if_II,
    support_pil_img=raw_pil_image,
    style_prompt=[
        'A fantasy landscape in style lego',
        'A fantasy landscape in style zombie',
        'A fantasy landscape in style origami',
        'A fantasy landscape in style anime',
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 10.0,
        "sample_timestep_respacing": "10,10,10,10,10,10,10,10,0,0",
        'support_noise_less_qsample_steps': 5,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        "sample_timestep_respacing": 'smart50',
        "support_noise_less_qsample_steps": 5,
    },
)
if_I.show(result['II'], 1, 20)
```
![](./pics/style-transfer.jpg)

## III. Super Resolution

`96px --> 1024px`:

![](./pics/super-res-0.jpg)

`384px --> 1024px` with aspect-ratio:

![](./pics/super-res-1.jpg)


### IV. Inpainting

![](./pics/inpainting-mask.jpg)

```python
result = inpainting(
    t5=t5, if_I=if_I,
    if_II=if_II,
    if_III=if_III,
    support_pil_img=raw_pil_image,
    inpainting_mask=inpainting_mask,
    prompt=[
        'blue sunglasses',
        'yellow sunglasses',
        'red sunglasses',
        'green sunglasses',
    ],
    seed=42,
    if_I_kwargs={
        "guidance_scale": 7.0,
        "sample_timestep_respacing": "10,10,10,10,10,0,0,0,0,0",
        'support_noise_less_qsample_steps': 0,
    },
    if_II_kwargs={
        "guidance_scale": 4.0,
        'aug_level': 0.0,
        "sample_timestep_respacing": '100',
    },
    if_III_kwargs={
        "guidance_scale": 4.0,
        'aug_level': 0.0,
        "sample_timestep_respacing": '40',
        'support_noise_less_qsample_steps': 0,
    },
)
if_I.show(result['I'], 2, 3)
if_I.show(result['II'], 2, 6)
if_I.show(result['III'], 2, 14)
```
![](./pics/inpainting.jpg)


## License

The code in this repository is released under the GNU GPL License.

The weights are available via [the DeepFloyd organization at Hugging Face](https://huggingface.co/DeepFloyd).


## Citation

```bibtex
@misc{IF2023,
    title={IF only: a pixel diffusion model with ...},
    author={A Shonenkov and M Konstantinov and D Bakshandaeva and C Schuhmann and R Vencu and D Ha and E Mostaque},
    year={2023},
    eprint={...},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgements

Thanks StabilityAI, LAION and ...

## 🚀 Contributors 🚀
- Thanks, [@Dango233](https://github.com/Dango233), for adaptation IF with xformers memory efficient attention
