import gradio as gr
import numpy as np
import random


def _update_result_view(show_gallery: bool) -> tuple[dict, dict]:
    return gr.update(visible=show_gallery), gr.update(visible=not show_gallery)


def show_gallery_view() -> tuple[dict, dict]:
    return _update_result_view(True)


def show_upscaled_view() -> tuple[dict, dict]:
    return _update_result_view(False)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)
    return seed


def update_upscale_button(selected_index: int) -> tuple[dict, dict]:
    if selected_index == -1:
        return gr.update(interactive=False), gr.update(interactive=False)
    else:
        return gr.update(interactive=True), gr.update(interactive=True)


def get_stage2_index(evt: gr.SelectData) -> int:
    return evt.index


def check_if_stage2_selected(index: int) -> None:
    if index == -1:
        raise gr.Error(
            'You need to select the image you would like to upscale from the Stage 1 results by clicking.'
        )


def get_device_map(device):
    return {
        'shared': device,
        'encoder.embed_tokens': device,
        'encoder.block.0': device,
        'encoder.block.1': device,
        'encoder.block.2': device,
        'encoder.block.3': device,
        'encoder.block.4': device,
        'encoder.block.5': device,
        'encoder.block.6': device,
        'encoder.block.7': device,
        'encoder.block.8': device,
        'encoder.block.9': device,
        'encoder.block.10': device,
        'encoder.block.11': device,
        'encoder.block.12': 'cpu',
        'encoder.block.13': 'cpu',
        'encoder.block.14': 'cpu',
        'encoder.block.15': 'cpu',
        'encoder.block.16': 'cpu',
        'encoder.block.17': 'cpu',
        'encoder.block.18': 'cpu',
        'encoder.block.19': 'cpu',
        'encoder.block.20': 'cpu',
        'encoder.block.21': 'cpu',
        'encoder.block.22': 'cpu',
        'encoder.block.23': 'cpu',
        'encoder.final_layer_norm': 'cpu',
        'encoder.dropout': 'cpu',
    }
