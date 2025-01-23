import gradio as gr
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import dream


# If the default space is insufficient, set a custom cache location for the model
#config_path = '/root/autodl-tmp/transformers-cache/'
config_path = None
device = 'cuda:0'

# Initialize the model
if_I = IFStageI('IF-I-XL-v1.0', device=device, cache_dir=config_path)
if_II = IFStageII('IF-II-L-v1.0', device=device, cache_dir=config_path)
if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device, cache_dir=config_path)
t5 = T5Embedder(device="cpu", cache_dir=config_path)

# Define a function to generate images
def generate_images(prompt, count, seed):
    result = dream(
        t5=t5, if_I=if_I, if_II=if_II, if_III=if_III,
        prompt=[prompt] * count,
        seed=seed,
        if_I_kwargs={
            "guidance_scale": 7.0,
            "sample_timestep_respacing": "smart100",
        },
        if_II_kwargs={
            "guidance_scale": 4.0,
            "sample_timestep_respacing": "smart50",
        },
        if_III_kwargs={
            "guidance_scale": 9.0,
            "noise_level": 20,
            "sample_timestep_respacing": "75",
        },
    )

    # Return the generated image object directly
    images = []
    for stage, stage_images in result.items():
        #if stage != 'II':
        #    continue
        for image in stage_images:
            images.append(image)

    return images

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here"),
        gr.Slider(label="Count", minimum=1, maximum=10, value=4),
        gr.Number(label="Seed", value=42)
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="DeepFloyd IF Image Generator",
    description="Generate images based on your prompt using DeepFloyd IF model.",
)

# Specify the IP and port for startup
iface.launch(server_name="127.0.0.1", server_port=6006)