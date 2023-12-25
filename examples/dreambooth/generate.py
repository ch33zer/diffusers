from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipeline.load_lora_weights("lily_saved_model", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("a digital art drawing by sks of an elf, dnd", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("test_lora.png")