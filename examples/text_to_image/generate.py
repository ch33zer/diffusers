from diffusers import AutoPipelineForText2Image
import torch
import sys
prompt = sys.argv[1:]
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipeline.load_lora_weights("lily_saved_model", weight_name="pytorch_lora_weights.safetensors")

image = pipeline(prompt).images[0]
image.save("example.png")