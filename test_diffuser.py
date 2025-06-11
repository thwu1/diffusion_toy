# %%
import torch

from diffusers import DDIMScheduler, DDPMPipeline

# ddpm = DDPMPipeline.from_pretrained("anton-l/ddpm-ema-flowers-64").to("cuda")
ddpm = DDPMPipeline.from_pretrained("/home/tianhao/ddpm-singledata-1d-layer2").to("cuda")
#%%
# ddpm.scheduler = DDIMScheduler.from_config(ddpm.scheduler.config)
# ddpm.scheduler.config.clip_sample =  False
# image = ddpm(num_inference_steps=25)

ddpm.scheduler.set_timesteps(100)
ddpm.scheduler.config.clip_sample = False
image = torch.randn(1, 2, 32).to("cuda")
for t in ddpm.scheduler.timesteps:
    # print(t)
    eps = ddpm.unet(image, t.unsqueeze(0)).sample
    image = ddpm.scheduler.step(eps, t, image).prev_sample

image = image.detach().cpu().numpy()

# %%
# Assuming 'image' is your tensor with shape (batch, channels, height, width)
import matplotlib.pyplot as plt

img_to_plot = plt.scatter(image[0][0], image[0][1])
plt.imshow(img_to_plot)
# %%
from datasets import load_dataset

mnist = load_dataset("mnist")
mnist
# %%
image = torch.randn(1, 1, 64, 64).to("cuda")
for t in ddpm.scheduler.timesteps:
    eps = ddpm.unet(image, t).sample
    image = ddpm.scheduler.step(eps, t, image).prev_sample

image