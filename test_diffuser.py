# %%
import torch
import numpy as np
from diffusers import DDIMScheduler, DDPMPipeline
from tqdm import tqdm
import types
from projection import step_with_proj, monotonic_proj
# ddpm = DDPMPipeline.from_pretrained("anton-l/ddpm-ema-flowers-64").to("cuda")
ddpm = DDPMPipeline.from_pretrained("/home/tianhao/diffusion_toy/1dmonotonic_2data").to("cuda")
#%%
# ddpm.scheduler = DDIMScheduler.from_config(ddpm.scheduler.config)
# ddpm.scheduler.config.clip_sample =  False
# image = ddpm(num_inference_steps=25)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
ddpm.scheduler.set_timesteps(1000)
ddpm.scheduler.config.clip_sample = True
ddpm.scheduler.step_with_proj = types.MethodType(step_with_proj, ddpm.scheduler)


executor = ProcessPoolExecutor(max_workers=4)
# executor = None
# executor = ThreadPoolExecutor(max_workers=4)

def sample_image(bs):
    image = torch.randn(bs, 2, 32).to("cuda")
    for t in tqdm(ddpm.scheduler.timesteps):
        eps = ddpm.unet(image, t.unsqueeze(0)).sample
        image = ddpm.scheduler.step(eps, t, image).prev_sample

    image = image.detach().cpu().numpy()
    return image

def sample_image_with_proj(bs):
    image = torch.randn(bs, 2, 32).to("cuda")
    for t in tqdm(ddpm.scheduler.timesteps):
        eps = ddpm.unet(image, t.unsqueeze(0)).sample
        image = ddpm.scheduler.step_with_proj(eps, t, image, executor=executor).prev_sample

    image[:, 0:1, :] = monotonic_proj(image[:, 0:1, :], executor=executor)
    image = image.detach().cpu().numpy()
    return image

# image = sample_image_with_proj(1024)
# image = sample_image(1)
# image.shape

from dataset1d import Dataset1D
dataset = Dataset1D()
gt1 = dataset.data1  # Already a torch tensor
gt2 = dataset.data2  # Already a torch tensor

# Convert to numpy for consistency with generated samples, then back to torch
gt1_tensor = gt1.float()
gt2_tensor = gt2.float()

print(f"gt1 shape: {gt1_tensor.shape}")
print(f"gt2 shape: {gt2_tensor.shape}")

# Generate multiple samples for evaluation
def evaluate_model_performance(num_samples=128, use_projection=False):
    """
    Evaluate model performance by calculating L2 distance to ground truth.
    
    Args:
        num_samples: Number of samples to generate
        use_projection: Whether to use monotonic projection
    
    Returns:
        mean_min_distance: Average minimum L2 distance to gt1 or gt2
    """
    if use_projection:
        samples = sample_image_with_proj(num_samples)
    else:
        samples = sample_image(num_samples)
    
    # Convert to torch tensor
    samples_tensor = torch.from_numpy(samples).float()
    
    print(f"Generated samples shape: {samples_tensor.shape}")
    
    # Calculate L2 distances
    distances_to_gt1 = []
    distances_to_gt2 = []
    min_distances = []
    
    for i in range(num_samples):
        sample = samples_tensor[i]  # Shape: (2, 32)
        
        # Calculate L2 distance to gt1
        dist_gt1 = torch.norm(sample - gt1_tensor, p=2).item()
        
        # Calculate L2 distance to gt2  
        dist_gt2 = torch.norm(sample - gt2_tensor, p=2).item()
        
        # Take minimum distance
        min_dist = min(dist_gt1, dist_gt2)
        
        distances_to_gt1.append(dist_gt1)
        distances_to_gt2.append(dist_gt2)
        min_distances.append(min_dist)
    
    # Calculate statistics
    mean_min_distance = np.mean(min_distances)
    std_min_distance = np.std(min_distances)
    mean_dist_gt1 = np.mean(distances_to_gt1)
    mean_dist_gt2 = np.mean(distances_to_gt2)
    
    print(f"\nResults for {num_samples} samples ({'with' if use_projection else 'without'} projection):")
    print(f"Mean minimum L2 distance: {mean_min_distance:.4f} ± {std_min_distance:.4f}")
    print(f"Mean distance to gt1: {mean_dist_gt1:.4f}")
    print(f"Mean distance to gt2: {mean_dist_gt2:.4f}")
    
    # Count which ground truth is closer more often
    closer_to_gt1 = sum(1 for d1, d2 in zip(distances_to_gt1, distances_to_gt2) if d1 < d2)
    closer_to_gt2 = num_samples - closer_to_gt1
    
    print(f"Samples closer to gt1: {closer_to_gt1} ({100*closer_to_gt1/num_samples:.1f}%)")
    print(f"Samples closer to gt2: {closer_to_gt2} ({100*closer_to_gt2/num_samples:.1f}%)")
    
    return mean_min_distance, min_distances

# Evaluate without projection
print("="*50)
print("EVALUATION WITHOUT MONOTONIC PROJECTION")
print("="*50)
mean_dist_no_proj, distances_no_proj = evaluate_model_performance(num_samples=1024, use_projection=False)

# Evaluate with projection
print("\n" + "="*50)
print("EVALUATION WITH MONOTONIC PROJECTION")
print("="*50)
mean_dist_with_proj, distances_with_proj = evaluate_model_performance(num_samples=1024, use_projection=True)

# Compare results
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"Mean distance without projection: {mean_dist_no_proj:.4f}")
print(f"Mean distance with projection: {mean_dist_with_proj:.4f}")
improvement = mean_dist_no_proj - mean_dist_with_proj
print(f"Improvement with projection: {improvement:.4f} ({100*improvement/mean_dist_no_proj:.2f}%)")

# Optional: Plot histogram of distances
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(distances_no_proj, bins=20, alpha=0.7, label='Without projection')
plt.hist(distances_with_proj, bins=20, alpha=0.7, label='With projection')
plt.xlabel('L2 Distance to closest GT')
plt.ylabel('Frequency')
plt.title('Distribution of L2 Distances')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(range(len(distances_no_proj)), distances_no_proj, alpha=0.6, label='Without projection', s=10)
plt.scatter(range(len(distances_with_proj)), distances_with_proj, alpha=0.6, label='With projection', s=10)
plt.xlabel('Sample Index')
plt.ylabel('L2 Distance to closest GT')
plt.title('L2 Distances per Sample')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate and print some example comparisons
print("\n" + "="*30)
print("EXAMPLE SAMPLE COMPARISONS")
print("="*30)
for i in range(min(5, len(distances_no_proj))):
    print(f"Sample {i}: No proj = {distances_no_proj[i]:.4f}, With proj = {distances_with_proj[i]:.4f}")


# %%


def is_monotonic_rate(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if data.dim() == 2:
        data = data.unsqueeze(0)
    assert data.dim() == 3
    rate = torch.sum(data[:, 0, :-1] > data[:, 0, 1:] + 1e-3, axis=1)
    rate = (rate <= 0).float()
    return rate.mean().item()

# sampled_data = sample_image_with_proj(1024)
is_monotonic_rate(sampled_data)

# %%
def invalid_data(data):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if data.dim() == 2:
        data = data.unsqueeze(0)
    assert data.dim() == 3
    rate = torch.sum(data[:, 0, :-1] > data[:, 0, 1:], axis=1)
    rate = (rate > 0).float()
    return rate

def separate_valid_invalid(data):
    """
    Separate data into valid and invalid samples.
    """
    if isinstance(data, np.ndarray):
        data_tensor = torch.from_numpy(data)
    else:
        data_tensor = data
    
    invalid_mask = invalid_data(data_tensor)
    valid_mask = (invalid_mask == 0.0)
    
    valid_samples = data[valid_mask.numpy()] if isinstance(data, np.ndarray) else data[valid_mask]
    invalid_samples = data[invalid_mask.bool().numpy()] if isinstance(data, np.ndarray) else data[invalid_mask.bool()]
    
    return valid_samples, invalid_samples, valid_mask, invalid_mask.bool()

# Usage:
valid_samples, invalid_samples, valid_mask, invalid_mask = separate_valid_invalid(sampled_data)
print(f"Original shape: {sampled_data.shape}")
print(f"Valid samples shape: {valid_samples.shape}")


# %%
invalid_samples[0]