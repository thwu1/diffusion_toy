#%%
from scipy.stats import wasserstein_distance

# %%
import numpy as np
from scipy.stats import norm

def generate_gaussian_mixture(means, stds, weights, num_samples=1000):
    """
    Generate samples from a mixture of Gaussian distributions.
    
    Args:
        means (list): List of means for each Gaussian component
        stds (list): List of standard deviations for each component
        weights (list): List of weights for each component (should sum to 1)
        num_samples (int): Number of samples to generate
        
    Returns:
        np.array: Samples drawn from the Gaussian mixture
    """
    if not (len(means) == len(stds) == len(weights)):
        raise ValueError("means, stds, and weights must have same length")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("weights must sum to 1")
        
    # Choose which Gaussian to sample from based on weights
    components = np.random.choice(len(means), size=num_samples, p=weights)
    
    # Generate samples
    samples = np.zeros(num_samples)
    for i in range(len(means)):
        mask = components == i
        samples[mask] = np.random.normal(means[i], stds[i], size=mask.sum())
        
    return samples

# %%


# Generate a mixture of two Gaussians
means = [0, 3]  # Two peaks at x=0 and x=3
stds = [1, 0.5]  # Different spreads
weights = [0.6, 0.4]  # 60% from first Gaussian, 40% from second

samples = generate_gaussian_mixture(means, stds, weights, num_samples=10000)
samples2 = generate_gaussian_mixture([-1, 4], [1.2, 1], weights, num_samples=10000)

# Plot the distributions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue', label='Samples 1')
plt.hist(samples2, bins=50, density=True, alpha=0.7, color='lightgreen', label='Samples 2')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Comparison of Two Gaussian Mixtures')

# Add the true mixture density curves for comparison
x = np.linspace(min(min(samples), min(samples2)), max(max(samples), max(samples2)), 1000)
mixture_pdf = weights[0] * norm.pdf(x, means[0], stds[0]) + \
             weights[1] * norm.pdf(x, means[1], stds[1])
mixture_pdf2 = weights[0] * norm.pdf(x, -1, 1.2) + \
             weights[1] * norm.pdf(x, 4, 1)
plt.plot(x, mixture_pdf, 'r-', lw=2, label='True Density 1')
plt.plot(x, mixture_pdf2, 'b-', lw=2, label='True Density 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
wasserstein_distance(samples, samples2)

# %%
from scipy.stats import wasserstein_distance

def wasserstein_p_distance(samples1, samples2, p=1):
    """
    Calculate the p-Wasserstein distance between two sets of samples.
    For p=1, this is equivalent to the Earth Mover's Distance (EMD).
    
    Args:
        samples1: Array of samples from first distribution
        samples2: Array of samples from second distribution 
        p: Order of the Wasserstein distance (default=1)
    
    Returns:
        float: The p-Wasserstein distance
    """
    # For p=1, we can use scipy's implementation
    if p == 1:
        return wasserstein_distance(samples1, samples2)
    
    # For p>1, we need to sort the samples and compute manually
    # This implements the 1D Wasserstein distance formula
    samples1_sorted = np.sort(samples1)
    samples2_sorted = np.sort(samples2)
    
    # Get quantile points
    n = len(samples1_sorted)
    m = len(samples2_sorted)
    quantiles = np.linspace(0, 1, 1000)
    
    # Interpolate points from both distributions
    points1 = np.quantile(samples1_sorted, quantiles)
    points2 = np.quantile(samples2_sorted, quantiles)
    
    # Calculate p-Wasserstein distance
    return np.power(np.mean(np.power(np.abs(points1 - points2), p)), 1/p)

# Example usage:
print(f"1-Wasserstein distance: {wasserstein_p_distance(samples, samples2, p=1):.4f}")
print(f"2-Wasserstein distance: {wasserstein_p_distance(samples, samples2, p=2):.4f}")


# %%

np.std(samples)
# %%
wasserstein_p_distance(samples, [np.mean(samples)], p=2)

# %%
wasserstein_p_distance(samples, [np.median(samples)], p=1)