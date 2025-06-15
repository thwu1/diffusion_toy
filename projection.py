from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
from typing import Union, Tuple
import torch
from sklearn.isotonic import IsotonicRegression

def _apply_isotonic_single(args):
    """
    Helper function to apply isotonic regression to a single sequence.
    This function is designed to be used with executor.map().
    
    Args:
        args: tuple of (sequence_data, seq_length)
        
    Returns:
        projected sequence as numpy array
    """
    sequence_data, seq_length = args
    
    # Apply isotonic regression
    iso_reg = IsotonicRegression(increasing=True)
    x = torch.arange(seq_length, dtype=torch.float32).numpy()
    
    # Fit and predict
    projected_y = iso_reg.fit_transform(x, sequence_data)
    return projected_y

def monotonic_proj(sequences, executor=None):
    """
    L2 projection onto monotonic sequences using isotonic regression.
    
    Args:
        sequences: torch.Tensor of shape (batch_size, channels, seq_length)
        executor: Optional concurrent.futures.Executor for parallel processing
    
    Returns:
        projected sequences with same shape
    """
    
    batch_size, channels, seq_length = sequences.shape
    projected = torch.zeros_like(sequences)

    if batch_size < 16:
        executor = None
    
    if executor is None:
        # Sequential processing (original implementation)
        for b in range(batch_size):
            for c in range(channels):
                # Apply isotonic regression to each channel independently
                iso_reg = IsotonicRegression(increasing=True)
                # Use range as x-coordinates for isotonic regression
                x = torch.arange(seq_length, dtype=torch.float32)
                y = sequences[b, c, :].cpu().numpy()
                
                # Fit and predict
                projected_y = iso_reg.fit_transform(x.numpy(), y)
                projected[b, c, :] = torch.from_numpy(projected_y)
    else:
        # Parallel processing using executor
        # Prepare arguments for parallel execution
        args_list = []
        indices_list = []
        
        for b in range(batch_size):
            for c in range(channels):
                sequence_data = sequences[b, c, :].cpu().numpy()
                args_list.append((sequence_data, seq_length))
                indices_list.append((b, c))
        
        # Execute in parallel
        results = list(executor.map(_apply_isotonic_single, args_list))
        
        # Assign results back to projected tensor
        for (b, c), result in zip(indices_list, results):
            projected[b, c, :] = torch.from_numpy(result)
    
    return projected


def step_with_proj(
    self,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    generator=None,
    return_dict: bool = True,
    executor=None,
) -> Union[DDPMSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).

    Args:
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    t = timestep

    prev_t = self.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://huggingface.co/papers/2006.11239
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )
    
    pred_original_sample = pred_original_sample.detach()

    pred_original_sample[:, 0:1,:] = monotonic_proj(pred_original_sample[:, 0:1,:], executor=executor)

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://huggingface.co/papers/2006.11239
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://huggingface.co/papers/2006.11239
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    # 6. Add noise
    variance = 0
    if t > 0:
        device = model_output.device
        variance_noise = randn_tensor(
            model_output.shape, generator=generator, device=device, dtype=model_output.dtype
        )
        if self.variance_type == "fixed_small_log":
            variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
        elif self.variance_type == "learned_range":
            variance = self._get_variance(t, predicted_variance=predicted_variance)
            variance = torch.exp(0.5 * variance) * variance_noise
        else:
            variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

    pred_prev_sample = pred_prev_sample + variance

    if not return_dict:
        return (
            pred_prev_sample,
            pred_original_sample,
        )

    return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
