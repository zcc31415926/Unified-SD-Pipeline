import torch
from diffusers import DDIMScheduler, PNDMScheduler, \
    EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler


def scheduling(scheduler, e, t, z):
    if isinstance(scheduler, DDIMScheduler):
        return DDIMScheduling(scheduler, e, t, z)
    elif isinstance(scheduler, PNDMScheduler):
        return PNDMScheduling(scheduler, e, t, z)
    elif isinstance(scheduler, EulerDiscreteScheduler):
        return EulerDiscreteScheduling(scheduler, e, t, z)
    elif isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return FlowMatchEulerDiscreteScheduling(scheduler, e, t, z)
    else:
        raise NotImplementedError


def invScheduling(scheduler, e, t, z):
    if isinstance(scheduler, DDIMScheduler):
        return DDIMInvScheduling(scheduler, e, t, z)
    elif isinstance(scheduler, PNDMScheduler):
        return PNDMInvScheduling(scheduler, e, t, z)
    elif isinstance(scheduler, EulerDiscreteScheduler):
        return EulerDiscreteInvScheduling(scheduler, e, t, z)
    elif isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        return FlowMatchEulerDiscreteInvScheduling(scheduler, e, t, z)
    else:
        raise NotImplementedError


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py#L342
def DDIMScheduling(scheduler, e, t, z):
    return scheduler.step(e, t, z).prev_sample
    dtype = e.dtype
    e = e.to(torch.float32)
    z = z.to(torch.float32)
    prev_t = t - scheduler.config.num_train_timesteps // \
        scheduler.num_inference_steps
    alpha_t = scheduler.alphas_cumprod[t]
    prev_alpha_t = scheduler.alphas_cumprod[max(prev_t, 0)]
    beta_t = 1 - alpha_t
    if scheduler.config.prediction_type == 'epsilon':
        pred_ori = (z - beta_t ** 0.5 * e) / alpha_t ** 0.5
        pred_e = e
    elif scheduler.config.prediction_type == 'sample':
        pred_ori = e
        pred_e = (z - alpha_t ** 0.5 * pred_ori) / beta_t ** 0.5
    else:
        pred_ori = alpha_t ** 0.5 * z - beta_t ** 0.5 * e
        pred_e = alpha_t ** 0.5 * e + beta_t ** 0.5 * z
    pred_dir = (1 - prev_alpha_t) ** 0.5 * pred_e
    prev_z = prev_alpha_t ** 0.5 * pred_ori + pred_dir
    return prev_z.to(dtype)


def DDIMInvScheduling(scheduler, e, t, z):
    dtype = e.dtype
    e = e.to(torch.float32)
    z = z.to(torch.float32)
    next_t = t + scheduler.config.num_train_timesteps // \
        scheduler.num_inference_steps
    alpha_t = scheduler.alphas_cumprod[t]
    next_alpha_t = scheduler.alphas_cumprod[
        min(next_t, scheduler.config.num_train_timesteps - 1)]
    beta_t = 1 - alpha_t
    if scheduler.config.prediction_type == 'epsilon':
        pred_ori = (z - beta_t ** 0.5 * e) / alpha_t ** 0.5
        pred_e = e
    elif scheduler.config.prediction_type == 'sample':
        pred_ori = e
        pred_e = (z - alpha_t ** 0.5 * pred_ori) / beta_t ** 0.5
    else:
        pred_ori = alpha_t ** 0.5 * z - beta_t ** 0.5 * e
        pred_e = alpha_t ** 0.5 * e + beta_t ** 0.5 * z
    pred_dir = (1 - next_alpha_t) ** 0.5 * pred_e
    next_z = next_alpha_t ** 0.5 * pred_ori + pred_dir
    return next_z.to(dtype)


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py#L226
# XXX: step_prk not implemented
def PNDMScheduling(scheduler, e, t, z):
    prev_t = t - scheduler.config.num_train_timesteps // \
        scheduler.num_inference_steps
    if scheduler.counter != 1:
        scheduler.ets = scheduler.ets[-3 :]
        scheduler.ets.append(e)
    else:
        prev_t = t
        t = t + scheduler.config.num_train_timesteps // \
            scheduler.num_inference_steps
    if len(scheduler.ets) == 1 and scheduler.counter == 0:
        scheduler.cur_sample = z
    elif len(scheduler.ets) == 1 and scheduler.counter == 1:
        e = (e + scheduler.ets[-1]) / 2
        z = scheduler.cur_sample
        scheduler.cur_sample = None
    elif len(scheduler.ets) == 2:
        e = (3 * scheduler.ets[-1] - scheduler.ets[-2]) / 2
    elif len(scheduler.ets) == 3:
        e = (23 * scheduler.ets[-1] - 16 * scheduler.ets[-2] +
             5 * scheduler.ets[-3]) / 12
    else:
        e = (55 * scheduler.ets[-1] - 59 * scheduler.ets[-2] +
             37 * scheduler.ets[-3] - 9 * scheduler.ets[-4]) / 24
    alpha_t = scheduler.alphas_cumprod[t]
    prev_alpha_t = scheduler.alphas_cumprod[prev_t] \
        if prev_t >= 0 else scheduler.final_alpha_cumprod
    beta_t = 1 - alpha_t
    prev_beta_t = 1 - prev_alpha_t
    if scheduler.config.prediction_type == 'v_prediction':
        e = alpha_t ** 0.5 * e + beta_t ** 0.5 * z
    z_coeff = (prev_alpha_t / alpha_t) ** 0.5
    e_coeff = alpha_t * prev_beta_t ** 0.5 + \
        (alpha_t * beta_t * prev_alpha_t) ** 0.5
    prev_z = z_coeff * z - (prev_alpha_t - alpha_t) * e / e_coeff
    scheduler.counter += 1
    return prev_z


# XXX: inversion of PNDMScheduler not implemented
def PNDMInvScheduling(scheduler, e, t, z):
    raise NotImplementedError


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py#L493
def EulerDiscreteScheduling(scheduler, e, t, z):
    dtype = e.dtype
    e = e.to(torch.float32)
    z = z.to(torch.float32)
    sigma = scheduler.sigmas[scheduler.step_index]
    if scheduler.config.prediction_type in ['original_sample', 'sample']:
        pred_ori = e
    elif scheduler.config.prediction_type == 'epsilon':
        pred_ori = z - sigma * e
    else:
        pred_ori = -sigma * e / (sigma ** 2 + 1) ** 0.5 + \
            z / (sigma ** 2 + 1)
    derivative = (z - pred_ori) / sigma
    dt = scheduler.sigmas[scheduler.step_index + 1] - sigma
    prev_z = z + derivative * dt
    return prev_z.to(dtype)


# XXX: temporal approximation of the model output `e` results in large error
def EulerDiscreteInvScheduling(scheduler, e, t, z):
    dtype = e.dtype
    e = e.to(torch.float32)
    z = z.to(torch.float32)
    sigma = scheduler.sigmas[scheduler.step_index + 1]
    if scheduler.config.prediction_type in ['original_sample', 'sample']:
        pred_ori = e
    elif scheduler.config.prediction_type == 'epsilon':
        pred_ori = z - sigma * e
    else:
        pred_ori = -sigma * e / (sigma ** 2 + 1) ** 0.5 + \
            z / (sigma ** 2 + 1)
    if sigma == 0:
        if scheduler.config.prediction_type in \
            ['original_sample', 'sample']:
            derivative = (z - e) / scheduler.sigmas[scheduler.step_index]
        elif scheduler.config.prediction_type == 'epsilon':
            derivative = e
        else:
            derivative = 0
    else:
        derivative = (z - pred_ori) / sigma
    dt = scheduler.sigmas[scheduler.step_index] - sigma
    prev_z = z + derivative * dt
    return prev_z.to(dtype)


# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L235
# equal to EulerDiscreteScheduler with mode `epsilon`
def FlowMatchEulerDiscreteScheduling(scheduler, e, t, z):
    dtype = e.dtype
    e = e.to(torch.float32)
    z = z.to(torch.float32)
    sigma = scheduler.sigmas[scheduler.step_index]
    pred_ori = z - sigma * e
    derivative = (z - pred_ori) / sigma
    dt = scheduler.sigmas[scheduler.step_index + 1] - sigma
    prev_z = z + derivative * dt
    return prev_z.to(dtype)


# XXX: temporal approximation of the model output `e` results in huge error
def FlowMatchEulerDiscreteInvScheduling(scheduler, e, t, z):
    dtype = e.dtype
    e = e.to(torch.float32)
    z = z.to(torch.float32)
    sigma = scheduler.sigmas[scheduler.step_index + 1]
    pred_ori = z - sigma * e
    derivative = e if sigma == 0 else (z - pred_ori) / sigma
    dt = scheduler.sigmas[scheduler.step_index] - sigma
    prev_z = z + derivative * dt
    return prev_z.to(dtype)

