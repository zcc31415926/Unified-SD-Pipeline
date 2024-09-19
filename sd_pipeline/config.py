from easydict import EasyDict


def getModelCfg(model_type, use_ddim_scheduler, fp16, device):
    config = EasyDict({
        'use_ddim_scheduler': use_ddim_scheduler,
        'fp16': fp16,
        'device': device
    })
    if model_type == 'sd1':
        # config.model_id = 'CompVis/stable-diffusion-v1-4'
        config.model_id = 'runwayml/stable-diffusion-v1-5'
        config.img_size = 512
        config.num_steps = 50
    elif model_type == 'sd2':
        config.model_id = 'stabilityai/stable-diffusion-2-1'
        config.img_size = 768
        config.num_steps = 50
    elif model_type == 'sdxl':
        config.model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
        config.img_size = 1024
        config.num_steps = 50
        # config.refiner_id = 'stabilityai/stable-diffusion-xl-refiner-1.0'
        config.refiner_id = None
    elif model_type == 'sd3':
        config.model_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
        config.img_size = 1024
        config.num_steps = 28
        config.t5 = False
        config.max_seq_len = 256
    elif model_type == 'flux':
        config.model_id = 'black-forest-labs/FLUX.1-schnell'
        config.img_size = 1024
        config.num_steps = 4
        config.max_seq_len = 256
        config.use_guidance = False
    else:
        raise NotImplementedError
    return config


def getGuidanceScale(model_type):
    if model_type in ['sd1', 'sd2']:
        return 7.5
    elif model_type == 'sdxl':
        return 5
    elif model_type == 'sd3':
        return 7
    elif model_type == 'flux':
        return 0
    else:
        raise NotImplementedError

