from .sd12 import SD12
from .sdxl import SDXL
from .sd3 import SD3
from .flux import Flux


def SD(model_id, num_steps, img_size,
       use_ddim_scheduler, fp16, device, **kwargs):
    if 'stable-diffusion-xl' in model_id.lower():
        Pipeline = SDXL
    elif 'stable-diffusion-3' in model_id.lower():
        Pipeline = SD3
    elif 'flux' in model_id.lower():
        Pipeline = Flux
    else:
        Pipeline = SD12
    return Pipeline(model_id, num_steps, img_size,
                    use_ddim_scheduler, fp16, device, **kwargs)

