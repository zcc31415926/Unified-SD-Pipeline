import torch
from diffusers import StableDiffusionPipeline

from .sd import SD


class SD12(SD):
    '''
    Stable Diffusion (SD) v1 and v2 Pipeline
    Defaults:
        num_steps (int, default: 50): Number of sampling steps
        img_size (int, default: 512): Size of the generated images
        guidance_scale (float, default: 7.5): Scale coefficient of the classfier-free guidance in sampling
        Downscaling coefficient of the VAE of the pipeline (int, default: 8)
        Number of latent channels (int, default: 4)
    '''
    def __init__(self, model_id, num_steps, img_size,
                 use_ddim_scheduler, fp16, device):
        super().__init__(model_id, num_steps, img_size,
                         use_ddim_scheduler, fp16, device)
        self.initializeModules(StableDiffusionPipeline, {})

    def getEmb(self, texts):
        if texts is None:
            return None
        # CLIP encoding
        tokens = self.tokenizer(
            texts, padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt')
        return self.text_encoder(tokens.input_ids.to(self.device))[0]

    def getNoisePred(self, z, c_pos, c_neg, t):
        if c_neg is None:
            model_input = z
            c = c_pos
        else:
            model_input = torch.cat([z, z])
            c = torch.cat([c_neg, c_pos])
        model_input = self.scheduler.scale_model_input(
            model_input, timestep=t)
        return self.unet(model_input, t, encoder_hidden_states=c).sample

