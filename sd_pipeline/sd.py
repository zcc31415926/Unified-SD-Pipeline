import torch
import numpy as np
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler

from .utils import CenterPad, removeCenterPad
from .scheduling import scheduling, invScheduling


class SD:
    def __init__(self, model_id, num_steps, img_size,
                 use_ddim_scheduler, fp16, device):
        self.model_id = model_id
        self.num_steps = num_steps
        self.img_size = img_size
        self.use_ddim_scheduler = use_ddim_scheduler
        self.torch_dtype = torch.float16 if fp16 else torch.float32
        self.device = device
        self.transforms = transforms.Compose([
            CenterPad(0),
            transforms.Resize([self.img_size, self.img_size]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)])

    def initializeModules(self, Pipeline, kwargs):
        pipeline = Pipeline.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, **kwargs)
        # pipeline.to(self.device)
        self.vae_scale_factor = pipeline.vae_scale_factor
        # UNet is replaced by Transformer in SD3
        self.unet = pipeline.unet \
            if hasattr(pipeline, 'unet') else pipeline.transformer
        if torch.__version__ < '2.0':
            self.unet.enable_xformers_memory_efficient_attention()
        self.latent_channels = self.unet.config.in_channels
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.vae = pipeline.vae
        self.vae.float()
        print(f'SD version: {self.model_id}')
        # EulerDiscreteScheduler (SDXL) does not support accurate inversion
        # FlowMatchEulerDiscreteScheduler (SD3) does not support accurate inversion
        if self.use_ddim_scheduler:
            self.scheduler = DDIMScheduler.from_pretrained(
                self.model_id, subfolder='scheduler')
            assert self.scheduler.config.num_train_timesteps % \
                self.num_steps == 0, 'The number of inference steps ' + \
                'must be a divisor of that used in training'
            print('Replacing default ' +
                  f'{self.scheduler.__class__.__name__} by DDIMScheduler')
        else:
            self.scheduler = pipeline.scheduler
            print(f'Using default {self.scheduler.__class__.__name__}')
        if hasattr(self.scheduler.config, 'prediction_type'):
            print('UNet & scheduler prediction type: ' +
                  self.scheduler.config.prediction_type)
        self.scheduler.set_timesteps(self.num_steps)
        self.scheduler.timesteps = \
            self.scheduler.timesteps.to(self.device)
        return pipeline

    def image2Latent(self, img):
        img = img.to(self.device)
        assert img.dim() == 4, 'batch dimension required'
        z0 = self.vae.encode(img).latent_dist.mean
        if self.vae.config.shift_factor is not None:
            z0 -= self.vae.config.shift_factor
        z0 *= self.vae.config.scaling_factor
        return z0.to(self.torch_dtype)

    def latent2Image(self, z):
        z = z.float() / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            z += self.vae.config.shift_factor
        img = self.vae.decode(z).sample / 2 + 0.5
        img = torch.clamp(img, 0, 1).permute(0, 2, 3, 1).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return img

    def getEmb(self, texts):
        raise NotImplementedError

    def step(self, z, c_pos, c_neg, t, guidance_scale, inv):
        e = self.getNoisePred(z, c_pos, c_neg, t)
        if c_neg is not None:
            uncond_e, cond_e = e.chunk(2)
            e = uncond_e + guidance_scale * (cond_e - uncond_e)
        return invScheduling(self.scheduler, e, t, z) \
            if inv else scheduling(self.scheduler, e, t, z)

    def getNoisePred(self, z, c_pos, c_neg, t):
        if c_neg is None:
            model_input = z
            c = c_pos
        else:
            model_input = torch.cat([z, z])
            c = torch.cat([c_neg, c_pos])
        return self.unet(model_input, t, encoder_hidden_states=c).sample

    def txt2Img(self, z, prompts_pos, prompts_neg,
                guidance_scale, ori_sizes):
        assert isinstance(prompts_pos, list), 'batch dimension required'
        assert prompts_neg is None or isinstance(prompts_neg, list), \
            'batch dimension required'
        if z is None:
            z = torch.randn(
                len(prompts_pos), self.latent_channels,
                self.img_size // self.vae_scale_factor,
                self.img_size // self.vae_scale_factor
            ).to(dtype=self.torch_dtype, device=self.device)
            if hasattr(self.scheduler, 'init_noise_sigma'):
                z *= self.scheduler.init_noise_sigma
        c_pos = self.getEmb(prompts_pos)
        c_neg = self.getEmb(prompts_neg)
        for t in self.scheduler.timesteps:
            z = self.step(z, c_pos, c_neg, t, guidance_scale, inv=False)
        imgs = self.latent2Image(z)
        if ori_sizes is not None:
            size = ori_sizes[i] \
                if isinstance(ori_sizes[0], list) else ori_sizes
            resized_imgs = []
            for i, img in enumerate(imgs):
                img = removeCenterPad(img, size)
                resized_imgs.append(img)
            return resized_imgs
        else:
            return imgs

    def inversion(self, imgs, prompts_pos, prompts_neg, guidance_scale):
        assert isinstance(prompts_pos, list), 'batch dimension required'
        assert prompts_neg is None or isinstance(prompts_neg, list), \
            'batch dimension required'
        c_pos = self.getEmb(prompts_pos)
        c_neg = self.getEmb(prompts_neg)
        imgs = [self.transforms(img).unsqueeze(0) for img in imgs]
        imgs = torch.cat(imgs, dim=0).to(self.device)
        z = self.image2Latent(imgs)
        for t in torch.flip(self.scheduler.timesteps, dims=[0]):
            z = self.step(z, c_pos, c_neg, t, guidance_scale, inv=True)
        return z

