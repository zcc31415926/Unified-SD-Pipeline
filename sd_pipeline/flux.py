import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline

from .sd import SD
from .utils import removeCenterPad
from .scheduling import scheduling, invScheduling


class Flux(SD):
    '''
    Stable Diffusion (SD) Flux Pipeline
    Defaults:
        num_steps (int, default: 4): Number of sampling steps
        img_size (int, default: 1024): Size of the generated images
        guidance_scale (float, default: 0.0, disabled): Guidance offset to the time embedding in sampling
            Different from `guidance_scale` in classifier-free guidance
        Downscaling coefficient of the VAE of the pipeline (int, default: 8)
        Number of latent channels (int, default: 16)
    '''
    def __init__(self, model_id, num_steps, img_size, use_ddim_scheduler,
                 fp16, device, max_seq_len, use_guidance):
        super().__init__(model_id, num_steps, img_size,
                         use_ddim_scheduler, fp16, device)
        self.max_seq_len = max_seq_len
        self.use_guidance = use_guidance
        self.initializeModules()

    def initializeModules(self):
        pipeline = super().initializeModules(FluxPipeline, {})
        self.tokenizer_2 = pipeline.tokenizer_2
        self.text_encoder_2 = pipeline.text_encoder_2
        sigmas = np.linspace(1, 1 / self.num_steps, self.num_steps)
        self.scheduler.set_timesteps(sigmas=sigmas)
        self.scheduler.timesteps = \
            self.scheduler.timesteps.to(self.device)
        self.scheduler.sigmas = self.scheduler.sigmas.to(self.device)
        self.img_ids = self.getImgIds()

    def getImgIds(self):
        latent_size = self.img_size // self.vae_scale_factor
        img_ids = torch.zeros(latent_size, latent_size, 3)
        img_ids[..., 1] += torch.arange(latent_size).unsqueeze(-1)
        img_ids[..., 2] += torch.arange(latent_size).unsqueeze(0)
        return img_ids.contiguous().view(1, -1, 3).to(
            dtype=self.torch_dtype, device=self.device)

    def img2Seq(self, z):
        n, c, h, w = z.size()
        z = z.contiguous().view(n, c, h // 2, 2, w // 2, 2)
        z = z.permute(0, 2, 4, 1, 3, 5)
        return z.contiguous().view(n, (h * w) // 4, c * 4)

    def seq2Img(self, z):
        n, d, c = z.size()
        c = c // 4
        h = w = int((d * 4) ** 0.5)
        z = z.contiguous().view(n, h // 2, w // 2, c, 2, 2)
        z = z.permute(0, 3, 1, 4, 2, 5)
        return z.contiguous().view(n, c, h, w)

    def getEmb(self, texts):
        if texts is None:
            return None
        # CLIP encoding
        tokens = self.tokenizer(
            texts, padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_overflowing_tokens=False,
            return_length=False, return_tensors='pt')
        emb_set = self.text_encoder(tokens.input_ids.to(self.device))
        emb_pool = emb_set.pooler_output
        # T5 encoding
        tokens_2 = self.tokenizer_2(
            texts, padding='max_length', max_length=self.max_seq_len,
            truncation=True, return_length=False,
            return_overflowing_tokens=False, return_tensors='pt')
        emb_2 = self.text_encoder_2(tokens_2.input_ids.to(self.device))[0]
        text_ids = torch.zeros(len(texts), self.max_seq_len, 3).to(
            dtype=self.torch_dtype, device=self.device)
        return emb_2, emb_pool, text_ids

    def step(self, z, c, t, guidance_scale, inv):
        e = self.getNoisePred(z, c, t, guidance_scale)
        return invScheduling(self.scheduler, e, t, z) \
            if inv else scheduling(self.scheduler, e, t, z)

    def getNoisePred(self, z, c, t, guidance_scale):
        guidance = torch.full([1], guidance_scale, device=self.device,
                              dtype=torch.float32) \
            if self.use_guidance else None
        img_ids = torch.cat([self.img_ids] * z.size(0), dim=0)
        self.scheduler._init_step_index(t.float())
        ts = t.expand(z.size(0)).float().to(self.device)
        return self.unet(hidden_states=z, timestep=ts / \
                             self.scheduler.config.num_train_timesteps,
                         guidance=guidance, pooled_projections=c[1],
                         encoder_hidden_states=c[0],
                         txt_ids=c[2], img_ids=img_ids).sample

    # FluxPipeline does not support negative prompts
    def txt2Img(self, z, prompts_pos, prompts_neg,
                guidance_scale, ori_sizes):
        assert isinstance(prompts_pos, list), 'batch dimension required'
        if z is None:
            z = torch.randn(
                len(prompts_pos),
                self.img_size ** 2 // self.vae_scale_factor ** 2,
                self.latent_channels).to(
                dtype=self.torch_dtype, device=self.device)
        c = self.getEmb(prompts_pos)
        for t in self.scheduler.timesteps:
            z = self.step(z, c, t, guidance_scale, inv=False)
        z = self.seq2Img(z)
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

    # FluxPipeline does not support negative prompts
    def inversion(self, imgs, prompts_pos, prompts_neg, guidance_scale):
        assert isinstance(prompts_pos, list), 'batch dimension required'
        c = self.getEmb(prompts_pos)
        imgs = [self.transforms(img).unsqueeze(0) for img in imgs]
        imgs = torch.cat(imgs, dim=0).to(self.device)
        z = self.image2Latent(imgs)
        z = self.img2Seq(z)
        for t in torch.flip(self.scheduler.timesteps, dims=[0]):
            z = self.step(z, c, t, guidance_scale, inv=True)
        return z

