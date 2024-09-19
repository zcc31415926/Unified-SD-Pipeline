import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, \
    StableDiffusionXLImg2ImgPipeline

from .sd import SD


class SDXL(SD):
    '''
    Stable Diffusion (SD) XL Pipeline
    Defaults:
        num_steps (int, default: 50): Number of sampling steps
        img_size (int, default: 1024): Size of the generated images
        guidance_scale (float, default: 5.0): Scale coefficient of the classfier-free guidance in sampling
        Downscaling coefficient of the VAE of the pipeline (int, default: 8)
        Number of latent channels (int, default: 4)
    '''
    def __init__(self, model_id, num_steps, img_size,
                 use_ddim_scheduler, fp16, device, refiner_id):
        super().__init__(model_id, num_steps, img_size,
                         use_ddim_scheduler, fp16, device)
        self.refiner_id = refiner_id
        self.initializeModules()

    def initializeModules(self):
        pipeline = super().initializeModules(
            StableDiffusionXLPipeline, {})
        self.tokenizer_2 = pipeline.tokenizer_2
        self.text_encoder_2 = pipeline.text_encoder_2
        ori_size = self.unet.config.sample_size * \
            (2 ** (len(self.vae.config.block_out_channels) - 1))
        self.add_time_ids = torch.Tensor(
            [[ori_size, ori_size, 0, 0, self.img_size, self.img_size]]
        ).to(dtype=self.torch_dtype, device=self.device)
        if self.refiner_id is not None:
            self.refine_pipeline = \
                StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.refiner_id, torch_dtype=self.torch_dtype)
            self.refine_pipeline.to(self.device)

    def getEmb(self, texts):
        if texts is None:
            return None
        # CLIP encoding
        tokens = self.tokenizer(
            texts, padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt')
        emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
        # CLIP-2 encoding
        tokens_2 = self.tokenizer_2(
            texts, padding='max_length',
            max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors='pt')
        emb_2_set = self.text_encoder_2(
            tokens_2.input_ids.to(self.device), output_hidden_states=True)
        emb_2 = emb_2_set.hidden_states[-2]
        emb_2_pool = emb_2_set[0]
        # embedding concatenation
        emb = torch.cat([emb, emb_2], dim=-1)
        return emb, emb_2_pool

    def getNoisePred(self, z, c_pos, c_neg, t):
        if c_neg is None:
            model_input = z
            c = c_pos[0]
            add_text_embeds = c_pos[1]
            add_time_ids = self.add_time_ids
        else:
            model_input = torch.cat([z, z])
            c = torch.cat([c_neg[0], c_pos[0]])
            add_text_embeds = torch.cat([c_neg[1], c_pos[1]])
            add_time_ids = torch.cat([self.add_time_ids] * c.size(0))
        # XXX: bug in diffusers: the official EulerDiscreteScheduler can only do one round of sampling
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py#L292
        self.scheduler._init_step_index(t)
        model_input = self.scheduler.scale_model_input(
            model_input, timestep=t)
        added_cond_kwargs = {'text_embeds': add_text_embeds,
                             'time_ids': add_time_ids}
        return self.unet(model_input, t, encoder_hidden_states=c,
                         added_cond_kwargs=added_cond_kwargs).sample

    def txt2Img(self, z, prompts_pos, prompts_neg,
                guidance_scale, ori_sizes):
        raw_imgs = super().txt2Img(
            z, prompts_pos, prompts_neg, guidance_scale, None)
        if self.refiner_id is not None:
            resized_imgs = []
            for i, raw_img in enumerate(raw_imgs):
                raw_img = Image.fromarray(raw_img)
                img = self.refine_pipeline(
                    prompts_pos[i], image=raw_img).images[0]
                img = np.array(img)
                if ori_sizes is not None:
                    img = removeCenterPad(img, ori_sizes[i])
                resized_imgs.append(img)
            return resized_imgs
        else:
            return raw_imgs

