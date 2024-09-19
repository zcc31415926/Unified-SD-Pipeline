import torch
from diffusers import StableDiffusion3Pipeline

from .sd import SD


class SD3(SD):
    '''
    Stable Diffusion (SD) 3 Pipeline
    Defaults:
        num_steps (int, default: 28): Number of sampling steps
        img_size (int, default: 1024): Size of the generated images
        guidance_scale (float, default: 7.0): Scale coefficient of the classfier-free guidance in sampling
        Downscaling coefficient of the VAE of the pipeline (int, default: 8)
        Number of latent channels (int, default: 16)
    '''
    def __init__(self, model_id, num_steps, img_size,
                 use_ddim_scheduler, fp16, device, t5, max_seq_len):
        super().__init__(model_id, num_steps, img_size,
                         use_ddim_scheduler, fp16, device)
        self.t5 = t5
        self.max_seq_len = max_seq_len
        self.initializeModules()

    def initializeModules(self):
        t5_kwargs = {'text_encoder_3': None, 'tokenizer_3': None} \
            if not self.t5 else {}
        pipeline = super().initializeModules(
            StableDiffusion3Pipeline, t5_kwargs)
        self.tokenizer_2 = pipeline.tokenizer_2
        self.text_encoder_2 = pipeline.text_encoder_2
        if self.t5:
            self.tokenizer_3 = pipeline.tokenizer_3
            self.text_encoder_3 = pipeline.text_encoder_3

    def getEmb(self, texts):
        if texts is None:
            return None
        # CLIP encoding
        tokens = self.tokenizer(
            texts, padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt')
        emb_set = self.text_encoder(tokens.input_ids.to(self.device),
                                    output_hidden_states=True)
        emb = emb_set.hidden_states[-2]
        emb_pool = emb_set[0]
        # CLIP-2 encoding
        tokens_2 = self.tokenizer_2(
            texts, padding='max_length',
            max_length=self.tokenizer_2.model_max_length,
            truncation=True, return_tensors='pt')
        emb_2_set = self.text_encoder_2(
            tokens_2.input_ids.to(self.device), output_hidden_states=True)
        emb_2 = emb_2_set.hidden_states[-2]
        emb_2_pool = emb_2_set[0]
        # T5 encoding
        if self.t5:
            tokens_3 = self.tokenizer_3(
                texts, padding='max_length', max_length=self.max_seq_len,
                truncation=True, add_special_tokens=True,
                return_tensors='pt')
            emb_3 = self.text_encoder_3(
                tokens_3.input_ids.to(self.device))[0]
        else:
            # XXX: without T5, the output dimensions change from 256 to self.tokenizer.model_max_length (77)
            emb_3 = torch.zeros(
                emb.size(0), self.tokenizer.model_max_length,
                self.unet.config.joint_attention_dim
            ).to(dtype=self.torch_dtype, device=self.device)
        # embedding concatenation
        emb_clip = torch.cat([emb, emb_2], dim=-1)
        emb_clip = torch.nn.functional.pad(
            emb_clip, (0, emb_3.shape[-1] - emb_clip.shape[-1]))
        emb = torch.cat([emb_clip, emb_3], dim=-2)
        emb_pool = torch.cat([emb_pool, emb_2_pool], dim=-1)
        return emb, emb_pool

    def getNoisePred(self, z, c_pos, c_neg, t):
        if c_neg is None:
            model_input = z
            c = c_pos[0]
            c_pool = c_pos[1]
        else:
            model_input = torch.cat([z, z])
            c = torch.cat([c_neg[0], c_pos[0]])
            c_pool = torch.cat([c_neg[1], c_pos[1]])
        # XXX: bug in diffusers: the official FlowMatchEulerDiscreteScheduler can only do one round of sampling
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py#L288
        self.scheduler._init_step_index(t.float())
        ts = t.expand(model_input.size(0)).float().to(self.device)
        return self.unet(
            hidden_states=model_input, timestep=ts,
            encoder_hidden_states=c, pooled_projections=c_pool).sample

