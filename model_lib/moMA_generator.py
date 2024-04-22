from typing import List
import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
from model_lib.attention_processor import IPAttnProcessor, IPAttnProcessor_Self, get_mask_from_cross
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
import tqdm


def get_subject_idx(model,prompt,src_subject,device):
    tokenized_prompt = model.tokenizer(prompt,padding="max_length",max_length=model.tokenizer.model_max_length,truncation=True,return_tensors="pt",).to(device)
    input_ids = tokenized_prompt['input_ids']
    src_subject_idxs = []
    for subject,input_id in zip(src_subject,input_ids):
        src_subject_token_id = [model.tokenizer.encode(i, add_special_tokens=False)[0] for i in subject.split(' ')]
        src_subject_idxs = [i for i, x in enumerate(input_id.tolist()) if x in src_subject_token_id]
    return [src_subject_idxs]


def add_function(model):
    @torch.no_grad()
    def generate_with_adapters(
        model,
        prompt_embeds,
        num_inference_steps,
        generator,
        t_range=list(range(0,950)),
    ):
        
        latents = model.prepare_latents(prompt_embeds.shape[0]//2,4,512,512,prompt_embeds.dtype,prompt_embeds.device,generator)

        model.scheduler.set_timesteps(num_inference_steps)

        iterator = tqdm.tqdm(model.scheduler.timesteps)
        mask_ig_prev = None
        for i, t in enumerate(iterator):
            if not t in t_range: 
                model.moMA_generator.toggle_enable_flag('cross')
            else:
                model.moMA_generator.toggle_enable_flag('all')

            latent_model_input = torch.cat([latents] * 2)
            noise_pred = model.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            latents = model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            mask_ig_prev = (get_mask_from_cross(model.unet.attn_processors))[latents.shape[0]:]

            model.moMA_generator.set_self_mask('self','ig',mask_ig_prev)
            model.moMA_generator.set_self_mask('cross',mask=mask_ig_prev.clone().detach())

        image = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
        return image ,mask_ig_prev.repeat(1,3,1,1) if (not mask_ig_prev==None) else None
    model.generate_with_adapters = generate_with_adapters


class ImageProjModel(torch.nn.Module):
    """Projection Model"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MoMA_generator:
    def __init__(self, device,args):
        self.args = args
        self.device = device
        
        noise_scheduler = DDIMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,beta_schedule="scaled_linear",clip_sample=False,set_alpha_to_one=False,steps_offset=1,)
        
        print('Loading VAE: stabilityai--sd-vae-ft-mse...')
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        
        print('Loading StableDiffusion: Realistic_Vision...')
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V4.0_noVAE",
            torch_dtype=torch.bfloat16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
        ).to(self.device)

        self.unet = self.pipe.unet
        add_function(self.pipe)
        self.pipe.moMA_generator = self

        self.set_ip_adapter()
        self.image_proj_model = self.init_proj()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=768,
            clip_embeddings_dim=1024,
            clip_extra_context_tokens=4,
        ).to(self.device, dtype=torch.bfloat16)
        return image_proj_model
        
    def set_ip_adapter(self):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = IPAttnProcessor_Self(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,scale=1.0,num_tokens=4).to(self.device, dtype=torch.float16)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,scale=1.0,num_tokens=4).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)

    @torch.inference_mode()
    def get_image_embeds_CFG(self, llava_emb):
        clip_image_embeds = llava_emb
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def get_image_crossAttn_feature(
            self,
            llava_emb,
            num_samples=1,
    ):
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_CFG(llava_emb)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        return image_prompt_embeds, uncond_image_prompt_embeds

    # feature are from self-attention layers of Unet: feed reference image to Unet with t=0
    def get_image_selfAttn_feature(
            self,
            pil_image,
            prompt,
    ):  
        self.toggle_enable_flag('self')
        self.toggle_extract_inject_flag('self', 'extract')
        tokenized_prompt = self.pipe.tokenizer(prompt,padding="max_length",truncation=True,return_tensors="pt",).to(self.device)
        text_embeddings = self.pipe.text_encoder(input_ids=tokenized_prompt.input_ids)[0]

        ref_image = pil_image
        ref_image.to(self.device)

        with torch.no_grad(): latents = self.pipe.vae.encode(ref_image).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.tensor([0],device=latents.device).long() # fixed to 0
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timesteps)
        
        _ = self.unet(noisy_latents,timestep=timesteps,encoder_hidden_states=text_embeddings)["sample"]
        # features are stored in attn_processors

        return None
    
    @torch.no_grad()
    def generate_with_MoMA(
        self,
        batch,
        llava_emb=None,
        seed=None,
        device='cuda',
    ):
        self.reset_all()
        img_ig,mask_id,subject,prompt = batch['image'].half().to(device),batch['mask'].half().to(device),batch['label'][0],batch['text'][0]

        prompt = [f"photo of a {subject}. "+ prompt]
        subject_idx = get_subject_idx(self.pipe,prompt,[subject],self.device)
        negative_prompt = None 
            
        # get context-cross-attention feature (from MLLM decoder)
        cond_llava_embeds, uncond_llava_embeds = self.get_image_crossAttn_feature(llava_emb,num_samples=1)
        # get subject-cross-attention feature (from Unet)
        self.get_image_selfAttn_feature(img_ig,subject) # features are stored in attn_processors

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, cond_llava_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_llava_embeds], dim=1)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        
        self.set_self_mask('eraseAll')
        self.toggle_enable_flag('all')
        self.toggle_extract_inject_flag('all','masked_generation')
        self.set_self_mask('self','id',mask_id) 
        self.set_cross_subject_idxs(subject_idx)
        
        images, mask = self.pipe.generate_with_adapters(
            self.pipe,
            prompt_embeds,
            50,
            generator,
        )
        images = torch.clip((images+1)/2.0,min=0.0,max=1.0)

        return images.cpu(), mask.cpu()
    
    def set_selfAttn_strength(self, strength):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = 1.0
            if isinstance(attn_processor, IPAttnProcessor_Self):
                attn_processor.scale = strength

    def set_cross_subject_idxs(self, subject_idxs):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.subject_idxs = subject_idxs

    def set_self_mask(self,mode,id_ig='', mask=None): #only have effect on self attn of the generation process
        for attn_processor in self.unet.attn_processors.values():
            if mode == 'eraseAll':
                if isinstance(attn_processor, IPAttnProcessor_Self):
                    attn_processor.mask_id,attn_processor.mask_ig = None,None
                if isinstance(attn_processor, IPAttnProcessor):
                    attn_processor.mask_i, attn_processor.mask_ig_prev = None, None
            if mode == 'self':
                if isinstance(attn_processor, IPAttnProcessor_Self):
                    if id_ig == 'id':attn_processor.mask_id = mask
                    if id_ig == 'ig':attn_processor.mask_ig = mask
            if mode == 'cross':
                if isinstance(attn_processor, IPAttnProcessor):
                    attn_processor.mask_ig_prev = mask
    
    def toggle_enable_flag(self, processor_enable_mode):
        for attn_processor in self.unet.attn_processors.values():
            if processor_enable_mode == 'cross':
                if isinstance(attn_processor, IPAttnProcessor):attn_processor.enabled = True
                if isinstance(attn_processor, IPAttnProcessor_Self):attn_processor.enabled = False
            if processor_enable_mode == 'self':
                if isinstance(attn_processor, IPAttnProcessor):attn_processor.enabled = False
                if isinstance(attn_processor, IPAttnProcessor_Self):attn_processor.enabled = True
            if processor_enable_mode == 'all':
                attn_processor.enabled = True
            if processor_enable_mode == 'none':
                attn_processor.enabled = False

    def toggle_extract_inject_flag(self, processor_name, mode): # mode: str, 'extract' or 'inject' or 'both'(cross only)
        for attn_processor in self.unet.attn_processors.values():
            if processor_name == 'cross':
                if isinstance(attn_processor, IPAttnProcessor):attn_processor.mode = mode
            if processor_name == 'self':
                if isinstance(attn_processor, IPAttnProcessor_Self):attn_processor.mode = mode
            if processor_name == 'all':
                attn_processor.mode = mode

    def reset_all(self,keep_self=False):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.store_attn, attn_processor.subject_idxs, attn_processor.mask_i, attn_processor.mask_ig_prev, self.subject_idxs = None, None, None, None, None

            if isinstance(attn_processor, IPAttnProcessor_Self):
                attn_processor.mask_id, attn_processor.mask_ig = None, None
                if not keep_self: attn_processor.store_ks, attn_processor.store_vs = [], []
