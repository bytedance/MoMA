import os
from PIL import Image
import torch
import torch.nn as nn
from typing import List, Optional
import torch.utils.checkpoint
from torchvision.transforms import ToPILImage
from model_lib.moMA_generator import MoMA_generator
from transformers.activations import ACT2FN
from huggingface_hub import hf_hub_download

from dataset_lib.dataset_eval_MoMA import Dataset_evaluate_MoMA

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX

def add_function(model):
    def my_llava_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        (_,position_ids,attention_mask,_,inputs_embeds,_) = self.prepare_inputs_labels_for_multimodal(input_ids,position_ids,attention_mask,None,None,images)
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs[0]
    
    model.my_llava_forward = my_llava_forward


class LlamaMLP_mapping(nn.Module):
    def __init__(self, hidden_size,hidden_size_out):
        super().__init__()
        self.hidden_size, self.hidden_size_out = hidden_size,hidden_size_out
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size_out, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.hidden_size_out, bias=False)
        self.down_proj = nn.Linear(self.hidden_size_out, self.hidden_size_out, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.act_fn_output = ACT2FN["tanh"]
        self.init_linear()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

    def init_linear(self):
        torch.nn.init.xavier_normal_(self.gate_proj.weight) 
        self.gate_proj.weight.data=self.gate_proj.weight.data/4.0
        torch.nn.init.xavier_normal_(self.up_proj.weight) 
        self.up_proj.weight.data=self.up_proj.weight.data/4.0
        torch.nn.init.xavier_normal_(self.down_proj.weight) 
        self.down_proj.weight.data=self.down_proj.weight.data/4.0

class MoMA_main_modal(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.device = args.device

        self.moMA_generator = MoMA_generator(self.device,args)
        self.unet = self.moMA_generator.pipe.unet
        self.vae = self.moMA_generator.pipe.vae
        
        print('Loading MoMA: its Multi-modal LLM...')
        model_name = get_model_name_from_path(args.model_path)
        self.tokenizer_llava, self.model_llava, self.image_processor_llava, self.context_len_llava = load_pretrained_model(args.model_path, None, model_name, load_8bit=self.args.load_8bit, load_4bit=self.args.load_4bit, device=args.device)
        
        add_function(self.model_llava)

        self.mapping = LlamaMLP_mapping(4096,1024).to(self.device, dtype=torch.bfloat16)
        self.load_saved_components()
        self.freeze_modules()

    def load_saved_components(self):
        if not os.path.exists(self.args.load_attn_adapters):
            print('Loading Attentions and LLM mappings...')
            hf_hub_download(repo_id=self.args.model_path, filename="attn_adapters_projectors.th",local_dir='/'.join(self.args.load_attn_adapters.split('/')[:-1]))

        #load attention adapters and self cross attentions
        state_dict = torch.load(self.args.load_attn_adapters, map_location="cpu")
        self.moMA_generator.image_proj_model.load_state_dict(state_dict["projectors"])
        attn_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        attn_layers.load_state_dict(state_dict["self_cross_attentions"],strict=False)

        #load LLM projectors
        self.load_state_dict(state_dict['llm_mapping'],strict=False)

    def freeze_modules(self): 
        all_modules = [self.moMA_generator.pipe.vae,self.moMA_generator.pipe.text_encoder,self.unet,self.model_llava,self.mapping]
        for module in all_modules:
            module.train = False
            module.requires_grad_(False)

    def forward_MLLM(self,batch):
        llava_processeds,subjects,prompts = batch['llava_processed'].half().to(self.device),batch['label'],batch['text']
        
        input_ids,attention_masks,position_ids = [],[],[]
        for subject,prompt in zip(subjects,prompts):
            prompt_construct = f"USER: <image>\n A photo of a {subject}. Describe a new image of the same {subject} in: {prompt}. ASSISTANT: *" 
            input_id = tokenizer_image_token(prompt_construct, self.tokenizer_llava, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            attention_mask = torch.ones(input_id.shape, dtype=torch.long, device=self.device)
            position_id = torch.tensor(list(range(input_id.shape[-1])), device=self.device)
            
            position_ids += [position_id]
            attention_masks += [attention_mask[0]]
            input_ids += [input_id[0]] 
        
        input_ids = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[-1])  for i in input_ids],batch_first=True,padding_value=self.tokenizer_llava.pad_token_id).flip(dims=[1]) 
        position_ids = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[-1])  for i in position_ids],batch_first=True,padding_value=self.tokenizer_llava.pad_token_id).flip(dims=[1]) 
        attention_masks = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[-1])  for i in attention_masks],batch_first=True,padding_value=self.tokenizer_llava.pad_token_id).flip(dims=[1]) 
        
        output = self.model_llava.my_llava_forward(self.model_llava,input_ids=input_ids,attention_mask=attention_masks,position_ids=position_ids,images=llava_processeds)
        output = self.mapping(output)
        return output[:,-1,:]

    def reset(self):
        self.moMA_generator.reset_all()

    def generate_images(self, rgb_path, mask_path, subject, prompt, strength=1.0, num=1, seed=0, return_mask=False):
        batch = Dataset_evaluate_MoMA(rgb_path, prompt, subject, mask_path,self)
        self.moMA_generator.set_selfAttn_strength(strength)
        
        num = 1 if return_mask else num
        results = []
        for sample_id in range(num):
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
                with torch.no_grad(): 
                    
                    ### key steps
                    llava_emb = self.forward_MLLM(batch).clone().detach()
                    img,mask = self.moMA_generator.generate_with_MoMA(batch,llava_emb=llava_emb,seed=sample_id+seed,device=self.args.device)                            
                    self.reset()
                    ###
                    
                    if return_mask:
                        return torch.cat([(batch['image'].cpu()+1)/2.0,img,mask],dim=0)
                    else:
                        results += [img[0]]
        
        to_pil = ToPILImage()
        images = [to_pil(results[i]) for i in range(len(results))]
        concatenated_image = Image.new('RGB', (images[0].width * num, images[0].height))
        x_offset = 0
        for img in images:
            concatenated_image.paste(img, (x_offset, 0))
            x_offset += img.width

        return concatenated_image