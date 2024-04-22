# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from torchvision.utils import save_image
import torchvision.transforms as T

def get_mask_from_cross(attn_processors):
    reference_masks = []
    for attn_processor in attn_processors.values():
        if isinstance(attn_processor, IPAttnProcessor):
            reference_masks.append(attn_processor.mask_i)
    mask = torch.cat(reference_masks,dim=1).mean(dim=1)
    mask = (mask-mask.min())/(mask.max()-mask.min())
    mask = (mask>0.2).to(torch.float32)*mask
    mask = (mask-mask.min())/(mask.max()-mask.min())
    return mask.unsqueeze(1)

class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

        self.store_attn = None
        self.enabled = True
        self.mode = 'inject'

        self.subject_idxs = None
        self.mask_i = None
        self.mask_ig_prev = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # for ip-adapter
        if self.enabled:
            if self.mode == 'inject' or self.mode == 'masked_generation':
                ip_key = self.to_k_ip(ip_hidden_states.to(torch.float16))
                ip_value = self.to_v_ip(ip_hidden_states.to(torch.float16))
                ip_key = attn.head_to_batch_dim(ip_key)
                ip_value = attn.head_to_batch_dim(ip_value)
                ip_attention_probs = attn.get_attention_scores(query, ip_key.to(torch.float32), None)
                ip_hidden_states = torch.bmm(ip_attention_probs, ip_value.to(torch.float32))
                ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
                if (self.mask_ig_prev is not None) and self.mode == 'masked_generation': 
                    mask_ig_prev = rearrange(F.interpolate(self.mask_ig_prev,size=int(math.sqrt(query.shape[1]))),"b c h w -> b (h w) c")
                    if not mask_ig_prev.shape[0]==ip_hidden_states.shape[0]: mask_ig_prev = mask_ig_prev.repeat(2,1,1)
                    ip_hidden_states = ip_hidden_states * mask_ig_prev
                hidden_states = hidden_states + self.scale * ip_hidden_states
            if self.mode == 'extract' or self.mode == 'masked_generation':
                subject_idxs = self.subject_idxs*2 if not (hidden_states.shape[0] == len(self.subject_idxs)) else self.subject_idxs
                assert (hidden_states.shape[0] == len(subject_idxs))
                attentions = rearrange(attention_probs, '(b h) n d -> b h n d', h=8).mean(1)
                attn_extracted = [attentions[i, :, subject_idxs[i]].sum(-1) for i in range(hidden_states.shape[0])]  
                attn_extracted = [(atn-atn.min())/(atn.max()-atn.min()) for atn in attn_extracted]
                attn_extracted = torch.stack(attn_extracted, dim=0)
                attn_extracted = rearrange(attn_extracted, 'b (h w) -> b h w', h=int(math.sqrt(attention_probs.shape[1])))
                attn_extracted = torch.clamp(F.interpolate(attn_extracted.unsqueeze(1),size=512),min=0,max=1)
                self.mask_i = attn_extracted

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states

### added for self attention
class IPAttnProcessor_Self(nn.Module):
    r"""
    Attention processor for IP-Adapater. (But for self attention)
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale_learnable = torch.nn.Parameter(torch.zeros(1),requires_grad=True)

        self.enabled = True
        self.mode = 'extract'

        self.store_ks, self.store_vs = [], []
        self.mask_id, self.mask_ig = None, None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key_0 = attn.to_k(encoder_hidden_states)
        value_0 = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key_0)
        value = attn.head_to_batch_dim(value_0)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.enabled:
            if self.mode == 'extract':
                ks, vs = attn.head_to_batch_dim(self.to_k_ip(key_0)), attn.head_to_batch_dim(self.to_v_ip(value_0))
                self.store_ks, self.store_vs = self.store_ks+[ks], self.store_vs+[vs]
                self.store_ks, self.store_vs = torch.cat(self.store_ks,dim=0), torch.cat(self.store_vs,dim=0)

            if self.mode == 'masked_generation':
                if not self.store_ks.shape[0]==query.shape[0]: self.store_ks,self.store_vs = self.store_ks.repeat(2,1,1), self.store_vs.repeat(2,1,1)
                mask_id = self.mask_id.clone()
                mask_id.masked_fill_(self.mask_id==False, -torch.finfo(mask_id.dtype).max)
                mask_id = rearrange(F.interpolate(mask_id,size=int(math.sqrt(query.shape[1]))),"b c h w -> b c (h w)").repeat(1,query.shape[1],1)
                mask_id = mask_id.repeat(8,1,1) # 8 is head dim
                if not mask_id.shape[0]==int(query.shape[0]): mask_id = mask_id.repeat(2,1,1)
                attention_probs_ref = attn.get_attention_scores(query, self.store_ks, mask_id.to(query.dtype))
                hidden_states_ref = torch.bmm(attention_probs_ref, self.store_vs)
                hidden_states_ref = attn.batch_to_head_dim(hidden_states_ref)
                scale = self.scale.repeat(int(batch_size/self.scale.shape[0])).unsqueeze(-1).unsqueeze(-1) if type(self.scale)==torch.Tensor else self.scale
                if self.mask_ig == None:
                    hidden_states = hidden_states + scale * hidden_states_ref * self.scale_learnable
                else:
                    mask_ig = rearrange(F.interpolate(self.mask_ig,size=int(math.sqrt(query.shape[1]))),"b c h w -> b (h w) c")
                    if not mask_ig.shape[0]==hidden_states_ref.shape[0]: mask_ig = mask_ig.repeat(2,1,1)
                    hidden_states = hidden_states + scale * hidden_states_ref * mask_ig * self.scale_learnable

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states