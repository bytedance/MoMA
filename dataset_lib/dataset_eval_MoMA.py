from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def Dataset_evaluate_MoMA(rgb_path, prompt,subject, mask_path, moMA_main_modal):

    LLaVa_processor = moMA_main_modal.image_processor_llava
    llava_config = moMA_main_modal.model_llava.config
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
    ])

    rgb_path, prompt,mask_path = rgb_path, prompt,mask_path
    image_pil = Image.open(rgb_path) 
    mask_pil = Image.open(mask_path) 
    blip2_opt = prompt
    
    if transform is not None:
        image_pil = transform(image_pil)
        mask_pil = transform(mask_pil)
    
    mask_pil = np.array(mask_pil)
    mask_pil = mask_pil[:,:,0] if len(mask_pil.shape)==3 else mask_pil
    image = torch.from_numpy(np.array(image_pil)).permute(2,0,1)
    mask = (torch.clamp((torch.from_numpy(mask_pil).unsqueeze(0)).float(),min=0.0,max=1.0)>0).float()

    res = {'image':  (image/127.5-1).unsqueeze(0),\
        'mask': mask.unsqueeze(0), \
        'text': [blip2_opt]}
    
    image_wb = image * mask + torch.ones_like(image)* (1-mask)*255
    image_pil = Image.fromarray(image_wb.permute(1,2,0).numpy().astype(np.uint8))

    res['llava_processed'] = process_images([image_pil], LLaVa_processor, llava_config)
    res['label'] = [subject]
    return res

