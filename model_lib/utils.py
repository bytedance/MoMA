import argparse
import torch
from torchvision.transforms import ToPILImage
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of MoMA.")
    parser.add_argument("--load_attn_adapters",type=str,default="checkpoints/attn_adapters_projectors.th",help="self_cross attentions and LLM projectors.")
    parser.add_argument("--output_path",type=str,default="output",help="output directory.")
    parser.add_argument("--device",type=str,default="cuda:0",help="device.")
    parser.add_argument("--model_path",type=str,default="KunpengSong/MoMA_llava_7b",help="fine tuned llava (Multi-modal LLM decoder)")
    
    args = parser.parse_known_args()[0]
    return args

def show_PIL_image(tensor):
    # tensor of shape [3, 3, 512, 512]
    to_pil = ToPILImage()
    images = [to_pil(tensor[i]) for i in range(tensor.shape[0])]

    concatenated_image = Image.new('RGB', (images[0].width * 3, images[0].height))
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return concatenated_image
