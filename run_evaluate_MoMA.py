import os
import torch
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
import warnings 

from model_lib.modules import MoMA_main_modal
from model_lib.utils import parse_args
warnings.filterwarnings('ignore') 


seed_everything(0)
args = parse_args()

args.device = torch.device("cuda", 0)
args.model_path = "KunpengSong/MoMA_llava_7b"



# if you have 22 Gb GPU memory:
args.load_8bit, args.load_4bit = False, False

# if you have 18 Gb GPU memory:
# args.load_8bit, args.load_4bit = True, False

# if you have 14 Gb GPU memory:
# args.load_8bit, args.load_4bit = False, True



#load MoMA from HuggingFace. Auto download
moMA_main_modal = MoMA_main_modal(args).to(args.device, dtype=torch.bfloat16)


# reference image and its mask
rgb_path = "example_images/newImages/3.jpg"
mask_path = "example_images/newImages/3_mask.jpg"
subject = 'car'


# Let's generate new images!

################ change context ##################
prompt = "A car in autumn with falling leaves."
generated_image = moMA_main_modal.generate_images(rgb_path, mask_path, subject, prompt, strength=1.0, seed=2, return_mask=True)  # set strength to 1.0 for more accurate details
save_image(generated_image,f"{args.output_path}/{subject}_{prompt}.jpg")

################ change texture ##################
prompt = "A wooden sculpture of a car on the table."
generated_image = moMA_main_modal.generate_images(rgb_path, mask_path, subject, prompt, strength=0.4, seed=4, return_mask=True)  # set strength to 0.4 for better prompt fidelity
save_image(generated_image,f"{args.output_path}/{subject}_{prompt}.jpg")







