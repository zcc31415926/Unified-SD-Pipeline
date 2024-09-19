import os
import cv2
import torch
import argparse
from PIL import Image
from tqdm import trange

from sd_pipeline import SD
from sd_pipeline.config import getModelCfg, getGuidanceScale

torch.set_grad_enabled(False)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_type', type=str, default='sd1',
                    choices=['sd1', 'sd2', 'sdxl', 'sd3', 'flux'],
                    help='diffusion backbone')
parser.add_argument('--prompt', type=str, default='',
                    help='prompt as condition for guided generation')
parser.add_argument('--num_generation', type=int, default=1,
                    help='number of the generated images')
parser.add_argument('--out_dir', type=str, default='gen',
                    help='saving directory of the generated images')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size for each round')
parser.add_argument('--out_size', type=int, default=512,
                    help='size of the generated images')
parser.add_argument('--guidance_scale', type=float,
                    help='guidance scale in classifier-free guidance')
parser.add_argument('--use_ddim_scheduler', action='store_true',
                    help='whether to replace the default scheduler ' +
                         'by a DDIM scheduler')
parser.add_argument('--fp16', action='store_true',
                    help='whether to use FP16 precision')
parser.add_argument('--device', type=str, default='cuda',
                    help='device to load the model and the data')
args = parser.parse_args()

# initialization
model_config = getModelCfg(args.model_type, args.use_ddim_scheduler,
                           args.fp16, args.device)
sd = SD(**model_config)

target_dir = f'{args.out_dir}/{args.model_type}'
os.makedirs(target_dir, exist_ok=True)

guidance_scale = getGuidanceScale(args.model_type) \
    if args.guidance_scale is None else args.guidance_scale
assert args.num_generation % args.batch_size == 0, \
    'number of generations should be divided by batch size'
print(f'Generating {args.num_generation} images ' +
      f'with guidance scale {guidance_scale}')
print(f'Prompt: {args.prompt}')

# inference
idx = 0
for _ in trange(args.num_generation // args.batch_size):
    imgs = sd.txt2Img(
        None, [args.prompt] * args.batch_size,
        [''] * args.batch_size, guidance_scale, ori_sizes=None)
    imgs = [cv2.resize(img, (args.out_size, args.out_size))
            for img in imgs]
    for img in imgs:
        Image.fromarray(img).save(f'{target_dir}/{idx}.png')
        idx += 1

