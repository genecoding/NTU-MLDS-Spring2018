import os
import torch
from argparse import ArgumentParser

from models import *


parser = ArgumentParser()
parser.add_argument('--model', required=True, help='DCGAN | WGAN | WGAN_GP | CGAN')
parser.add_argument('--load_path', required=True, help='path to load the trained model')
parser.add_argument('--tag_file', default=None, help='tag file for CGAN model')
parser.add_argument('--save_imgname', default='', help='save name for generated image')
args = parser.parse_args()
print(args)

save_dir = './samples'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, args.save_imgname)

torch.manual_seed(12345)

model = args.model.upper()
if model == 'DCGAN':
    m = DCGAN()
elif model == 'WGAN':
    m = WGAN()
elif model == 'WGAN_GP':
    m = WGAN_GP()
elif model == 'CGAN':
    m = CGAN()
else:
    raise ValueError('Unknown Model!')

m.load_model(args.load_path)

if model == 'CGAN':
    assert args.tag_file != None
    with open(args.tag_file) as f:
        tags = f.readlines()
    tags = [t.split(',')[1].split('\n')[0] for t in tags]
    m.test(tags=tags, save_imgname=save_path)
else:  # DCGAN, WGAN, WGAN_GP
    m.test(save_imgname=save_path)
    