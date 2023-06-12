import os
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split

from utils import *
from s2vt import S2VT
from solver import Solver
from data import HW21Dataset


parser = ArgumentParser()
parser.add_argument('--train_data', required=True, help='path to training data')
parser.add_argument('--use_attention', action='store_true', help='use attention or not')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
args = parser.parse_args()
print(args)

feat_dir = os.path.join(args.train_data, 'training_data/feat')
label_file = os.path.join(args.train_data, 'training_label.json')

hw21_dataset = HW21Dataset(feat_dir, label_file)
train_data, valid_data = random_split(hw21_dataset, [0.95, 0.05])

# use all training data to build the vocabulary
voc = load_vocab(label_file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

collate_fn_21 = lambda batch: collate_fn(batch, tokenize_en, voc, device)

train_iter = DataLoader(
    train_data, 
    batch_size=128, 
    shuffle=True, 
    collate_fn=collate_fn_21,
    )
valid_iter = DataLoader(
    valid_data, 
    batch_size=128, 
    shuffle=True, 
    collate_fn=collate_fn_21,
    )
    
s2vt = S2VT(
    device=device, 
    output_dim=len(voc),
    bos_id=voc.get_stoi()['<bos>'],
    pad_id=voc.get_stoi()['<pad>'],
    use_attention=args.use_attention
    ).to(device)
    
solver = Solver(s2vt, voc.get_stoi()['<pad>'])
solver.train(train_iter, valid_iter, num_epochs=args.num_epochs)
