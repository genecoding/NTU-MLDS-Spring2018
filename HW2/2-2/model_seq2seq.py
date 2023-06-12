import os
import torch
from torch import nn
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split, RandomSampler

from utils import *
from seq2seq import *
from solver import Solver
from data import HW22Dataset


parser = ArgumentParser()
parser.add_argument('--train_data', required=True, help='path to training data')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
args = parser.parse_args()
print(args)

# use all training data to build the vocabulary
voc = load_vocab(args.train_data)

hw22_dataset = HW22Dataset(args.train_data, voc)
train_data, valid_data = random_split(hw22_dataset, [0.9, 0.1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

collate_fn_22 = lambda batch: collate_fn(batch, voc, device)

NUM_TRAIN_SAMPLES = 100000
NUM_VALID_SAMPLES = 5000
train_sampler = RandomSampler(train_data, num_samples=NUM_TRAIN_SAMPLES)
valid_sampler = RandomSampler(valid_data, num_samples=NUM_VALID_SAMPLES)

train_iter = DataLoader(
    train_data, 
    batch_size=128, 
    sampler=train_sampler,
    collate_fn=collate_fn_22,
    )
valid_iter = DataLoader(
    valid_data, 
    batch_size=128, 
    sampler=valid_sampler,
    collate_fn=collate_fn_22,
    )

OUTPUT_DIM = len(voc)
EMB_DIM = 1024
EN_HID_DIM = 512
DE_HID_DIM = 512
DROPOUT = 0.5

bos_id = voc.get_stoi()['<bos>']
pad_id = voc.get_stoi()['<pad>']
    
embed = nn.Embedding(OUTPUT_DIM, EMB_DIM, padding_idx=pad_id)
attn = Attention(EN_HID_DIM, DE_HID_DIM)
enc = Encoder(embed, EMB_DIM, EN_HID_DIM, DROPOUT)
dec = Decoder(embed, attn, OUTPUT_DIM, EMB_DIM, EN_HID_DIM, DE_HID_DIM, DROPOUT)
seq2seq = Seq2Seq(enc, dec, device, bos_id).to(device)
    
solver = Solver(seq2seq, pad_id)
solver.train(train_iter, valid_iter, num_epochs=args.num_epochs)