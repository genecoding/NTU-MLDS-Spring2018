import re
import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from utils import *
from seq2seq import *
from data import HW22Dataset


def post_process(s):
    s = ''.join(s)
    # remove redundant space before punctuations except for <unk>
    # s = re.sub(r' (?=\W[^<unk>])', '', s)
    # remove the first <eos> and any content after that
    s = re.split(r'<eos>', s)[0]
  
    return s


@torch.no_grad()
def chat_generation(test_iter, voc, model, device, save_name):
    model.eval()
    
    result = []
    for i, (src) in enumerate(test_iter):
        output = model.inference(src)
        pred = output.argmax(-1).squeeze(0)

        pred = pred[:, 1:].detach().cpu().numpy()
        # "The vectorize function is provided primarily for convenience, not for performance." (by official manual)
        itos = np.vectorize(lambda token: voc.get_itos()[token])
        
        pred_lines = itos(pred)
        result += [post_process(p_line) for p_line in pred_lines]

    with open(save_name, 'w') as f:
        for line in result:
            f.write(f'{line}\n')
            
    print(f'{save_name} saved.')
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_data', required=True, help='path to testing data')
    parser.add_argument('--output_file', type=str, default='output.txt', 
                        help='name of the output file')
    args = parser.parse_args()
    print(args)
    
    test_data = HW22Dataset(args.test_data, mode='test')
    
    # load the built vocabulary
    voc = torch.load('vocab.pt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    collate_fn_22 = lambda batch: collate_fn(batch, voc, device)
    
    test_iter = DataLoader(
        test_data, 
        batch_size=200,
        shuffle=False, 
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
    
    seq2seq.load_state_dict(torch.load('./saved_model/seq2seq_model_final.pt'))
    chat_generation(test_iter, voc, seq2seq, device, args.output_file)
    