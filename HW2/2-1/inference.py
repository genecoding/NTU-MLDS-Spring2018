import os
import re
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from utils import *
from s2vt import S2VT
from data import HW21Dataset


def post_process(s):
    s = ' '.join(s)
    # remove redundant space before punctuations except for <unk>
    s = re.sub(r' (?=\W[^<unk>])', '', s)
    # remove the first <eos> and any content after that
    s = re.split(r'<eos>', s)[0]
  
    return s


@torch.no_grad()
def video_captioning(test_iter, voc, model, device, save_name):
    model.eval()
    
    result = {}
    for i, (id, feats, captions) in enumerate(test_iter):
        output = model.inference(feats)
        pred = output.argmax(-1).squeeze(0)

        pred_caption = [voc.get_itos()[token] for token in pred]
        orig_caption = [voc.get_itos()[token] for token in captions[0]]
        
        print(id[0])
        print('original caption:', post_process(orig_caption))
        print('predicted caption:', post_process(pred_caption))
        print()
        
        result[id[0]] = post_process(pred_caption)

    with open(save_name, 'w') as f:
        for id in result:
            f.write(f'{id},{result[id]}\n')
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_data', required=True, help='path to testing data')
    parser.add_argument('--use_attention', action='store_true', 
                        help='use attention or not, must be the same as training')
    parser.add_argument('--output_file', type=str, default='caption_output.txt', 
                        help='name of the output file')
    args = parser.parse_args()
    print(args)
    
    test_feat_dir = os.path.join(args.test_data, 'testing_data/feat')
    test_label_file = os.path.join(args.test_data, 'testing_label.json')
    
    test_data = HW21Dataset(test_feat_dir, test_label_file)
    
    # load the built vocabulary
    voc = torch.load('vocab.pt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    collate_fn_21 = lambda batch: collate_fn(batch, tokenize_en, voc, device)
    
    test_iter = DataLoader(
        test_data, 
        batch_size=1,
        shuffle=False, 
        collate_fn=collate_fn_21,
    )
    
    s2vt = S2VT(
        device=device, 
        output_dim=len(voc),
        bos_id=voc.get_stoi()['<bos>'],
        pad_id=voc.get_stoi()['<pad>'],
        use_attention=args.use_attention
    ).to(device)
    
    s2vt.load_state_dict(torch.load('./saved_model/s2vt_model_bestscore.pt'))
    video_captioning(test_iter, voc, s2vt, device, args.output_file)
    