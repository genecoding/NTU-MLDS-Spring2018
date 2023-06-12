import json
import spacy
import numpy as np
from os.path import exists
import torch
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence


spacy_en = spacy.load('en_core_web_sm')


def tokenize_en(text):
    """
    Tokenize English text from a string into a list of strings (tokens).
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
    
    
def yield_tokens(dataset, tokenizer):
    for data in dataset:
        yield tokenizer(data)


def build_vocabulary(dataset):
    print('Building vocabulary...')
    vocab = build_vocab_from_iterator(
        yield_tokens(dataset, tokenize_en),
        min_freq=3,
        specials=['<pad>', '<bos>', '<eos>', '<unk>']  # set '<pad>' as 0
    )
    vocab.set_default_index(vocab['<unk>'])

    return vocab


def load_vocab(label_file):
    if not exists('vocab.pt'):
        with open(label_file) as f:
            labels = json.load(f)
        all_captions = [caption for label in labels for caption in label['caption']]
        vocab = build_vocabulary(all_captions)
        torch.save(vocab, 'vocab.pt')
    else:
        vocab = torch.load('vocab.pt')
    print(f'Finished.\nVocabulary sizes: {len(vocab)}')
    
    return vocab
    
    
def collate_fn(batch, tokenizer, vocab, device):
    """
    Insert <eos> at the end of captions, and 
    pad <pad> to the max length in the batch.
    """
    pad_id = vocab.get_stoi()['<pad>']  # e.g. 0
    eos_id = torch.tensor(vocab(['<eos>']))  # e.g. [2]
    
    id_list = []
    cap_list = []
    feat_list = []
    for (id, feat, caption) in batch:
        processed_cap = torch.cat(
            [
                torch.tensor(
                    vocab(tokenizer(caption)),
                    dtype=torch.int32,
                ),
                eos_id,
            ],
            dim=0,
        )
        
        id_list.append(id)
        feat_list.append(feat)
        cap_list.append(processed_cap)
        
    video_feats = torch.tensor(np.array(feat_list), dtype=torch.float32).to(device)
    captions = pad_sequence(cap_list, batch_first=True, padding_value=pad_id).to(device)

    return id_list, video_feats, captions

        
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
        

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs