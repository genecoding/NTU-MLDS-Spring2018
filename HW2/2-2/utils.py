from os.path import exists
import torch
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence


def yield_tokens(conversation):
    for line in conversation:
        yield line.split()


def build_vocabulary(dataset):
    print('Building vocabulary...')
    vocab = build_vocab_from_iterator(
        yield_tokens(dataset),
        min_freq=90,
        specials=['<pad>', '<bos>', '<eos>', '<unk>']  # set '<pad>' as 0
    )
    vocab.set_default_index(vocab['<unk>'])

    return vocab


def load_vocab(convo_file):
    if not exists('vocab.pt'):
        with open(convo_file) as f:
          convo_list = [line for convo in f.read().split('+++$+++\n') for line in convo.split('\n')[:-1]]
        vocab = build_vocabulary(convo_list)
        torch.save(vocab, 'vocab.pt')
    else:
        vocab = torch.load('vocab.pt')
    print(f'Finished.\nVocabulary sizes: {len(vocab)}')

    return vocab
    
    
def collate_fn(batch, vocab, device):
    """
    Insert <bos> at the start of sentences and <eos> at the end,
    and pad <pad> to the max length in the batch.
    """
    pad_id = vocab.get_stoi()['<pad>']  # e.g. 0
    bos_id = torch.tensor(vocab(['<bos>']))  # e.g. [1]
    eos_id = torch.tensor(vocab(['<eos>']))

    x_list = []
    y_list = []

    preprocess = lambda line: torch.cat(
        [
            bos_id,
            torch.tensor(
                vocab(line.split()),
                dtype=torch.int32,
            ),
            eos_id,
        ],
        dim=0,
    )

    for x, y in batch:
        processed_x = preprocess(x)
        x_list.append(processed_x)
        if y is not None:
            processed_y = preprocess(y)
            y_list.append(processed_y)

    x = pad_sequence(x_list, batch_first=True, padding_value=pad_id).to(device)
    if y_list == []:  # test mode
        return x
    else:  # train mode
        y = pad_sequence(y_list, batch_first=True, padding_value=pad_id).to(device)
        return x, y


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