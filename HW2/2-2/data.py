from torch.utils.data import Dataset


class HW22Dataset(Dataset):
    def __init__(self, convo_file, vocab=None, mode='train'):
        MIN_LEN = 2
        MAX_LEN = 22
        self.mode = mode
        self.data_list = []

        # remove conversations which have fewer than 2 sentences,
        # remove sentences which have no word
        with open(convo_file, 'r') as f:
            convo_list = [[line for line in convo.split('\n')[:-1] if len(line) > 0] 
                          for convo in f.read().split('+++$+++\n') if len(convo.split('\n')) > 2]

        if mode == 'train':
            unk_id = vocab.get_stoi()['<unk>']
            x_list = []
            y_list = []
            for convo in convo_list:
                x_list += convo[:-1]
                y_list += convo[1:]

            # create dialogue pairs,
            # remove pairs which have too long or too short sentence(s),
            # remove pairs which have <unk> token in it
            self.data_list = [[x, y] for x, y in zip(x_list, y_list) if 
                              len(x.split()) >= MIN_LEN and len(y.split()) >= MIN_LEN 
                              and len(x.split()) <= MAX_LEN and len(y.split()) <= MAX_LEN
                              and unk_id not in vocab(x.split()) and unk_id not in vocab(y.split())]
        elif mode == 'test':
            self.data_list = [line for convo in convo_list for line in convo]
        else:
            raise ValueError('mode must be train or test!')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.data_list[idx][0], self.data_list[idx][1]
        elif self.mode == 'test':
            return self.data_list[idx], None
    