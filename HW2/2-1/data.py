import os
import json
import random
import numpy as np
from torch.utils.data import Dataset


class HW21Dataset(Dataset):
    def __init__(self, feat_dir, label_file):
        self.feat_dir = feat_dir

        with open(label_file) as f:  
            labels = json.load(f)
        
        self.all_captions = {label['id']: label['caption'] for label in labels}
        self.id_list = list(self.all_captions.keys())
  
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        id = self.id_list[idx]
        video_feat = np.load(os.path.join(self.feat_dir, id+'.npy'))
        caption = random.choice(self.all_captions[id])  # select one corresponding caption randomly
      
        return id, video_feat, caption
    