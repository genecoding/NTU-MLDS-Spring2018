import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class HW3Dataset(Dataset):
    def __init__(self, img_dir, tag_file, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.tags = pd.read_csv(tag_file, header=None)  # no header in this homework datasets
        self.transform = transform
        self.target_transform = target_transform
  
    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(idx)+'.jpg')
        img = Image.open(img_path)
        tag = self.tags.iloc[idx][1]
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            tag = self.target_transform(tag)

        return img, tag
