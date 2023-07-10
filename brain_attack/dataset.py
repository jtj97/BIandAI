import os
from collections import OrderedDict

from torch.utils.data import Dataset
from torch import nn
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from util import characters


class CaptchaDataset(Dataset):
    def __init__(self, root_dir: str, input_length):
        self.root_dir = root_dir
        raw_filename = os.listdir(self.root_dir)
        self.transformer = transforms.ToTensor()
        # 字符串标签
        self.labels = [filename[:-4] for filename in raw_filename]
        # 向量标签
        self.target = [torch.tensor([characters.find(x) for x in label], dtype=torch.long) for label in self.labels]
        self.image = []
        for filename in tqdm(raw_filename):
            im = Image.open(os.path.join(self.root_dir, filename))
            im = im.resize([192, 64], Image.ANTIALIAS)
            self.image.append(self.transformer(im))
        self.input_length = torch.full(size=(1,), fill_value=input_length, dtype=torch.long)
        self.target_length = torch.full(size=(1,), fill_value=4, dtype=torch.long)

    def __getitem__(self, idx):
        return self.image[idx], self.target[idx], self.input_length, self.target_length

    def __len__(self):
        return len(self.labels)

