import os

import torch
from PIL import Image
import glob


class LSUN:
    def __init__(self, root, transform=None):
        self.x = glob.glob(os.path.join(root, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        img = Image.open(self.x[item])
        if self.transform:
            img = self.transform(img)
        dummy_target = torch.empty(1)
        return img,  dummy_target
