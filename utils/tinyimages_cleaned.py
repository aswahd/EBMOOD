import numpy as np
import matplotlib.pyplot as plt
import torch


class TinyImages:

    def __init__(self, root, transform=None):
        self.x = np.load(root)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.empty(1)
