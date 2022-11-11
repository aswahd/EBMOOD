import os
import glob
import torch
import torchvision.transforms
from PIL import Image


class SyntheticOOD:
    def __init__(self, root, transform=None):
        filenames = os.listdir(root)
        filenames.sort(key=lambda fn: int(fn.split('.')[0]))
        self.x = [os.path.join(root, fn) for fn in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        img = Image.open(self.x[item])
        if self.transform:
            img = self.transform(img)
        dummy_target = torch.empty(1)
        return img,  dummy_target


class SyntheticSoftOOD:

    def __init__(self, root, transform=None):
        categories = list(os.listdir(root))
        all_files = []
        target = []
        for cat in categories:
            fnames = glob.glob(os.path.join(root, cat, '*.png'))
            all_files.extend(fnames)
            target.extend([cat] * len(fnames))

        self.x = all_files
        self.y = target
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        img = Image.open(self.x[item])
        if self.transform:
            img = self.transform(img)
        y = self.y[item]  # e.g., '69'
        lam = 0.5   # lambda: class-mix weight
        y = [int(y[0]), int(y[1]), lam]

        return img, torch.tensor(y, dtype=torch.float64)


if __name__ == "__main__":
    ds = SyntheticSoftOOD('../data/boundary_data_alpha_soft_labels', torchvision.transforms.ToTensor())
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, 4, )
    x, y = next(iter(dl))
    print(x.shape, y.shape)