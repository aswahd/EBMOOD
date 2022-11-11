import torch.nn as nn

cross_entropy = nn.CrossEntropyLoss()


def soft_entropy(pred, y_a, y_b, lam):
    return lam * cross_entropy(pred, y_a) + (1 - lam) * cross_entropy(pred, y_b)

