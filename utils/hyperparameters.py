import torch
import torch.nn as nn


def weights_init(m):
    """
    Xavier initialization for CNNs
        :param m: model
        :return: No return, inplace operation

    Usage:
        model = Net()
        model.apply(weights_init)
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


if __name__ == "__main__":
    pass
