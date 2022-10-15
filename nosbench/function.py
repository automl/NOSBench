import torch


class Function:
    __slots__ = "func", "n_args"

    def __init__(self, func, n_args):
        self.func = func
        self.n_args = n_args

    def __call__(self, args):
        return self.func(*args)

    def __str__(self):
        return str(self.func.__name__)

    def __repr__(self):
        return str(self.func.__name__)


def size(x):
    if x.dim() == 0:
        return torch.tensor(1.0)
    return torch.tensor(x.shape.numel(), dtype=torch.float)


def clip(x, max):
    return torch.clip(x, min=None, max=max)


def interpolate(x1, x2, beta):
    return x1 * beta + x2 * (1.0 - beta)


def bias_correct(x1, beta, step):
    bias_correction = 1.0 - torch.pow(beta, step)
    return x1 / bias_correction
