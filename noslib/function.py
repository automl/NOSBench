import torch


class Function:
    __slots__ = "func"

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return str(self.func.__name__)

    def __repr__(self):
        return str(self.func.__name__)


UnaryFunction = type("UnaryFunction", (Function,), {})
BinaryFunction = type("BinaryFunction", (Function,), {})
DMABinaryFunction = type("DMABinaryFunction", (BinaryFunction,), {})
DMAUnaryFunction = type("DMAUnaryFunction", (UnaryFunction,), {})


def _interpolate(x1, x2, beta):
    return x1 * beta + x2 * (1.0 - beta)


def interpolate1(x1, x2, memory):
    return _interpolate(x1, x2, memory[3])


def interpolate2(x1, x2, memory):
    return _interpolate(x1, x2, memory[4])


def _bias_correct(x1, beta, step):
    bias_correction = 1.0 - torch.pow(beta, step)
    return x1 / bias_correction


def bias_correct1(x1, memory):
    return _bias_correct(x1, memory[3], memory[2])


def bias_correct2(x1, memory):
    return _bias_correct(x1, memory[4], memory[2])
