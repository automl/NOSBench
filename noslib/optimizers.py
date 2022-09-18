import torch

from noslib.program import Program, Instruction, Pointer, READONLY_REGION
from noslib.function import interpolate, bias_correct, Function


# Rename memory locations for easier reading
w = Pointer(0)
g = Pointer(1)
step = Pointer(2)
weight_decay = Pointer(6)
momentum = Pointer(READONLY_REGION + 1)
beta1 = Pointer(READONLY_REGION + 1)
beta2 = Pointer(READONLY_REGION + 2)
alpha = Pointer(READONLY_REGION + 2)
g_square = Pointer(READONLY_REGION + 3)
m = Pointer(READONLY_REGION + 4)
v = Pointer(READONLY_REGION + 5)
m_hat = Pointer(READONLY_REGION + 6)
v_hat = Pointer(READONLY_REGION + 7)
eps = Pointer(READONLY_REGION + 8)
v_hat_sqrt = Pointer(READONLY_REGION + 9)
v_sqrt = Pointer(READONLY_REGION + 9)
wd = Pointer(READONLY_REGION + 10)
update = Pointer(READONLY_REGION + 11)


AdamW = Program(
    [
        Instruction(Function(torch.square, 1), [g], g_square),
        Instruction(Function(torch.sub, 2), [Pointer(3), Pointer(5)], beta1),
        Instruction(Function(torch.sub, 2), [Pointer(3), Pointer(7)], beta2),
        Instruction(Function(interpolate, 3), [m, g, beta1], m),
        Instruction(Function(interpolate, 3), [v, g_square, beta2], v),
        Instruction(Function(bias_correct, 3), [m, beta1, step], m_hat),
        Instruction(Function(bias_correct, 3), [v, beta2, step], v_hat),
        Instruction(Function(torch.mul, 2), [Pointer(6), Pointer(8)], eps),
        Instruction(Function(torch.sqrt, 1), [v_hat], v_hat_sqrt),
        Instruction(Function(torch.add, 2), [v_hat_sqrt, eps], v_hat_sqrt),
        Instruction(Function(torch.div, 2), [m_hat, v_hat_sqrt], update),
        Instruction(Function(torch.mul, 2), [w, weight_decay], wd),
        Instruction(Function(torch.add, 2), [update, wd], update),
    ]
)

Adam = Program(AdamW[:-2])

SGD = Program(
    [
        Instruction(Function(torch.sub, 2), [Pointer(3), Pointer(5)], momentum),
        Instruction(Function(torch.mul, 2), [m, momentum], m),
        Instruction(Function(torch.add, 2), [g, m], m),
    ]
)

RMSprop = Program(
    [
        Instruction(Function(torch.sub, 2), [Pointer(3), Pointer(7)], alpha),
        Instruction(Function(torch.square, 1), [g], g_square),
        Instruction(Function(interpolate, 3), [v, g_square, alpha], v),
        Instruction(Function(torch.mul, 2), [Pointer(6), Pointer(8)], eps),
        Instruction(Function(torch.sqrt, 1), [v], v_sqrt),
        Instruction(Function(torch.add, 2), [v_sqrt, eps], v_sqrt),
        Instruction(Function(torch.div, 2), [g, v_sqrt], update),
    ]
)


Adagrad = Program(
    [
        Instruction(Function(torch.square, 1), [g], g_square),
        Instruction(Function(torch.add, 2), [g_square, v], v),
        Instruction(Function(torch.sqrt, 1), [v], v_sqrt),
        Instruction(Function(torch.mul, 2), [Pointer(6), Pointer(8)], eps),
        Instruction(Function(torch.add, 2), [v_sqrt, eps], v_sqrt),
        Instruction(Function(torch.div, 2), [g, v_sqrt], update),
    ]
)
