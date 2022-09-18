import torch

from noslib.program import Program, Instruction, Pointer
from noslib.function import interpolate, bias_correct, Function


AdamW = Program(
    [
        Instruction(Function(torch.square, 2), [Pointer(1)], Pointer(7)),
        Instruction(
            Function(interpolate, 3), [Pointer(8), Pointer(1), Pointer(3)], Pointer(8)
        ),
        Instruction(
            Function(interpolate, 3), [Pointer(9), Pointer(7), Pointer(4)], Pointer(9)
        ),
        Instruction(
            Function(bias_correct, 3), [Pointer(8), Pointer(3), Pointer(2)], Pointer(10)
        ),
        Instruction(
            Function(bias_correct, 3), [Pointer(9), Pointer(4), Pointer(2)], Pointer(11)
        ),
        Instruction(Function(torch.sqrt, 1), [Pointer(11)], Pointer(7)),
        Instruction(Function(torch.add, 2), [Pointer(7), Pointer(6)], Pointer(7)),
        Instruction(Function(torch.div, 2), [Pointer(10), Pointer(7)], Pointer(7)),
        Instruction(Function(torch.mul, 2), [Pointer(0), Pointer(5)], Pointer(12)),
        Instruction(Function(torch.add, 2), [Pointer(7), Pointer(12)], Pointer(7)),
    ]
)

Adam = Program(AdamW[:-2])

SGD = Program(
    [
        Instruction(Function(torch.mul, 2), [Pointer(3), Pointer(7)], Pointer(7)),
        Instruction(Function(torch.add, 2), [Pointer(1), Pointer(7)], Pointer(7)),
    ]
)

RMSprop = Program(
    [
        Instruction(Function(torch.square, 2), [Pointer(1)], Pointer(7)),
        Instruction(
            Function(interpolate, 3), [Pointer(8), Pointer(7), Pointer(4)], Pointer(8)
        ),
        Instruction(Function(torch.sqrt, 1), [Pointer(8)], Pointer(7)),
        Instruction(Function(torch.add, 2), [Pointer(7), Pointer(6)], Pointer(7)),
        Instruction(Function(torch.div, 2), [Pointer(1), Pointer(7)], Pointer(7)),
    ]
)


Adagrad = Program(
    [
        Instruction(Function(torch.square, 1), [Pointer(1)], Pointer(7)),
        Instruction(Function(torch.add, 2), [Pointer(7), Pointer(8)], Pointer(8)),
        Instruction(Function(torch.sqrt, 1), [Pointer(8)], Pointer(7)),
        Instruction(Function(torch.add, 2), [Pointer(7), Pointer(6)], Pointer(7)),
        Instruction(Function(torch.div, 2), [Pointer(1), Pointer(7)], Pointer(7)),
    ]
)
