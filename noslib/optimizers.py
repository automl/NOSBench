import torch

from noslib.program import Program, Instruction, Pointer
from noslib.function import (
    UnaryFunction,
    BinaryFunction,
    DMABinaryFunction,
    DMAUnaryFunction,
)
from noslib.function import interpolate1, interpolate2, bias_correct1, bias_correct2


AdamW = Program(
    [
        Instruction(UnaryFunction(torch.square), Pointer(1), None, Pointer(7)),
        Instruction(DMABinaryFunction(interpolate1), Pointer(8), Pointer(1), Pointer(8)),
        Instruction(DMABinaryFunction(interpolate2), Pointer(9), Pointer(7), Pointer(9)),
        Instruction(DMAUnaryFunction(bias_correct1), Pointer(8), None, Pointer(10)),
        Instruction(DMAUnaryFunction(bias_correct2), Pointer(9), None, Pointer(11)),
        Instruction(UnaryFunction(torch.sqrt), Pointer(11), None, Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(6), Pointer(7)),
        Instruction(BinaryFunction(torch.div), Pointer(10), Pointer(7), Pointer(7)),
        Instruction(BinaryFunction(torch.mul), Pointer(0), Pointer(5), Pointer(12)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(12), Pointer(7)),
    ]
)

Adam = Program(AdamW[:-2])

SGD = Program(
    [
        Instruction(BinaryFunction(torch.mul), Pointer(3), Pointer(7), Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(1), Pointer(7), Pointer(7)),
    ]
)

RMSprop = Program(
    [
        Instruction(UnaryFunction(torch.square), Pointer(1), None, Pointer(7)),
        Instruction(DMABinaryFunction(interpolate2), Pointer(8), Pointer(7), Pointer(8)),
        Instruction(UnaryFunction(torch.sqrt), Pointer(8), None, Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(6), Pointer(7)),
        Instruction(BinaryFunction(torch.div), Pointer(1), Pointer(7), Pointer(7)),
    ]
)

Adagrad = Program(
    [
        Instruction(UnaryFunction(torch.square), Pointer(1), None, Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(8), Pointer(8)),
        Instruction(UnaryFunction(torch.sqrt), Pointer(8), None, Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(6), Pointer(7)),
        Instruction(BinaryFunction(torch.div), Pointer(1), Pointer(7), Pointer(7)),
    ]
)
