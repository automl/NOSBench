import torch

from zero import Program, Instruction, Pointer
from zero import UnaryFunction, BinaryFunction, DMABinaryFunction, DMAUnaryFunction
from zero import create_optimizer, interpolate1, interpolate2, interpolate_bc1, interpolate_bc2


AdamW = Program(
    [
        Instruction(UnaryFunction(torch.square), Pointer(1), None, Pointer(7)),
        Instruction(DMABinaryFunction(interpolate_bc1), Pointer(8), Pointer(1), Pointer(10)),
        Instruction(DMABinaryFunction(interpolate_bc2), Pointer(9), Pointer(7), Pointer(11)),
        Instruction(UnaryFunction(torch.sqrt), Pointer(11), None, Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(6), Pointer(7)),
        Instruction(BinaryFunction(torch.div), Pointer(10), Pointer(7), Pointer(7)),
        Instruction(BinaryFunction(torch.mul), Pointer(0), Pointer(5), Pointer(12)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(12), Pointer(7)),
    ]
)

Adam = AdamW[:-2]

SGD = Program(
    [
        Instruction(BinaryFunction(torch.mul), Pointer(3), Pointer(7), Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(1), Pointer(7), Pointer(7)),
    ]
)

RMSProp = Program(
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


if __name__ == "__main__":
    torch.manual_seed(123)
    model = torch.nn.Linear(1, 1)
    optimizer_class = create_optimizer(Adam)
    optim = optimizer_class(model.parameters(), lr=0.01)
    # optim = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.999)
    # optim = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.0099999)
    # optim = torch.optim.Adagrad(model.parameters(), lr=0.01)
    initial_loss = torch.nn.functional.mse_loss(model(torch.tensor([1.0])), torch.tensor([1.0])).item()
    import time

    prev_time = time.time()
    for _ in range(1000):
        output = model(torch.tensor([1.0]))
        loss = torch.nn.functional.mse_loss(output, torch.tensor([1.0]))
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss = loss.item()
    print(f"Initial Loss: {initial_loss}, Final Loss: {loss}")
    print(f"Elapsed seconds: {time.time() - prev_time}")
