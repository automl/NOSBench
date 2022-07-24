from dataclasses import dataclass
from typing import Optional, Callable, NewType

import torch


"""
- Hyperparameters are fixed
- Memory Structure: [weights, gradients, step, lr, beta1, beta2, eps, decay, ... zero]
- Return last instructions output as update, empty program has 0 update
- Instructions: unary, binary, direct memory access operations
"""


Pointer = NewType("Pointer", int)
Program = NewType("Program", list)


class Function:
    __slots__ = "func"

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


UnaryFunction = type("UnaryFunction", (Function,), {})
BinaryFunction = type("BinaryFunction", (Function,), {})
DMABinaryFunction = type("DMABinaryFunction", (BinaryFunction,), {})


class TensorMemory(list):
    def __init__(self, iterable=()):
        assert all([isinstance(x, torch.Tensor) for x in iterable])
        super().__init__(iterable)

    def append(self, item):
        assert isinstance(item, torch.Tensor)
        list.append(self, item)

    def __getitem__(self, idx):
        if idx < self.__len__():
            return list.__getitem__(self, idx)
        else:
            if self.__len__() == 0:
                raise ValueError("Empty memory: Can not determine the type from first item")
            while not self.__len__() > idx:
                tensor = list.__getitem__(self, 0)
                self.append(torch.zeros_like(tensor))
            return self[idx]


@dataclass
class Instruction:
    __slots__ = "op", "in1", "in2", "out"
    op: Callable
    in1: Pointer
    in2: Optional[Pointer]
    out: Pointer

    def execute(self, memory):
        if isinstance(self.op, UnaryFunction):
            output = self.op(memory[self.in1])
        elif type(self.op) == BinaryFunction:
            output = self.op(memory[self.in1], memory[self.in2])
        elif isinstance(self.op, DMABinaryFunction):
            output = self.op(memory[self.in1], memory[self.in2], memory)
        memory[self.out].data = output.data
        return output


def create_optimizer(program, default_lr=1e-3):
    class Optimizer(torch.optim.Optimizer):
        def __init__(self, params, lr=default_lr):
            defaults = dict(lr=lr)
            self.memory = {}
            super(Optimizer, self).__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        state = self.state[p]
                        if len(state) == 0:
                            # Initialize vector memory
                            beta1 = torch.tensor(0.9)
                            beta2 = torch.tensor(0.999)
                            eps = torch.tensor(1e-8)
                            weight_decay = torch.tensor(1e-2)
                            state["step"] = torch.tensor(0.0)
                            self.memory[p] = TensorMemory(
                                [
                                    p,
                                    p.grad,
                                    state["step"],
                                    beta1,
                                    beta2,
                                    weight_decay,
                                    eps,
                                ]
                            )

                        state["step"] += 1

                        d_p = 0.0  # If program is empty no updates

                        # Execute the program
                        for instruction in program:
                            assert instruction.out > 6
                            d_p = instruction.execute(self.memory[p])

                        # Update weights
                        p.add_(d_p, alpha=-self.defaults["lr"])
            return loss

    return Optimizer


def _interpolate(x1, x2, memory, beta_loc):
    beta1 = memory[beta_loc]
    step = memory[2]
    bias_correction = 1 - torch.pow(beta1, step)
    y = x1 * beta1 + x2 * (1 - beta1)
    return y / bias_correction


def interpolate1(x1, x2, memory):
    return _interpolate(x1, x2, memory, 3)


def interpolate2(x1, x2, memory):
    return _interpolate(x1, x2, memory, 4)


if __name__ == "__main__":
    # Memory positions (0-6) are reserved
    # AdamW
    program = Program(
        [
            Instruction(UnaryFunction(torch.square), Pointer(1), None, Pointer(7)),
            Instruction(DMABinaryFunction(interpolate1), Pointer(1), Pointer(8), Pointer(8)),
            Instruction(DMABinaryFunction(interpolate2), Pointer(7), Pointer(9), Pointer(9)),
            Instruction(UnaryFunction(torch.sqrt), Pointer(9), None, Pointer(7)),
            Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(6), Pointer(7)),
            Instruction(BinaryFunction(torch.div), Pointer(8), Pointer(7), Pointer(10)),
            Instruction(BinaryFunction(torch.mul), Pointer(0), Pointer(5), Pointer(11)),
            Instruction(BinaryFunction(torch.add), Pointer(10), Pointer(11), Pointer(10)),
        ]
    )

    torch.manual_seed(123)
    model = torch.nn.Linear(1, 1)
    optimizer_class = create_optimizer(program)
    optim = optimizer_class(model.parameters(), lr=0.01)
    # optim = torch.optim.RMSprop(model.parameters(), lr=0.01)
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
