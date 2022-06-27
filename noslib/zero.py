from dataclasses import dataclass
from typing import Optional, Callable
from collections import namedtuple, defaultdict
from enum import IntEnum
from functools import partial

import torch
from torch import nn


Pointer = namedtuple("Pointer", "name index")
Program = namedtuple("Program", "setup step")


class MemoryType(IntEnum):
    SCALAR = 0
    VECTOR = 1


class TensorMemory(list):
    def __init__(self, iterable=(), initializer=lambda: torch.tensor(0)):
        self.initializer = initializer
        assert all([isinstance(x, torch.Tensor) for x in iterable])
        super().__init__(iterable)

    def append(self, item):
        assert isinstance(item, torch.Tensor)
        list.append(self, item)

    def __getitem__(self, item):
        if item < self.__len__():
            return list.__getitem__(self, item)
        else:
            while not self.__len__() > item:
                self.append(self.initializer())
            return self[item]


@dataclass
class Instruction:
    __slots__ = "op", "in1", "in2", "out"
    op: Callable
    in1: Pointer
    in2: Optional[Pointer]
    out: Pointer

    def execute(self, memory):
        if self.in2 is None:
            output = self.op(memory[self.in1.name][self.in1.index])
        else:
            output = self.op(
                memory[self.in1.name][self.in1.index],
                memory[self.in2.name][self.in2.index],
            )
        memory[self.out.name][self.out.index].data = output.data
        return output


def create_optimizer(program, default_lr=1e-3):
    class Optimizer(torch.optim.Optimizer):
        def __init__(self, params, lr=default_lr):
            defaults = dict(lr=lr)

            def empty_bank():
                return [[] for _ in range(len(MemoryType))]

            self.memory = defaultdict(empty_bank)
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
                        memory = self.memory[p]
                        if len(state) == 0:
                            # Initialize vector memory
                            memory[MemoryType.VECTOR] = TensorMemory(
                                [p.grad, p], partial(torch.zeros_like, p.grad)
                            )

                            # Initialize scalar memory
                            state["step"] = torch.tensor(0.0)
                            memory[MemoryType.SCALAR] = TensorMemory(
                                [state["step"]], partial(torch.tensor, 1.0)
                            )

                            # Execute setup instructions
                            for instruction in program.setup:
                                instruction.execute(memory)

                        state["step"] += 1

                        # d_p = p.grad # If program is empty do an SGD update
                        d_p = 0.0

                        # Execute step instructions
                        for instruction in program.step:
                            d_p = instruction.execute(memory)

                        # Update weights
                        p.add_(d_p, alpha=-self.defaults["lr"])
            return loss

    return Optimizer


if __name__ == "__main__":
    # RMSProp
    program = Program(
        setup=[
            Instruction(
                torch.exp,
                Pointer(MemoryType.SCALAR, 5),
                None,
                Pointer(MemoryType.SCALAR, 6),
            ),
            Instruction(
                torch.div,
                Pointer(MemoryType.SCALAR, 5),
                Pointer(MemoryType.SCALAR, 6),
                Pointer(MemoryType.SCALAR, 1),
            ),
            Instruction(
                torch.sub,
                Pointer(MemoryType.SCALAR, 5),
                Pointer(MemoryType.SCALAR, 1),
                Pointer(MemoryType.SCALAR, 2),
            ),
            Instruction(
                torch.exp,
                Pointer(MemoryType.SCALAR, 2),
                None,
                Pointer(MemoryType.SCALAR, 7),
            ),
            Instruction(
                torch.sub,
                Pointer(MemoryType.SCALAR, 7),
                Pointer(MemoryType.SCALAR, 5),
                Pointer(MemoryType.SCALAR, 3),
            ),
            Instruction(
                torch.sub,
                Pointer(MemoryType.SCALAR, 5),
                Pointer(MemoryType.SCALAR, 3),
                Pointer(MemoryType.SCALAR, 4),
            ),
        ],
        step=[
            Instruction(
                torch.mul,
                Pointer(MemoryType.VECTOR, 0),
                Pointer(MemoryType.VECTOR, 0),
                Pointer(MemoryType.VECTOR, 2),
            ),
            Instruction(
                torch.mul,
                Pointer(MemoryType.VECTOR, 2),
                Pointer(MemoryType.SCALAR, 4),
                Pointer(MemoryType.VECTOR, 3),
            ),
            Instruction(
                torch.mul,
                Pointer(MemoryType.VECTOR, 4),
                Pointer(MemoryType.SCALAR, 3),
                Pointer(MemoryType.VECTOR, 4),
            ),
            Instruction(
                torch.add,
                Pointer(MemoryType.VECTOR, 4),
                Pointer(MemoryType.VECTOR, 3),
                Pointer(MemoryType.VECTOR, 4),
            ),
            Instruction(
                torch.sqrt,
                Pointer(MemoryType.VECTOR, 4),
                None,
                Pointer(MemoryType.VECTOR, 5),
            ),
            Instruction(
                torch.div,
                Pointer(MemoryType.VECTOR, 0),
                Pointer(MemoryType.VECTOR, 5),
                Pointer(MemoryType.VECTOR, 6),
            ),
        ],
    )

    torch.manual_seed(123)
    model = torch.nn.Linear(1, 1)
    optimizer_class = create_optimizer(program)
    optim = optimizer_class(model.parameters(), lr=0.01)
    # optim = torch.optim.RMSprop(model.parameters(), lr=0.01)
    initial_loss = torch.nn.functional.mse_loss(
        model(torch.tensor([1.0])), torch.tensor([1.0])
    ).item()
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
