import copy
from typing import Callable, NewType, Optional, Type
from dataclasses import dataclass

import torch

from noslib.function import (
    UnaryFunction,
    BinaryFunction,
    DMAUnaryFunction,
    DMABinaryFunction,
)


Pointer = NewType("Pointer", int)


READONLY_REGION = 6
MAX_MEMORY = 20


class Program(list):
    @staticmethod
    def _rosenbrock(data):
        return torch.sum(100 * (data[1:] - data[:-1] ** 2) ** 2 + (1 - data[:-1]) ** 2)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        params = torch.nn.Parameter(-torch.ones(32))
        optimizer_class = self.optimizer(default_lr=0.0001)
        optim = optimizer_class([params])
        output_string = ""
        for _ in range(50):
            output = self._rosenbrock(params)
            optim.zero_grad()
            output.backward()
            optim.step()
            if torch.isnan(output):
                return -2
            if torch.isinf(output):
                return -3
            output_string += f"{output:.5f}".replace(".", "")
        return int(output_string.replace("0", "")[-18:])

    def optimizer(self, default_lr=1e-3) -> Type[torch.optim.Optimizer]:
        return create_optimizer(self, default_lr)


@dataclass
class Instruction:
    __slots__ = "op", "in1", "in2", "out"
    op: Callable
    in1: Pointer
    in2: Optional[Pointer]
    out: Pointer

    def execute(self, memory):
        if type(self.op) == UnaryFunction:
            output = self.op(memory[self.in1])
        elif type(self.op) == DMAUnaryFunction:
            output = self.op(memory[self.in1], memory)
        elif type(self.op) == BinaryFunction:
            output = self.op(memory[self.in1], memory[self.in2])
        elif isinstance(self.op, DMABinaryFunction):
            output = self.op(memory[self.in1], memory[self.in2], memory)
        memory[self.out].data = output.data
        return output

    def __str__(self):
        in2 = f"in2={self.in2}, " if self.in2 is not None else ""
        return f"{self.op}(in1={self.in1}, {in2}out={self.out})"

    def __repr__(self):
        return str(self)


class _TensorMemory(list):
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


# TODO: Detect use of other hyperparameters and include in the constructor
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
                            self.memory[p] = _TensorMemory(
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
                            assert instruction.out > READONLY_REGION
                            d_p = instruction.execute(self.memory[p])

                        # Update weights
                        p.add_(d_p, alpha=-self.defaults["lr"])
            return loss

    return Optimizer


def bruteforce_optimize(program):
    i = 0
    while i < len(program):
        program_copy = copy.deepcopy(program)
        program_copy.pop(i)
        if program_copy == program:
            program = program_copy
        else:
            i += 1
    return program
