import copy
from typing import Callable, NewType, Optional, Type, List
from dataclasses import dataclass

import torch


Pointer = NewType("Pointer", int)


READONLY_REGION = 8


class Program(list):
    @staticmethod
    def _sphere(data):
        return torch.sum(data**2)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        generator = torch.Generator().manual_seed(42)
        data = torch.randn(100, generator=generator)
        params = torch.nn.Parameter(data)
        optimizer_class = self.optimizer(default_lr=0.1)
        optim = optimizer_class([params])

        output = self._sphere(params)
        optim.zero_grad()
        output.backward()
        optim.step()
        output = self._sphere(params)

        if torch.isnan(output):
            return -2
        if torch.isinf(output):
            return -3
        output_string = f"{output:.17f}".replace(".", "")
        return int(output_string[:18])

    def optimizer(self, default_lr=1e-3) -> Type[torch.optim.Optimizer]:
        return create_optimizer(self, default_lr)


@dataclass
class Instruction:
    __slots__ = "op", "inputs", "output"
    op: Callable
    inputs: List[Pointer]
    output: Pointer

    def execute(self, memory):
        output = self.op([memory[inp] for inp in self.inputs])
        memory[self.output].data = output.data
        return output

    def __str__(self):
        return f"{self.op}(inputs={self.inputs}, output={self.output})"

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
            while not self.__len__() > idx:
                self.append(torch.tensor(0.0))
            return self[idx]


# TODO: Detect use of other hyperparameters and include in the constructor
def create_optimizer(program, default_lr=1e-3):
    class Optimizer(torch.optim.Optimizer):
        def __init__(self, params, lr=default_lr):
            defaults = dict(lr=lr)
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
                            state["step"] = torch.tensor(0.0)
                            state["memory"] = _TensorMemory(
                                [
                                    p,
                                    p.grad,
                                    state["step"],
                                    torch.tensor(1.0),
                                    torch.tensor(0.5),
                                    torch.tensor(1e-01),
                                    torch.tensor(1e-02),
                                    torch.tensor(1e-03),
                                    torch.tensor(1e-06),
                                ]
                            )

                        state["step"] += 1

                        d_p = 0.0  # If program is empty no updates

                        # Execute the program
                        for instruction in program:
                            assert instruction.output > READONLY_REGION
                            d_p = instruction.execute(state["memory"])

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
