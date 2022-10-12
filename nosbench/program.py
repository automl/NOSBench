import copy
from typing import Callable, NewType, Type, List
from dataclasses import dataclass
from itertools import chain

import torch


Pointer = NewType("Pointer", int)


READONLY_REGION = 8


class Program(list):
    @staticmethod
    def _sphere(data):
        return torch.sum(data**2)

    @staticmethod
    def _rosenbrock(data):
        return torch.sum(100 * (data[1:] - data[:-1] ** 2) ** 2 + (1 - data[:-1]) ** 2)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        generator = torch.Generator().manual_seed(42)
        data = torch.randn(100, generator=generator)
        params = torch.nn.Parameter(data)
        optimizer_class = self.optimizer()
        optim = optimizer_class([params])

        output_string = ""
        output = self._sphere(params)
        for _ in range(5):
            optim.zero_grad()
            output.backward()
            optim.step()
            output = self._sphere(params)
            if torch.isnan(output):
                return -2
            if torch.isinf(output):
                return -3
            output_string += f"{output:.4f}"
        return int(output_string.replace(".", "")[:16])

    def optimizer(self) -> Type[torch.optim.Optimizer]:
        return create_optimizer(self)


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
        return f"Instruction(Function({self.op}, {self.op.n_args}), inputs={self.inputs}, output={self.output})"

    def __repr__(self):
        return str(self)


class _TensorMemory(list):
    def __init__(self, iterable=[]):
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

    def __setitem__(self, idx, value):
        if idx < self.__len__():
            return list.__setitem__(self, idx, value)
        else:
            while not self.__len__() > idx:
                self.append(torch.tensor(0.0))
            self[idx] = value


def create_optimizer(program):
    class Optimizer(torch.optim.Optimizer):
        def __init__(self, params):
            # Hyperparameters of the optimizer are part of the program
            self.memory = {}
            super(Optimizer, self).__init__(params, {})

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict)
            groups = self.param_groups
            saved_groups = state_dict["param_groups"]

            for old_id, p in zip(
                chain.from_iterable((g["params"] for g in saved_groups)),
                chain.from_iterable((g["params"] for g in groups)),
            ):
                self.memory[p] = state_dict["state"][old_id]["memory"]

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group_id, group in enumerate(self.param_groups):
                for param_id, p in enumerate(group["params"]):
                    if p.grad is not None:
                        state = self.state[p]
                        if len(state) == 0:
                            # Initialize vector memory
                            state["step"] = torch.tensor(0.0)
                            self.memory[p] = _TensorMemory()
                            state["memory"] = self.memory[p]

                        self.memory[p][0] = p
                        self.memory[p][1] = p.grad
                        self.memory[p][2] = state["step"]
                        self.memory[p][3] = torch.tensor(1.0)
                        self.memory[p][4] = torch.tensor(0.5)
                        self.memory[p][5] = torch.tensor(1e-01)
                        self.memory[p][6] = torch.tensor(1e-02)
                        self.memory[p][7] = torch.tensor(1e-03)
                        self.memory[p][8] = torch.tensor(1e-06)

                        state["step"] += 1

                        d_p = 0.0  # If program is empty no updates

                        # Execute the program
                        for instruction in program:
                            assert instruction.output > READONLY_REGION
                            d_p = instruction.execute(self.memory[p])

                        # Update weights
                        p.add_(-d_p)
            return loss

    return Optimizer
