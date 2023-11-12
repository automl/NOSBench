from typing import Callable, NewType, Type, List
from dataclasses import dataclass
from itertools import chain
from functools import cached_property
import copy

import sklearn.datasets
import torch


from nosbench.utils import deterministic


Pointer = NewType("Pointer", int)


READONLY_REGION = 8


class Program(list):
    @staticmethod
    def _sphere(data):
        return torch.sum(data**2)

    @staticmethod
    def _rosenbrock(data):
        return torch.sum(100 * (data[1:] - data[:-1] ** 2) ** 2 + (1 - data[:-1]) ** 2)

    @cached_property
    def dataset(self):
        dataset = sklearn.datasets.load_iris()
        data = torch.from_numpy(dataset.data).float()
        target = torch.from_numpy(dataset.target).long()
        return data, target

    def __eq__(self, other):
        return hash(self) == hash(other)

    @deterministic(seed=42)
    def __hash__(self):
        data, target = self.dataset
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Linear(8, 3),
            torch.nn.LogSoftmax(-1),
        )
        optimizer_class = self.optimizer()
        optimizer = optimizer_class(model.parameters())

        loss = torch.nn.functional.nll_loss(model(data), target)
        exp_avg = loss.item()
        for _ in range(10):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = torch.nn.functional.nll_loss(model(data), target)
            if torch.isinf(loss):
                return -3
            elif torch.isnan(loss):
                return -2
            exp_avg = loss.item() * 0.999 + exp_avg * 0.001
        return hash(exp_avg)

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
