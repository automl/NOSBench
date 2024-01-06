from typing import Callable, NewType, Type, List
from dataclasses import dataclass
from itertools import chain
from functools import cached_property
import copy
from collections import defaultdict
import weakref

import sklearn.datasets
import torch

from nosbench.utils import deterministic
from nosbench.device import Device

Pointer = NewType("Pointer", int)


READONLY_REGION = 8


class Program(list):
    __refs__: dict[str, list[Type["Program"]]] = defaultdict(list)

    def __init__(self, *args, **kwargs):
        self.__refs__[self.__class__].append(weakref.ref(self))
        super().__init__(*args, **kwargs)

    @classmethod
    def get_instances(cls):
        for inst_ref in cls.__refs__[cls]:
            inst = inst_ref()
            if inst is not None:
                yield inst

    def __eq__(self, other):
        return hash(self) == hash(other)

    @deterministic(seed=42)
    def __hash__(self):
        device = Device.get()
        Device.set("cpu")
        dim = 10
        a = torch.normal(torch.zeros(dim, dim), torch.ones(dim, dim) * 1e-1)
        b = torch.normal(torch.zeros(dim), torch.ones(dim) * 1e-1)
        x = torch.normal(torch.zeros(dim), torch.ones(dim) * 1e-1)
        y = -torch.ones(dim) / 2.0
        x.requires_grad_()

        optimizer_class = self.optimizer()
        optimizer = optimizer_class([x])

        # To eliminate 1/g optimizers
        optimizer.zero_grad()
        optimizer.step()

        exp_avg = 0.0
        try:
            for _ in range(10):
                z = x @ a + b
                yhat = z**2 / (z**2 + 1.0)
                loss = torch.nn.functional.l1_loss(yhat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if torch.isinf(loss):
                    return -3
                elif torch.isnan(loss):
                    return -2
                exp_avg = loss.item() * 0.1 + exp_avg * 0.9
        finally:
            Device.set(device)
        return hash(exp_avg)

    def optimizer(self) -> Type[torch.optim.Optimizer]:
        return create_optimizer(self)


class NamedProgram(Program):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)


@dataclass
class Instruction:
    __slots__ = "op", "inputs", "output"
    op: Callable
    inputs: List[Pointer]
    output: Pointer

    def execute(self, memory):
        device = Device.get()
        output = self.op([memory[inp].to(device) for inp in self.inputs])
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
            device = Device.get()
            while not self.__len__() > idx:
                self.append(torch.tensor(0.0, device=device))
            return self[idx]

    def __setitem__(self, idx, value):
        if idx < self.__len__():
            return list.__setitem__(self, idx, value)
        else:
            device = Device.get()
            while not self.__len__() > idx:
                self.append(torch.tensor(0.0, device=device))
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
            device = Device.get()
            for group_id, group in enumerate(self.param_groups):
                for param_id, p in enumerate(group["params"]):
                    if p.grad is not None:
                        state = self.state[p]
                        if len(state) == 0:
                            # Initialize vector memory
                            state["step"] = torch.tensor(0.0, device=device)
                            self.memory[p] = _TensorMemory()
                            state["memory"] = self.memory[p]

                        self.memory[p][0] = p
                        self.memory[p][1] = p.grad
                        self.memory[p][2] = state["step"]
                        self.memory[p][3] = torch.tensor(1.0, device=device)
                        self.memory[p][4] = torch.tensor(0.5, device=device)
                        self.memory[p][5] = torch.tensor(1e-01, device=device)
                        self.memory[p][6] = torch.tensor(1e-02, device=device)
                        self.memory[p][7] = torch.tensor(1e-03, device=device)
                        self.memory[p][8] = torch.tensor(1e-06, device=device)

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
