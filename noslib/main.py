import copy

from zero import Pointer, Instruction, Program
from zero import create_optimizer
from zero import UnaryFunction, BinaryFunction, DMABinaryFunction, DMAUnaryFunction
from regularized_evolution import RegularizedEvolution
from optimizers import AdamW

import torch
import numpy as np


MAX_MEMORY = 20

ops = [
    BinaryFunction(torch.div),
    BinaryFunction(torch.mul),
    BinaryFunction(torch.add),
    BinaryFunction(torch.sub),
    UnaryFunction(torch.square),
    UnaryFunction(torch.exp),
    UnaryFunction(torch.sign),
    UnaryFunction(torch.sqrt),
]


def random_step_instruction(rng, memory_size, ro_partition):
    op = ops[rng.randint(0, len(ops))]
    in1 = rng.randint(0, memory_size)
    in2 = None
    if isinstance(op, BinaryFunction):
        in2 = rng.randint(0, memory_size)
    out = rng.randint(ro_partition, memory_size)
    instruction = Instruction(op, Pointer(in1), Pointer(in2), Pointer(out))
    return instruction


def add_instruction_mutation(program, rng):
    instruction = random_step_instruction(rng, MAX_MEMORY, 7)
    pos = rng.randint(0, len(program) + 1)
    program.instructions.insert(pos, instruction)
    return program


def remove_instruction_mutation(program, rng):
    if len(program):
        pos = rng.randint(0, len(program))
        program.instructions.pop(pos)
    return program


class NOS(RegularizedEvolution):
    def __init__(self, population_size, tournament_size, rng=np.random.RandomState(), **kwargs):
        self.cache = {}
        super().__init__(population_size, tournament_size, rng, **kwargs)

    def evaluate_element(self, element, **kwargs):
        if element in self.cache:
            return self.cache[element]
        torch.manual_seed(123)
        model = torch.nn.Linear(2, 1)
        optimizer_class = create_optimizer(element)
        optim = optimizer_class(model.parameters(), lr=0.01)
        initial_loss = torch.nn.functional.mse_loss(model(torch.tensor([0.25, 0.25])), torch.tensor([10.0])).item()
        for _ in range(100):
            output = model(torch.tensor([0.25, -0.25]))
            loss = torch.nn.functional.mse_loss(output, torch.tensor([10.0]))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss = loss.item()
        self.cache[element] = -loss
        return -loss

    @staticmethod
    def random_element(rng, **kwargs):
        return copy.deepcopy(AdamW)

    @staticmethod
    def mutate_element(element, rng, **kwargs):
        mutation_type = rng.randint(0, 2)
        if mutation_type == 0:
            child = add_instruction_mutation(element, rng)
        elif mutation_type == 1:
            child = remove_instruction_mutation(element, rng)
        return child


re = NOS(100, 25, rng=np.random.RandomState(123))
for i in range(1000):
    x = max(re.history, key=lambda x: x.fitness)
    print(f"Step: {i+1}, Fitness: {x.fitness}")
    if x.fitness == 0.0:
        break
    re.step()

x = max(re.history, key=lambda x: x.fitness)
print(x.cls)

# [Instruction(op=sqrt, in1=18, in2=None, out=19),
#  Instruction(op=square, in1=18, in2=None, out=9),
#  Instruction(op=sub, in1=19, in2=9, out=7),
#  Instruction(op=sqrt, in1=10, in2=None, out=8),
#  Instruction(op=div, in1=17, in2=11, out=17),
#  Instruction(op=mul, in1=1, in2=2, out=17),
#  Instruction(op=sub, in1=4, in2=11, out=13),
#  Instruction(op=add, in1=17, in2=19, out=9)]

# [Instruction(op=mul, in1=1, in2=2, out=17),
#  Instruction(op=add, in1=17, in2=19, out=9)]

# update = grad * step

# [Instruction(op=square, in1=1, in2=None, out=7),
#  Instruction(op=interpolate1, in1=8, in2=1, out=8),
#  Instruction(op=sign, in1=7, in2=None, out=7),
#  Instruction(op=interpolate2, in1=9, in2=7, out=9),
#  Instruction(op=bias_correct1, in1=8, in2=None, out=10),
#  Instruction(op=bias_correct2, in1=9, in2=None, out=11),
#  Instruction(op=<sqrt, in1=11, in2=None, out=7),
#  Instruction(op=<add, in1=7, in2=6, out=7),
#  Instruction(op=<square, in1=6, in2=None, out=9),
#  Instruction(op=<div, in1=10, in2=7, out=7)]
#
# g2 = grad ** 2
# m = interpolate(m, g, beta=0.9)
# s_g2 = sign(g2)
# v = interpolate(v, s_g2, beta=0.999)
# m_hat = bias_correct(m, beta=0.9)
# v_hat = bias_correct(v, beta=0.999)
# v = eps ** 2
# update = v_hat / (sqrt(v_hat) + eps)
