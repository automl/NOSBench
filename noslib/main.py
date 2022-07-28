import copy

import zero
import noslib
from regularized_evolution import RegularizedEvolution
from optimizers import AdamW

import torch
import numpy as np


MAX_MEMORY = 20

ops = [
    zero.BinaryFunction(torch.div),
    zero.BinaryFunction(torch.mul),
    zero.BinaryFunction(torch.add),
    zero.BinaryFunction(torch.sub),
    zero.UnaryFunction(torch.square),
    zero.UnaryFunction(torch.exp),
    zero.UnaryFunction(torch.sign),
    zero.UnaryFunction(torch.sqrt),
]


def add_instruction_mutation(program, rng):
    op = ops[rng.randint(0, len(ops))]
    in1 = rng.randint(0, MAX_MEMORY)
    in2 = None
    if isinstance(op, zero.BinaryFunction):
        in2 = rng.randint(0, MAX_MEMORY)
    out = rng.randint(7, MAX_MEMORY)
    instruction = zero.Instruction(op, zero.Pointer(in1), zero.Pointer(in2), zero.Pointer(out))
    pos = rng.randint(0, len(program) + 1)
    program.insert(pos, instruction)
    return program


def remove_instruction_mutation(program, rng):
    if len(program):
        pos = rng.randint(0, len(program))
        program.pop(pos)
    return program


def modify_instruction_mutation(program, rng):
    if len(program):
        pos = rng.randint(0, len(program))
        instruction = program[pos]
        if rng.randint(0, 2) == 0:
            input_idx = rng.randint(1, 2 if isinstance(instruction, zero.UnaryFunction) else 3)
            setattr(instruction, f"in{input_idx}", rng.randint(0, MAX_MEMORY))
        else:
            instruction.out = rng.randint(7, MAX_MEMORY)
    return program


MUTATIONS = [add_instruction_mutation, remove_instruction_mutation, modify_instruction_mutation]


class NOS(RegularizedEvolution):
    def __init__(self, population_size, tournament_size, rng=np.random.RandomState(), **kwargs):
        self.noslib = noslib.NOSLib()
        super().__init__(population_size, tournament_size, rng, **kwargs)

    def evaluate_element(self, element, **kwargs):
        return -self.noslib.query(element)

    @staticmethod
    def random_element(rng, **kwargs):
        return copy.deepcopy(AdamW)

    @staticmethod
    def mutate_element(element, rng, **kwargs):
        mutation_type = rng.randint(0, len(MUTATIONS))
        return MUTATIONS[mutation_type](element, rng)


def main():
    import pprint

    re = NOS(100, 25, rng=np.random.RandomState(123))
    for i in range(1000):
        x = max(re.history, key=lambda x: x.fitness)
        print(f"Step: {i+1}, Fitness: {x.fitness}")
        if x.fitness == 0.0:
            break
        re.step()

    x = max(re.history, key=lambda x: x.fitness)
    pprint.pprint(zero.bruteforce_optimize(x.cls))


if __name__ == "__main__":
    main()


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

# [Instruction(op=sign, in1=6, in2=None, out=11),
#  Instruction(op=square, in1=1, in2=None, out=7),
#  Instruction(op=interpolate1, in1=8, in2=1, out=8),
#  Instruction(op=interpolate2, in1=9, in2=11, out=9),
#  Instruction(op=bias_correct1, in1=8, out=10),
#  Instruction(op=sqrt, in1=14, in2=None, out=7),
#  Instruction(op=bias_correct2, in1=9, in2=None, out=11),
#  Instruction(op=mul, in1=0, in2=7, out=7),
#  Instruction(op=sqrt, in1=11, in2=None, out=7),
#  Instruction(op=add, in1=7, in2=6, out=18),
#  Instruction(op=square, in1=6, in2=None, out=9),
#  Instruction(op=div, in1=1, in2=3, out=15),
#  Instruction(op=mul, in1=5, in2=5, out=18),
#  Instruction(op=sqrt, in1=13, in2=0, out=15),
#  Instruction(op=div, in1=10, in2=7, out=7),
#  Instruction(op=add, in1=15, in2=11, out=16),
#  Instruction(op=div, in1=1, in2=16, out=19)]
#
# v = interpolate(v, 1, beta=0.999)
# v_hat = bias_correct(v, beta=0.999)
# v = eps ** 2
# update = grad / v_hat
