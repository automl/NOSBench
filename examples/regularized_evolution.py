import abc
from functools import partial
from collections import namedtuple
import copy
import pprint

from nosbench.program import (
    Instruction,
    Pointer,
    READONLY_REGION,
)
from nosbench.nosbench import NOSBench, OPS, MAX_MEMORY
from nosbench.optimizers import AdamW, SGD


import numpy as np


_Element = namedtuple("_Element", "cls fitness")


class RegularizedEvolution(abc.ABC):
    def __init__(
        self, population_size, tournament_size, rng=np.random.RandomState(), **kwargs
    ):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.rng = rng
        self._evaluate_element = partial(self.evaluate_element, rng=rng, **kwargs)
        self._mutate_element = partial(self.mutate_element, rng=rng, **kwargs)
        self._random_element = partial(self.random_element, rng=rng, **kwargs)

        self.population = []
        self.history = []
        for _ in range(self.population_size):
            element = self._random_element()
            element = _Element(element, self._evaluate_element(element))
            self.population.append(element)
            self.history.append(element)

    @staticmethod
    @abc.abstractmethod
    def evaluate_element(element, rng, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def mutate_element(element, rng, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def random_element(rng, **kwargs):
        pass

    def step(self):
        idxs = self.rng.randint(0, self.population_size, self.tournament_size)
        samples = [self.population[idx] for idx in idxs]
        parent = max(samples, key=lambda x: x.fitness)
        child = self._mutate_element(copy.deepcopy(parent.cls))
        child = _Element(child, self._evaluate_element(child))
        self.population.append(child)
        self.history.append(child)
        self.population.pop(0)


class RE_NOS(RegularizedEvolution):
    def __init__(
        self,
        population_size,
        tournament_size,
        initial_program=AdamW,
        rng=np.random.RandomState(),
        **kwargs,
    ):
        self.benchmark = NOSBench()
        self.initial_program = initial_program
        super().__init__(population_size, tournament_size, rng, **kwargs)

    def evaluate_element(self, element, **kwargs):
        return -self.benchmark.query(element, 10)

    def random_element(self, rng, **kwargs):
        return copy.deepcopy(self.initial_program)

    @staticmethod
    def mutate_element(element, rng, **kwargs):
        mutation_type = rng.randint(0, len(MUTATIONS))
        return MUTATIONS[mutation_type](element, rng)


def add_instruction_mutation(program, rng):
    op = OPS[rng.randint(0, len(OPS))]
    inputs = [Pointer(rng.randint(0, MAX_MEMORY)) for _ in range(op.n_args)]
    output = rng.randint(READONLY_REGION + 1, MAX_MEMORY)
    instruction = Instruction(op, inputs, Pointer(output))
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
            input_idx = rng.randint(0, instruction.op.n_args)
            instruction.inputs[input_idx] = rng.randint(0, MAX_MEMORY)
        else:
            instruction.output = rng.randint(READONLY_REGION + 1, MAX_MEMORY)
    return program


MUTATIONS = [
    add_instruction_mutation,
    remove_instruction_mutation,
    modify_instruction_mutation,
]


if __name__ == "__main__":
    re = RE_NOS(100, 25, rng=np.random.RandomState(123), initial_program=AdamW)
    for i in range(20000):
        x = max(re.history, key=lambda x: x.fitness)
        print(f"Step: {i+1}, Fitness: {x.fitness}")
        if x.fitness == 0.0:
            break
        re.step()

    x = max(re.history, key=lambda x: x.fitness)
    print("Incumbent optimizer:")
    pprint.pprint(x.cls)
