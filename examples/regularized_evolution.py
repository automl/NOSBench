import abc
from functools import partial
from collections import namedtuple
import copy
import pprint
import argparse
from pathlib import Path
import json
import time
import pickle

from nosbench.program import (
    Instruction,
    Pointer,
    READONLY_REGION,
)
import nosbench


import numpy as np


_Element = namedtuple("_Element", "cls fitness")


MAX_MEMORY = 19


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
        benchmark,
        benchmark_epochs=50,
        initial_program=nosbench.optimizers.AdamW,
        rng=np.random.RandomState(),
        **kwargs,
    ):
        self.benchmark = benchmark
        self.benchmark_epochs = benchmark_epochs
        self.initial_program = initial_program
        self.mutations = [
            self.add_instruction_mutation,
            self.remove_instruction_mutation,
            self.modify_instruction_mutation,
        ]
        super().__init__(population_size, tournament_size, rng, **kwargs)

    def evaluate_element(self, element, **kwargs):
        return -self.benchmark.query(element, self.benchmark_epochs)

    def random_element(self, rng, **kwargs):
        return copy.deepcopy(self.initial_program)

    def mutate_element(self, element, rng, **kwargs):
        mutation_type = rng.randint(0, len(self.mutations))
        return self.mutations[mutation_type](element, rng)

    def add_instruction_mutation(self, program, rng):
        op = self.benchmark.ops[rng.randint(0, len(self.benchmark.ops))]
        inputs = [Pointer(rng.randint(0, MAX_MEMORY)) for _ in range(op.n_args)]
        output = rng.randint(READONLY_REGION + 1, MAX_MEMORY)
        instruction = Instruction(op, inputs, Pointer(output))
        pos = rng.randint(0, len(program) + 1)
        program.insert(pos, instruction)
        return program

    def remove_instruction_mutation(self, program, rng):
        if len(program):
            pos = rng.randint(0, len(program))
            program.pop(pos)
        return program

    def modify_instruction_mutation(self, program, rng):
        if len(program):
            pos = rng.randint(0, len(program))
            instruction = program[pos]
            if rng.randint(0, 2) == 0:
                input_idx = rng.randint(0, instruction.op.n_args)
                instruction.inputs[input_idx] = rng.randint(0, MAX_MEMORY)
            else:
                instruction.output = rng.randint(READONLY_REGION + 1, MAX_MEMORY)
        return program


class ProgramArgparseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        program = getattr(nosbench.optimizers, values)
        setattr(namespace, self.dest, program)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--cache_path", type=str, default="cache")
    parser.add_argument("--benchmark_name", type=str, default="toy")
    parser.add_argument("--benchmark_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--tournament_size", type=int, default=25)
    parser.add_argument("--evaluations", type=int, default=100000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--initial_program", type=str, default="AdamW")
    args = parser.parse_args()

    benchmark = nosbench.create(args.benchmark_name, path=args.cache_path)

    if args.initial_program == "random":
        cs = benchmark.configspace(seed=args.seed)
        config = cs.sample_configuration()
        initial_program = benchmark.configuration_to_program(config)
    else:
        initial_program = getattr(nosbench.optimizers, args.initial_program)

    re = RE_NOS(
        args.population_size,
        args.tournament_size,
        rng=np.random.RandomState(args.seed),
        initial_program=initial_program,
        benchmark=benchmark,
        benchmark_epochs=args.benchmark_epochs,
    )

    timestr = time.strftime("%Y-%m-%d")
    settings = {"search_algorithm": "RE", "args": vars(args)}
    dump = json.dumps(settings, sort_keys=True)
    path = Path(args.output_path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{timestr}-{hash(dump)}.json", "w") as f:
        f.write(dump)

    for i in range(args.population_size, args.evaluations):
        x = max(re.history, key=lambda x: x.fitness)
        print(f"Evaluation: {i+1}, Fitness: {x.fitness}")
        re.step()

        if ((i % args.save_every) == 0 and i > 0) or (i >= args.evaluations - 1):
            with open(path / f"{timestr}-{hash(dump)}.pickle", "wb") as f:
                pickle.dump(re.history, f)

    print(f"Number of Queries: {benchmark.stats.n_queries}, Hits: {benchmark.stats.hits}")
    x = max(re.history, key=lambda x: x.fitness)
    print("Incumbent optimizer:")
    pprint.pprint(x.cls)
