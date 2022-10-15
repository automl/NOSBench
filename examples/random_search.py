import pprint
from collections import namedtuple

from nosbench import NOSBench


_Element = namedtuple("_Element", "cls fitness")


if __name__ == "__main__":
    benchmark = NOSBench()

    cs = benchmark.configspace(seed=123)
    history = []

    for i in range(20000):
        config = cs.sample_configuration()
        program = benchmark.configuration_to_program(config)
        fitness = -benchmark.query(program, 10)
        history.append(_Element(program, fitness))
        x = max(history, key=lambda x: x.fitness)
        print(f"Step: {i+1}, Fitness: {x.fitness}")
        if x.fitness == 0.0:
            break

    x = max(history, key=lambda x: x.fitness)
    pprint.pprint(x.cls)
