import pprint
from collections import namedtuple

from noslib import NOSLib
from noslib.program import bruteforce_optimize


_Element = namedtuple("_Element", "cls fitness")


if __name__ == "__main__":
    nos = NOSLib()

    cs = nos.configspace(seed=123)
    history = []

    for i in range(20000):
        config = cs.sample_configuration()
        program = nos.configuration_to_program(config)
        fitness = -nos.query(program)
        history.append(_Element(program, fitness))
        x = max(history, key=lambda x: x.fitness)
        print(f"Step: {i+1}, Fitness: {x.fitness}")
        if x.fitness == 0.0:
            break

    x = max(history, key=lambda x: x.fitness)
    pprint.pprint(bruteforce_optimize(x.cls))
