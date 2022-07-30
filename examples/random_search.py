import copy
import pprint

import numpy as np

from examples.regularized_evolution import MUTATIONS, _Element
from noslib import NOSLib
from noslib.program import bruteforce_optimize
from noslib.optimizers import AdamW


if __name__ == "__main__":
    nos = NOSLib()

    cs = nos.configspace(seed=123)
    history = []

    for i in range(2000):
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
