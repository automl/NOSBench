from functools import wraps
from itertools import chain

import torch


def deterministic(seed):
    def _wrapper(f):
        @wraps(f)
        def func(*args, **kwargs):
            random_state = torch.get_rng_state()
            torch.manual_seed(seed)
            value = f(*args, **kwargs)
            torch.set_rng_state(random_state)
            return value

        return func

    return _wrapper


def prune_program(prog):
    while True:
        remove = set()
        for i, instruction in reversed(list(enumerate(prog))):
            for next in chain(prog[i + 1 :], prog):
                if instruction.output in next.inputs:
                    break
                if instruction.output == next.output:
                    if i != len(prog) - 1:
                        remove.add(i)
        for index in sorted(list(remove),reverse=True):
            prog.pop(index)
        if len(remove) == 0:
            return
