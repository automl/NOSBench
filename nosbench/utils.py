from functools import wraps

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
