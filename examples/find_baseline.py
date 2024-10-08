from functools import partial
from collections import namedtuple
import time
import json
from pathlib import Path
import pickle
import argparse

import torch
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario, Callback
from ConfigSpace import ConfigurationSpace, Float, Categorical, Beta, Normal, Integer
from ConfigSpace import InCondition, EqualsCondition

import nosbench

_Element = namedtuple("_Element", "cls fitness")

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="results")
parser.add_argument("--benchmark_epochs", type=int, default=50)
parser.add_argument("--save_every", type=int, default=10)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--evaluations", type=int, default=10000)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

benchmark = nosbench.NOSBench(device=args.device)

timestr = time.strftime("%Y-%m-%d")
settings = {"search_algorithm": "SMAC_Baseline", "args": vars(args)}
dump = json.dumps(settings, sort_keys=True)
path = Path(args.output_path)
path.mkdir(parents=True, exist_ok=True)
with open(path / f"{timestr}-{hash(dump)}.json", "w") as f:
    f.write(dump)
smac_cache = path / "smac_cache"
smac_cache.mkdir(parents=True, exist_ok=True)


class Program:
    def __init__(self, optimizer, *args, **kwargs):
        self.optim = optimizer
        self.args = args
        self.kwargs = kwargs

    def optimizer(self):
        return partial(self.optim, *self.args, **self.kwargs)

    def __hash__(self):
        return 42


def replace_lr(config, key):
    try:
        lr = config.pop(key)
        config["lr"] = lr
    except KeyError:
        pass


class CustomCallback(Callback):
    def __init__(self) -> None:
        self.trials_counter = 0

    def on_start(self, smbo):
        print("Start")

    def on_tell_end(self, smac, info, value) -> bool | None:
        global path
        global args
        global benchmark
        global timestr
        global dump

        self.trials_counter += 1

        if (
            (self.trials_counter % args.save_every) == 0 and self.trials_counter > 0
        ) or (self.trials_counter >= args.evaluations - 1):
            history = []
            for k, v in smac.runhistory.items():
                config = smac.runhistory.get_config(k.config_id)
                fitness = -v.cost
                history.append(_Element(config, fitness))

            with open(path / f"{timestr}-{hash(dump)}.pickle", "wb") as f:
                pickle.dump(history, f)

        return None


cs = ConfigurationSpace(seed=args.seed)

optimizer = Categorical(
    "optimizer", ["SGD", "Adam", "AdamW", "Adadelta", "Adagrad", "RMSprop"]
)

# Common parameters
weight_decay = Float("weight_decay", bounds=(1e-15, 1.0), default=1e-8, log=True)
lr = Float("lr", bounds=(1e-8, 5.0), default=0.001, log=True)

# Momentum
momentum = Float("momentum", bounds=(1e-15, 1.0), default=1e-8, log=True)

# Adam AdamW
inv_beta1 = Float("inv_beta1", bounds=(1e-5, 0.99999), default=0.1, log=True)
inv_beta2 = Float("inv_beta2", bounds=(1e-5, 0.99999), default=0.001, log=True)

# Adadelta
rho = Float("inv_rho", bounds=(1e-5, 0.99999), default=0.1, log=True)

# Adagrad
lr_decay = Float("lr_decay", bounds=(1e-15, 1.0), default=1e-8, log=True)

# RMSprop
alpha = Float("inv_alpha", bounds=(1e-5, 0.99999), default=0.01, log=True)

cs.add_hyperparameters(
    [optimizer, weight_decay, lr, momentum, inv_beta1, inv_beta2, rho, lr_decay, alpha]
)

# Momentum Condition
momentum_cond = InCondition(momentum, optimizer, ["SGD", "RMSprop"])

# Adagrad Conditions
lr_decay_cond = EqualsCondition(lr_decay, optimizer, "Adagrad")

# Adadelta Conditions
rho_cond = EqualsCondition(rho, optimizer, "Adadelta")

# AdamW Adam Conditions
beta1_cond = InCondition(inv_beta1, optimizer, ["Adam", "AdamW"])
beta2_cond = InCondition(inv_beta2, optimizer, ["Adam", "AdamW"])

alpha_cond = EqualsCondition(alpha, optimizer, "RMSprop")

cs.add_conditions(
    [momentum_cond, lr_decay_cond, rho_cond, beta1_cond, beta2_cond, alpha_cond]
)


def train(config: Configuration, seed: int = 0):
    config = dict(config)
    try:
        beta1 = config.pop("inv_beta1")
        beta2 = config.pop("inv_beta2")
        config["betas"] = (1 - beta1, 1 - beta2)
    except KeyError:
        pass

    try:
        rho = config.pop("inv_rho")
        config["rho"] = 1 - rho
    except KeyError:
        pass

    try:
        alpha = config.pop("inv_alpha")
        config["alpha"] = 1 - alpha
    except KeyError:
        pass

    optimizer = config.pop("optimizer")
    optimizer = getattr(torch.optim, optimizer)
    print(optimizer, config)

    program = Program(optimizer, **config)

    loss = benchmark.query(program, 19, skip_cache=True)
    if np.isnan(loss) or np.isinf(loss):
        return 2147483648
    return loss


scenario = Scenario(
    cs,
    deterministic=True,
    seed=0,
    n_trials=args.evaluations,
    output_directory=path / "output.smac",
)
smac = HyperparameterOptimizationFacade(scenario, train, callbacks=[CustomCallback()])
incumbent = smac.optimize()

history = []
for k, v in smac.runhistory.items():
    config = smac.runhistory.get_config(k.config_id)
    fitness = -v.cost
    history.append(_Element(config, fitness))

with open(path / f"{timestr}-{hash(dump)}.pickle", "wb") as f:
    pickle.dump(history, f)
