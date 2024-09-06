from functools import partial
from collections import namedtuple
import time
import json
from pathlib import Path
import pickle

import torch
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace import ConfigurationSpace, Float, Categorical, Beta, Normal

import nosbench


_Element = namedtuple("_Element", "cls fitness")


class Program:
    def __init__(self, optimizer, *args, **kwargs):
        self.optim = optimizer
        self.args = args
        self.kwargs = kwargs

    def optimizer(self):
        return partial(self.optim, *self.args, **self.kwargs)

    def __hash__(self):
        return 42


if __name__ == "__main__":
    benchmark = nosbench.NOSBench(device="cuda")

    output_path = "results"
    optimizer = torch.optim.Adam

    timestr = time.strftime("%Y-%m-%d")
    settings = {"search_algorithm": f"SMAC_{optimizer.__name__}"}
    dump = json.dumps(settings, sort_keys=True)
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{timestr}-{hash(dump)}.json", "w") as f:
        f.write(dump)


    def train(config: Configuration, seed: int = 0):
        program = Program(optimizer, **config)
        return benchmark.query(program, 19, skip_cache=True)

    logmean = np.log(1e-3)
    logstd = np.log(10.0)

    configspace = ConfigurationSpace(
        seed=123,
        space={
          "lr": Float('lr', bounds=(1e-5, 1e-1), default=1e-3, log=True, distribution=Normal(logmean, logstd)),
        }
    )    
    scenario = Scenario(configspace, deterministic=True, n_trials=200)
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()
    print(incumbent)

    history = []
    for k, v in smac.runhistory.items():
        config = smac.runhistory.get_config(k.config_id)
        fitness = -v.cost
        history.append(_Element(config, fitness))

    with open(path / f"{timestr}-{hash(dump)}.pickle", "wb") as f:
        pickle.dump(history, f)
