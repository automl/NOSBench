import pprint
from collections import namedtuple
import json
from pathlib import Path
import pickle
import time
import argparse

from smac import BlackBoxFacade, Scenario
from smac.initial_design import RandomInitialDesign
import numpy as np

import nosbench

# Requires SMAC >= 2.0.0

_Element = namedtuple("_Element", "cls fitness")

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="results")
parser.add_argument("--benchmark_name", type=str, default="toy")
parser.add_argument("--benchmark_epochs", type=int, default=50)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--evaluations", type=int, default=100000)
args = parser.parse_args()

benchmark = nosbench.create(args.benchmark_name)

timestr = time.strftime("%Y-%m-%d")
settings = {"search_algorithm": "SMAC_BO", "args": vars(args)}
dump = json.dumps(settings, sort_keys=True)
path = Path(args.output_path)
path.mkdir(parents=True, exist_ok=True)
with open(path / f"{timestr}-{hash(dump)}.json", "w") as f:
    f.write(dump)

scenario = Scenario(
    configspace=benchmark.configspace(seed=args.seed),
    n_trials=args.evaluations,
    output_directory=path / f"{timestr}-{hash(dump)}.smac",
    deterministic=True,
)


def runner(config, seed):
    program = benchmark.configuration_to_program(config)
    loss = benchmark.query(program, args.benchmark_epochs)
    if np.isnan(loss) or np.isinf(loss):
        return 2147483648
    return loss


initial_design = RandomInitialDesign(scenario, n_configs=10)
smac = BlackBoxFacade(scenario, runner, initial_design=initial_design, overwrite=True)
config = smac.optimize()
print("Incumbent optimizer:")
pprint.pprint(benchmark.configuration_to_program(config))

history = []
for k, v in smac.runhistory.items():
    config = smac.runhistory.get_config(k.config_id)
    program = benchmark.configuration_to_program(config)
    fitness = -v.cost
    history.append(_Element(program, fitness))

with open(path / f"{timestr}-{hash(dump)}.pickle", "wb") as f:
    pickle.dump(history, f)
