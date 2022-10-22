import json
import pathlib
import argparse
from functools import singledispatch

import torch

from nosbench.optimizers import SGD, Adam, AdamW, RMSprop, Adagrad


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

path = pathlib.Path(args.path)

@singledispatch
def get_program(state):
    return state["program"]

@get_program.register
def _(state: list):
    return state[0]["program"]


for run_path in path.rglob("*.run"):
    if int(run_path.stem) == -2 or int(run_path.stem) == -3:
        continue
    h = hash(get_program(torch.load(run_path)))
    if int(run_path.stem) != h:
        new_path = (run_path.parent / str(h)).with_suffix(".run")
        run_path.rename(new_path)

for metadata_path in path.rglob("metadata.json"):
        metadata = {
            "SGD": hash(SGD),
            "Adam": hash(Adam),
            "AdamW": hash(AdamW),
            "RMSprop": hash(RMSprop),
            "Adagrad": hash(Adagrad),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

