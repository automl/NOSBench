import json
import pathlib
import argparse
import pickle

import torch

from nosbench.optimizers import SGD, Adam, AdamW, RMSprop, Adagrad


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

path = pathlib.Path(args.path)

for run_path in path.rglob("*.run"):
    if int(run_path.stem) == -2 or int(run_path.stem) == -3:
        continue
    with open(run_path, "rb") as f:
        state_dict = pickle.load(f)
    h = hash(state_dict.program)
    if int(run_path.stem) != h:
        new_path = (run_path.parent / str(h)).with_suffix(".run")
        run_path.rename(new_path)
        print(f"{run_path.stem} -> {new_path.stem}")

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
