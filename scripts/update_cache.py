import json
import pathlib
import argparse

import torch

from nosbench.optimizers import SGD, Adam, AdamW, RMSprop, Adagrad


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
args = parser.parse_args()

path = pathlib.Path(args.path)

for run_path in path.rglob("*.run"):
    if int(run_path.stem) == -2 or int(run_path.stem) == -3:
        continue
    state_dict = torch.load(run_path)
    h = hash(state_dict["program"])
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

