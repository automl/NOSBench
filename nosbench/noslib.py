import pathlib
import json
from itertools import zip_longest

from filelock import FileLock
import torch

from nosbench.optimizers import SGD, Adam, AdamW, RMSprop, Adagrad


class NOSLib:
    def __init__(self, pipeline, path="cache"):
        self.path = pathlib.Path(path)
        self.lock_path = ".lock" / self.path
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock_path.mkdir(parents=True, exist_ok=True)
        with FileLock((self.lock_path / "metadata.lock")):
            metadata_path = self.path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    assert all(
                        [
                            metadata["SGD"] == hash(SGD),
                            metadata["Adam"] == hash(Adam),
                            metadata["AdamW"] == hash(AdamW),
                            metadata["RMSprop"] == hash(RMSprop),
                            metadata["Adagrad"] == hash(Adagrad),
                        ]
                    )
            else:
                metadata = {
                    "SGD": hash(SGD),
                    "Adam": hash(Adam),
                    "AdamW": hash(AdamW),
                    "RMSprop": hash(RMSprop),
                    "Adagrad": hash(Adagrad),
                }
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)

        self._exists = set()
        for run in self.path.glob("*.run"):
            self._exists.add(int(run.stem))
        self.pipeline = pipeline

    def query(self, program, epoch, return_state=False):
        stem = hash(program)
        path = (self.path / str(stem)).with_suffix(".run")
        lock = FileLock((self.lock_path / str(stem)).with_suffix(".lock"))
        with lock.acquire():
            if stem in self._exists or path.exists():
                state_dict = torch.load(path)
            else:
                state_dict = {
                    "program": program,
                    "n_epochs": 0,
                    "stats": [],
                    "states": [],
                    }
            if epoch >= state_dict["n_epochs"]:
                stats, states = self.pipeline.evaluate(
                    state_dict["program"],
                    epoch - state_dict["n_epochs"] + 1,
                    state_dict["states"],
                )
                fillvalue = stats[0].empty_like()
                concat_stats = []
                for s1, s2 in zip_longest(state_dict["stats"], stats, fillvalue=fillvalue):
                    concat_stats.append(s1.concat(s2))

                state_dict["states"] = states
                state_dict["n_epochs"] = epoch + 1
                state_dict["stats"] = concat_stats

                torch.save(state_dict, path)
                self._exists.add(stem)
        loss = self.pipeline.evaluation_metric.evaluate(state_dict["stats"], epoch)
        if return_state:
            return loss, state_dict
        return loss
