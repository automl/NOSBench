import pathlib
import json
from itertools import zip_longest
import pickle
from dataclasses import dataclass

from filelock import FileLock
import torch

from nosbench.optimizers import SGD, Adam, AdamW, RMSprop, Adagrad
from nosbench.pipeline import Result
from nosbench.program import Program
from nosbench.device import Device


@dataclass
class Run:
    program: Program
    epochs: int
    results: list[Result]


class NOSLib:
    def __init__(self, pipeline, path="cache", device="cpu"):
        self.device = device
        self.path = pathlib.Path(path)
        self.lock_path = ".lock" / self.path
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock_path.mkdir(parents=True, exist_ok=True)
        with FileLock((self.lock_path / "metadata.lock")):
            metadata_path = self.path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if not all(
                        [
                            metadata["SGD"] == hash(SGD),
                            metadata["Adam"] == hash(Adam),
                            metadata["AdamW"] == hash(AdamW),
                            metadata["RMSprop"] == hash(RMSprop),
                            metadata["Adagrad"] == hash(Adagrad),
                        ]
                    ):
                        raise UserWarning(
                            f"Metadata in '{path}' does not match with "
                            "calculated hashes. This could be due to version "
                            "difference between the machine running this and "
                            "the machine generated the data. Update hashes by "
                            f"running 'python -m scripts.update_cache --path {path}'"
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

    def query(self, program, epoch, return_run=False):
        Device.set(self.device)
        stem = hash(program)
        path = (self.path / str(stem)).with_suffix(".run")
        lock = FileLock((self.lock_path / str(stem)).with_suffix(".lock"))
        with lock.acquire():
            if stem in self._exists or path.exists():
                with path.open("rb") as f:
                    run = pickle.load(f)
            else:
                run = Run(
                    program=program,
                    epochs=0,
                    results=[],
                )
            if epoch >= run.epochs:
                state_path = (self.path / str(stem)).with_suffix(".states")
                states = []
                if run.epochs > 0:
                    states = torch.load(state_path)

                results, states = self.pipeline.evaluate(
                    run.program,
                    epoch - run.epochs + 1,
                    states,
                )
                fillvalue = results[0].empty_like()
                concat_results = []
                for s1, s2 in zip_longest(run.results, results, fillvalue=fillvalue):
                    concat_results.append(s1.concat(s2))

                run.epochs = epoch + 1
                run.results = concat_results

                with path.open("wb") as f:
                    pickle.dump(run, f)
                torch.save(states, state_path)
                self._exists.add(stem)
        loss = self.pipeline.evaluation_metric.evaluate(run.results, epoch)
        if return_run:
            return loss, run
        return loss
