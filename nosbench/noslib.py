import pathlib
from filelock import FileLock

import torch


class NOSLib:
    def __init__(self, pipeline, path="cache"):
        self.path = pathlib.Path(path)
        self.lock_path = (".lock" / self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock_path.mkdir(parents=True, exist_ok=True)
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
                state_dict = self.pipeline.initial_state(program)
            # if epoch >= state_dict["n_epochs"]:
            state_dict = self.pipeline.query(state_dict, epoch + 1)
            torch.save(state_dict, path)
            self._exists.add(stem)

        loss = self.pipeline.performance(state_dict, epoch)
        if return_state:
            return loss, state_dict
        return loss
