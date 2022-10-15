import pathlib

import torch


class NOSLib:
    def __init__(self, pipeline, path="cache"):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._exists = set()
        for run in self.path.glob("*.run"):
            self._exists.add(int(run.stem))
        self.pipeline = pipeline

    def query(self, program, epoch, return_state=False):
        stem = hash(program)
        if stem in self._exists:
            state_dict = torch.load((self.path / str(stem)).with_suffix(".run"))
        else:
            state_dict = self.pipeline.initial_state(program)
        # if epoch >= state_dict["n_epochs"]:
        state_dict = self.pipeline.query(state_dict, epoch + 1)
        torch.save(state_dict, (self.path / str(stem)).with_suffix(".run"))
        self._exists.add(stem)

        loss = self.pipeline.performance(state_dict, epoch)
        if return_state:
            return loss, state_dict
        return loss
