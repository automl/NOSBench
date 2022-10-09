import pathlib

import torch


def initial_state(program):
    return {
        "program": program,
        "training_losses": [],
        "validation_losses": [],
        "test_losses": [],
        "torch_state": None,
        "cost": 0,
    }


class NOSLib:
    def __init__(self, pipeline, path="cache"):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._exists = set()
        for run in self.path.glob("*.run"):
            self._exists.add(int(run.stem))
        self.pipeline = pipeline

    def query(self, program, epochs):
        stem = hash(program)
        if stem in self._exists:
            state_dict = torch.load((self.path / str(stem)).with_suffix(".run"))
        else:
            state_dict = initial_state(program)
        if epochs > len(state_dict["training_losses"]):
            state_dict = self.pipeline.query(state_dict, epochs)
            torch.save(state_dict, (self.path / str(stem)).with_suffix(".run"))
            self._exists.add(stem)
        return state_dict
