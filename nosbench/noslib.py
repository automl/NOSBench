import pathlib

import torch


def initial_state(program):
    return {
        "program": program,
        "training_losses": [],
        "validation_losses": [],
        "test_losses": [],
        "torch_state": None,
        "costs": [],
        "n_epochs": 0,
    }


def _default_getitem(l, idx, default=None):
    if idx >= len(l):
        return default
    return l[idx]


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
            state_dict = initial_state(program)
        if epoch >= state_dict["n_epochs"]:
            state_dict = self.pipeline.query(state_dict, epoch + 1)
            torch.save(state_dict, (self.path / str(stem)).with_suffix(".run"))
            self._exists.add(stem)

        return {
            "minibatch_losses": _default_getitem(state_dict["training_losses"], epoch),
            "validation_loss": _default_getitem(state_dict["validation_losses"], epoch),
            "test_loss": _default_getitem(state_dict["test_losses"], epoch),
            "cost": _default_getitem(state_dict["costs"], epoch),
            "state": state_dict if return_state else {},
        }
