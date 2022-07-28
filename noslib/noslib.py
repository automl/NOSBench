import pathlib

import torch
import numpy as np

from zero import create_optimizer


class NOSLib:
    def __init__(self, path="cache"):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._exists = set()
        for run in self.path.glob("*.run"):
            self._exists.add(int(run.stem))

    @staticmethod
    def _schaffer(data):
        x = data[0]
        y = data[1]
        num = torch.sin(x**2 - y**2) ** 2 - 0.5
        denom = (1 + 0.001 * (x**2 + y**2)) ** 2
        return 0.5 + num / denom

    def query(self, program):
        stem = hash(program)
        if stem in self._exists:
            return np.load((self.path / str(stem)).with_suffix(".run"))
        torch.manual_seed(123)
        params = torch.nn.Parameter(torch.rand(2) * 200 - 100)
        optimizer_class = create_optimizer(program)
        optim = optimizer_class([params], lr=0.001)
        for _ in range(100):
            output = self._schaffer(params)
            optim.zero_grad()
            output.backward()
            optim.step()
        with open((self.path / str(stem)).with_suffix(".run"), "wb") as f:
            np.save(f, output.item())
            self._exists.add(stem)
        return output.item()
