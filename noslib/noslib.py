import torch

from zero import create_optimizer


class NOSLib:
    def __init__(self):
        self.cache = {}

    def query(self, program):
        if program in self.cache:
            return self.cache[program]
        torch.manual_seed(123)
        model = torch.nn.Linear(2, 1)
        optimizer_class = create_optimizer(program)
        optim = optimizer_class(model.parameters(), lr=0.01)
        initial_loss = torch.nn.functional.mse_loss(model(torch.tensor([0.25, 0.25])), torch.tensor([10.0])).item()
        for _ in range(100):
            output = model(torch.tensor([0.25, -0.25]))
            loss = torch.nn.functional.mse_loss(output, torch.tensor([10.0]))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss = loss.item()
        self.cache[program] = loss
        return loss
