import unittest

import torch

from nosbench.program import create_optimizer
from nosbench.optimizers import SGD, AdamW, Adam, RMSprop, Adagrad


class TestOptimum(unittest.TestCase):
    def train(self, optimizer_class, **kwargs):
        torch.manual_seed(123)
        model = torch.nn.Linear(1, 1)
        optim = optimizer_class(model.parameters(), **kwargs)
        for _ in range(1000):
            output = model(torch.tensor([1.0]))
            loss = torch.nn.functional.mse_loss(output, torch.tensor([1.0]))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss = loss.item()
        return loss

    def test_sgd(self):
        ground_truth = self.train(torch.optim.SGD, lr=0.01, momentum=0.9)
        loss = self.train(create_optimizer(SGD), lr=0.01)
        torch.testing.assert_close(ground_truth, loss)

    def test_adamw(self):
        ground_truth = self.train(
            torch.optim.AdamW, lr=0.01, betas=(0.9, 0.999), weight_decay=1e-2
        )
        loss = self.train(create_optimizer(AdamW), lr=0.01)
        torch.testing.assert_close(ground_truth, loss)

    def test_adam(self):
        ground_truth = self.train(torch.optim.Adam, lr=0.01, betas=(0.9, 0.999))
        loss = self.train(create_optimizer(Adam), lr=0.01)
        torch.testing.assert_close(ground_truth, loss)

    def test_rmsprop(self):
        ground_truth = self.train(torch.optim.RMSprop, lr=0.01, alpha=0.999)
        loss = self.train(create_optimizer(RMSprop), lr=0.01)
        torch.testing.assert_close(ground_truth, loss)

    def test_adagrad(self):
        ground_truth = self.train(torch.optim.Adagrad, lr=0.01)
        loss = self.train(create_optimizer(Adagrad), lr=0.01)
        torch.testing.assert_close(ground_truth, loss)
