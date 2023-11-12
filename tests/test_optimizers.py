import unittest

import torch

from nosbench.program import create_optimizer
from nosbench.optimizers import SGD, AdamW, Adam, RMSprop, Adagrad, Adadelta


class TestOptimum(unittest.TestCase):
    def train(self, optimizer_class, **kwargs):
        torch.manual_seed(123)
        param = torch.nn.Parameter(torch.tensor([1.0]))
        optim = optimizer_class([param], **kwargs)
        for _ in range(3):
            loss = torch.nn.functional.mse_loss(param, torch.tensor([0.0]))
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss = loss.item()
        return loss

    def test_sgd(self):
        ground_truth = self.train(torch.optim.SGD, lr=1e-2)
        loss = self.train(create_optimizer(SGD))
        torch.testing.assert_close(ground_truth, loss)

    def test_adamw(self):
        ground_truth = self.train(torch.optim.AdamW)
        loss = self.train(create_optimizer(AdamW))
        torch.testing.assert_close(ground_truth, loss)

    def test_adam(self):
        ground_truth = self.train(torch.optim.Adam)
        loss = self.train(create_optimizer(Adam))
        torch.testing.assert_close(ground_truth, loss)

    def test_rmsprop(self):
        ground_truth = self.train(torch.optim.RMSprop)
        loss = self.train(create_optimizer(RMSprop))
        torch.testing.assert_close(ground_truth, loss)

    def test_adagrad(self):
        ground_truth = self.train(torch.optim.Adagrad, lr=1e-2, eps=1e-8)
        loss = self.train(create_optimizer(Adagrad))
        torch.testing.assert_close(ground_truth, loss)

    def test_adadelta(self):
        ground_truth = self.train(torch.optim.Adadelta, lr=1.0, eps=1e-6)
        loss = self.train(create_optimizer(Adadelta))
        torch.testing.assert_close(ground_truth, loss)
