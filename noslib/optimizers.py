import torch

from zero import Program, Instruction, Pointer
from zero import UnaryFunction, BinaryFunction, DMABinaryFunction, DMAUnaryFunction
from zero import create_optimizer, interpolate1, interpolate2, interpolate_bc1, interpolate_bc2


AdamW = Program(
    [
        Instruction(UnaryFunction(torch.square), Pointer(1), None, Pointer(7)),
        Instruction(DMABinaryFunction(interpolate_bc1), Pointer(8), Pointer(1), Pointer(10)),
        Instruction(DMABinaryFunction(interpolate_bc2), Pointer(9), Pointer(7), Pointer(11)),
        Instruction(UnaryFunction(torch.sqrt), Pointer(11), None, Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(6), Pointer(7)),
        Instruction(BinaryFunction(torch.div), Pointer(10), Pointer(7), Pointer(7)),
        Instruction(BinaryFunction(torch.mul), Pointer(0), Pointer(5), Pointer(12)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(12), Pointer(7)),
    ]
)

Adam = Program(AdamW[:-2])

SGD = Program(
    [
        Instruction(BinaryFunction(torch.mul), Pointer(3), Pointer(7), Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(1), Pointer(7), Pointer(7)),
    ]
)

RMSprop = Program(
    [
        Instruction(UnaryFunction(torch.square), Pointer(1), None, Pointer(7)),
        Instruction(DMABinaryFunction(interpolate2), Pointer(8), Pointer(7), Pointer(8)),
        Instruction(UnaryFunction(torch.sqrt), Pointer(8), None, Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(6), Pointer(7)),
        Instruction(BinaryFunction(torch.div), Pointer(1), Pointer(7), Pointer(7)),
    ]
)

Adagrad = Program(
    [
        Instruction(UnaryFunction(torch.square), Pointer(1), None, Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(8), Pointer(8)),
        Instruction(UnaryFunction(torch.sqrt), Pointer(8), None, Pointer(7)),
        Instruction(BinaryFunction(torch.add), Pointer(7), Pointer(6), Pointer(7)),
        Instruction(BinaryFunction(torch.div), Pointer(1), Pointer(7), Pointer(7)),
    ]
)


if __name__ == "__main__":
    import unittest

    class TestOptimum(unittest.TestCase):
        def train(self, optimizer_class, **kwargs):
            torch.manual_seed(123)
            model = torch.nn.Linear(1, 1)
            optim = optimizer_class(model.parameters(), **kwargs)
            initial_loss = torch.nn.functional.mse_loss(model(torch.tensor([1.0])), torch.tensor([1.0])).item()
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
            ground_truth = self.train(torch.optim.AdamW, lr=0.01, betas=(0.9, 0.999), weight_decay=1e-2)
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

    unittest.main(exit=False)
