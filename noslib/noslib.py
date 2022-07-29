import pathlib

import torch
import numpy as np

import zero


ops = [
    zero.BinaryFunction(torch.div),
    zero.BinaryFunction(torch.mul),
    zero.BinaryFunction(torch.add),
    zero.BinaryFunction(torch.sub),
    zero.UnaryFunction(torch.square),
    zero.UnaryFunction(torch.exp),
    zero.UnaryFunction(torch.sign),
    zero.UnaryFunction(torch.sqrt),
]

MAX_MEMORY = 20

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
        optimizer_class = zero.create_optimizer(program)
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

    @staticmethod
    def configspace(min_sloc=1, max_sloc=10, seed=None):
        import ConfigSpace as cs
        import ConfigSpace.hyperparameters as csh
        configuration_space = cs.ConfigurationSpace(seed)
        sloc = csh.UniformIntegerHyperparameter('sloc', lower=min_sloc, upper=max_sloc)
        configuration_space.add_hyperparameter(sloc)
        for i in range(min_sloc, max_sloc+1):
            op = csh.CategoricalHyperparameter(f'op_{i}', choices=ops)
            lt = cs.GreaterThanCondition(op, sloc, i)
            eq = cs.EqualsCondition(op, sloc, i)
            configuration_space.add_hyperparameter(op)
            configuration_space.add_condition(cs.OrConjunction(lt, eq))

            in1 = csh.UniformIntegerHyperparameter(f'in1_{i}', lower=0, upper=MAX_MEMORY)
            in2 = csh.UniformIntegerHyperparameter(f'in2_{i}', lower=0, upper=MAX_MEMORY)
            out = csh.UniformIntegerHyperparameter(f'out_{i}', lower=7, upper=MAX_MEMORY)
            configuration_space.add_hyperparameters([in1, in2, out])
            binary_ops = [op for op in ops if isinstance(op, zero.BinaryFunction)]
            configuration_space.add_condition(cs.InCondition(in2, op, binary_ops))
        return configuration_space

    @staticmethod
    def configuration_to_program(config):
        program = zero.Program()
        for i in range(1, config['sloc']+1):
            op = config[f'op_{i}']
            in1 = config[f'in1_{i}']
            in2 = config[f'in2_{i}'] if isinstance(op, zero.BinaryFunction) else None
            out = config[f'out_{i}']
            instruction = zero.Instruction(op, in1, in2, out)
            program.append(instruction)
        return program
