import pathlib

import torch
from torch import nn
import numpy as np
import sklearn.datasets

from noslib.program import Program, Instruction, Pointer, READONLY_REGION, MAX_MEMORY
from noslib.function import Function
from noslib.function import interpolate, bias_correct
from noslib.optimizers import AdamW
from noslib.pipeline import MLPClassificationPipeline, ScikitLearnDataset


OPS = [
    Function(torch.div, 2),
    Function(torch.mul, 2),
    Function(torch.add, 2),
    Function(torch.sub, 2),
    Function(torch.square, 1),
    Function(torch.exp, 1),
    Function(torch.sign, 1),
    Function(torch.sqrt, 1),
    Function(interpolate, 3),
    Function(bias_correct, 3),
]

_op_dict = {str(op): op for op in OPS}


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
    def __init__(self, path="cache"):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._exists = set()
        for run in self.path.glob("*.run"):
            self._exists.add(int(run.stem))
        iris = sklearn.datasets.load_iris()
        dataset = ScikitLearnDataset(iris)
        self.pipeline = MLPClassificationPipeline(
            dataset=dataset, hidden_layers=[16], optimizer_kwargs={"lr": 0.0001}
        )

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

    @staticmethod
    def configspace(min_sloc=1, max_sloc=10, seed=None, default_program=AdamW):
        # TODO Needs update
        import ConfigSpace as cs
        import ConfigSpace.hyperparameters as csh

        configuration_space = cs.ConfigurationSpace(seed=seed)
        sloc_default = len(default_program) if len(default_program) > 0 else None
        sloc = csh.UniformIntegerHyperparameter(
            "sloc", lower=min_sloc, upper=max_sloc, default_value=sloc_default
        )
        configuration_space.add_hyperparameter(sloc)
        for idx, i in enumerate(range(min_sloc, max_sloc + 1)):
            op_default = None
            in1_default = None
            in2_default = None
            out_default = None
            if idx < len(default_program):
                instruction = default_program[idx]
                op_default = str(instruction.op)
                in1_default = instruction.in1
                if isinstance(instruction.op, BinaryFunction):
                    in2_default = instruction.in2
                out_default = instruction.out
            op = csh.CategoricalHyperparameter(
                f"op_{idx}", choices=[str(op) for op in OPS], default_value=op_default
            )
            lt = cs.GreaterThanCondition(op, sloc, i)
            eq = cs.EqualsCondition(op, sloc, i)
            configuration_space.add_hyperparameter(op)
            configuration_space.add_condition(cs.OrConjunction(lt, eq))
            in1 = csh.UniformIntegerHyperparameter(
                f"in1_{idx}", lower=0, upper=MAX_MEMORY, default_value=in1_default
            )
            in2 = csh.UniformIntegerHyperparameter(
                f"in2_{idx}", lower=0, upper=MAX_MEMORY, default_value=in2_default
            )
            out = csh.UniformIntegerHyperparameter(
                f"out_{idx}",
                lower=READONLY_REGION + 1,
                upper=MAX_MEMORY,
                default_value=out_default,
            )
            configuration_space.add_hyperparameters([in1, in2, out])
            lt = cs.GreaterThanCondition(in1, sloc, i)
            eq = cs.EqualsCondition(in1, sloc, i)
            configuration_space.add_condition(cs.OrConjunction(lt, eq))
            lt = cs.GreaterThanCondition(out, sloc, i)
            eq = cs.EqualsCondition(out, sloc, i)
            configuration_space.add_condition(cs.OrConjunction(lt, eq))
            binary_ops = [str(op) for op in OPS if isinstance(op, BinaryFunction)]
            lt = cs.GreaterThanCondition(in2, sloc, i)
            eq = cs.EqualsCondition(in2, sloc, i)
            cond = cs.OrConjunction(lt, eq)
            configuration_space.add_condition(
                cs.AndConjunction(cond, cs.InCondition(in2, op, binary_ops))
            )
        return configuration_space

    @staticmethod
    def configuration_to_program(config):
        program = Program()
        for i in range(0, config["sloc"]):
            op = _op_dict[config[f"op_{i}"]]
            in1 = config[f"in1_{i}"]
            in2 = config[f"in2_{i}"] if isinstance(op, BinaryFunction) else None
            out = config[f"out_{i}"]
            instruction = Instruction(op, Pointer(in1), Pointer(in2), Pointer(out))
            program.append(instruction)
        return program
