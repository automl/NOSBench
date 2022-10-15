from collections import defaultdict
from functools import partial

import torch
import sklearn.datasets

from nosbench.noslib import NOSLib
from nosbench.optimizers import AdamW
from nosbench.pipeline import OpenMLTabularPipeline
from nosbench.function import interpolate, bias_correct, clip, size
from nosbench.function import Function
from nosbench.program import Program, Instruction, Pointer, READONLY_REGION


MAX_MEMORY = 20
OPS = [
    Function(torch.div, 2),
    Function(torch.mul, 2),
    Function(torch.add, 2),
    Function(torch.sub, 2),
    Function(torch.square, 1),
    Function(torch.exp, 1),
    Function(torch.log, 1),
    Function(torch.sign, 1),
    Function(torch.sqrt, 1),
    Function(torch.abs, 1),
    Function(torch.norm, 1),
    Function(clip, 2),
    Function(torch.sin, 1),
    Function(torch.cos, 1),
    Function(torch.tan, 1),
    Function(torch.arcsin, 1),
    Function(torch.arccos, 1),
    Function(torch.arctan, 1),
    Function(torch.mean, 1),
    Function(torch.std, 1),
    Function(size, 1),
    Function(torch.minimum, 2),
    Function(torch.maximum, 2),
    Function(torch.heaviside, 2),
    Function(interpolate, 3),
    Function(bias_correct, 3),
]

_op_dict = {str(op): op for op in OPS}
condition_map = defaultdict(list)
for op in OPS:
    for i in range(1, op.n_args + 1):
        condition_map[i].append(str(op))


class NOSBench(NOSLib):
    def __init__(
        self,
        path="cache",
    ):
        pipeline = OpenMLTabularPipeline(
            data_id=31, n_fold=10, batch_size=50, hidden_layers=[16]
        )
        super().__init__(pipeline=pipeline, path=path)

    @staticmethod
    def configspace(min_sloc=1, max_sloc=len(AdamW), max_memory=MAX_MEMORY, seed=None):
        import ConfigSpace as cs
        import ConfigSpace.hyperparameters as csh

        configuration_space = cs.ConfigurationSpace(seed=seed)
        sloc = csh.UniformIntegerHyperparameter("sloc", lower=min_sloc, upper=max_sloc)
        configuration_space.add_hyperparameter(sloc)
        for idx, i in enumerate(range(min_sloc, max_sloc + 1)):
            op = csh.CategoricalHyperparameter(
                f"op_{idx}", choices=[str(op) for op in OPS]
            )
            lt = cs.GreaterThanCondition(op, sloc, i)
            eq = cs.EqualsCondition(op, sloc, i)
            configuration_space.add_hyperparameter(op)
            configuration_space.add_condition(cs.OrConjunction(lt, eq))
            for input_idx in condition_map:
                inp = csh.UniformIntegerHyperparameter(
                    f"in{input_idx}_{idx}", lower=0, upper=max_memory
                )
                configuration_space.add_hyperparameter(inp)
                in_cond = cs.InCondition(inp, op, condition_map[input_idx])
                lt = cs.GreaterThanCondition(inp, sloc, i)
                eq = cs.EqualsCondition(inp, sloc, i)
                sloc_cond = cs.OrConjunction(lt, eq)
                configuration_space.add_condition(cs.AndConjunction(sloc_cond, in_cond))

            out = csh.UniformIntegerHyperparameter(
                f"out_{idx}",
                lower=READONLY_REGION + 1,
                upper=max_memory,
            )
            configuration_space.add_hyperparameter(out)
            lt = cs.GreaterThanCondition(out, sloc, i)
            eq = cs.EqualsCondition(out, sloc, i)
            configuration_space.add_condition(cs.OrConjunction(lt, eq))
        return configuration_space

    @staticmethod
    def configuration_to_program(config):
        program = Program()
        for i in range(0, config["sloc"]):
            op = _op_dict[config[f"op_{i}"]]
            inputs = [config[f"in{idx}_{i}"] for idx in range(1, op.n_args + 1)]
            output = config[f"out_{i}"]
            instruction = Instruction(op, inputs, output)
            program.append(instruction)
        return program
