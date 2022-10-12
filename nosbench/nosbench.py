from collections import defaultdict

import torch
import sklearn.datasets
from torch.utils.data import random_split

from nosbench.noslib import NOSLib
from nosbench.optimizers import AdamW
from nosbench.pipeline import MLPClassificationPipeline, ScikitLearnDataset
from nosbench.function import interpolate, bias_correct
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
    Function(torch.sign, 1),
    Function(torch.sqrt, 1),
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
        save_program: bool = True,
        save_training_losses: bool = True,
        save_validation_losses: bool = True,
        save_test_losses: bool = True,
        save_torch_state: bool = True,
        save_costs: bool = True,
    ):
        iris = sklearn.datasets.load_iris()
        dataset = ScikitLearnDataset(iris)
        split = [int(s * len(dataset)) for s in [0.8, 0.1, 0.1]]
        generator = torch.Generator().manual_seed(42)
        train, val, test = random_split(dataset, split, generator=generator)
        input_size = len(dataset.feature_names)
        output_size = len(dataset.target_names)
        pipeline = MLPClassificationPipeline(
            train=train,
            val=val,
            test=test,
            batch_size=-1,
            input_size=input_size,
            hidden_layers=[16],
            output_size=output_size,
            save_program=save_program,
            save_training_losses=save_training_losses,
            save_validation_losses=save_validation_losses,
            save_test_losses=save_test_losses,
            save_torch_state=save_torch_state,
            save_costs=save_costs,
        )
        super().__init__(pipeline=pipeline, path=path)

    @staticmethod
    def configspace(min_sloc=1, max_sloc=10, max_memory=MAX_MEMORY, seed=None):
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
            inputs = [config[f"in{idx}_{i}"] for idx in range(1, op.n_args+1)]
            output = config[f"out_{i}"]
            instruction = Instruction(op, inputs, output)
            program.append(instruction)
        return program
