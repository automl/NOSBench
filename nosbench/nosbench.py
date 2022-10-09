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
            optimizer_kwargs={"lr": 0.0001},
            save_program=save_program,
            save_training_losses=save_training_losses,
            save_validation_losses=save_validation_losses,
            save_test_losses=save_test_losses,
            save_torch_state=save_torch_state,
            save_costs=save_costs,
        )
        super().__init__(pipeline=pipeline, path=path)

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
