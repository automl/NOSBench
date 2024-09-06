import os
from collections import defaultdict
from itertools import cycle
import hashlib

import torch
import sklearn.datasets
from sklearn.preprocessing import StandardScaler

from nosbench.noslib import NOSLib
from nosbench.optimizers import AdamW
from nosbench.pipeline import (
    Pipeline,
    ClassificationTrainer,
    PFNTrainer,
    ScikitLearnDataset,
    RidgeRegressionDataset,
)
from nosbench.utils import deterministic
from nosbench.pipeline import ToyMLPModelFactory, MLPModelFactory, PFNModelFactory
from nosbench.pipeline import TrainValidationSplit, CrossValidation, PFNEvaluation
from nosbench.function import interpolate, bias_correct, clip, size
from nosbench.function import Function
from nosbench.program import Program, Instruction, READONLY_REGION
from nosbench.pfns.bar_distribution import FullSupportBarDistribution, get_bucket_limits
from nosbench.pfns.utils import sample_from_prior


class BaseBenchmark(NOSLib):
    ops = [
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
    __op_map = {str(op): op for op in ops}
    __condition_map = defaultdict(list)
    for op in ops:
        for i in range(1, op.n_args + 1):
            __condition_map[i].append(str(op))

    @staticmethod
    def configspace(min_sloc=1, max_sloc=len(AdamW), max_memory=19, seed=None):
        import ConfigSpace as cs
        import ConfigSpace.hyperparameters as csh

        configuration_space = cs.ConfigurationSpace(seed=seed)
        sloc = csh.UniformIntegerHyperparameter("sloc", lower=min_sloc, upper=max_sloc)
        configuration_space.add_hyperparameter(sloc)
        for idx, i in enumerate(range(min_sloc, max_sloc + 1)):
            op = csh.CategoricalHyperparameter(
                f"op_{idx}", choices=[str(op) for op in BaseBenchmark.ops]
            )
            lt = cs.GreaterThanCondition(op, sloc, i)
            eq = cs.EqualsCondition(op, sloc, i)
            configuration_space.add_hyperparameter(op)
            configuration_space.add_condition(cs.OrConjunction(lt, eq))
            for input_idx in BaseBenchmark.__condition_map:
                inp = csh.UniformIntegerHyperparameter(
                    f"in{input_idx}_{idx}", lower=0, upper=max_memory
                )
                configuration_space.add_hyperparameter(inp)
                in_cond = cs.InCondition(
                    inp, op, BaseBenchmark.__condition_map[input_idx]
                )
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
            op = BaseBenchmark.__op_map[config[f"op_{i}"]]
            inputs = [config[f"in{idx}_{i}"] for idx in range(1, op.n_args + 1)]
            output = config[f"out_{i}"]
            instruction = Instruction(op, inputs, output)
            program.append(instruction)
        return program

class ToyBenchmark(BaseBenchmark):
    def __init__(self, path="cache", device="cpu"):
        iris = sklearn.datasets.load_iris()
        dataset = ScikitLearnDataset(iris, StandardScaler())
        trainer = ClassificationTrainer()
        input_size = len(dataset.feature_names)
        model_factory = ToyMLPModelFactory(input_size, [16], dataset.n_classes)
        evaluation_metric = TrainValidationSplit(training_percentage=0.8, batch_size=-1)
        pipeline = Pipeline(dataset, trainer, model_factory, evaluation_metric)
        path = os.path.join(path, "toy")
        super().__init__(pipeline=pipeline, path=path, device=device)

class NOSMLPBench(BaseBenchmark):
    def __init__(
        self,
        data_id=31,
        n_splits=10,
        batch_size=60,
        backbone=[290, 272, 254, 236, 218, 200],
        head=[128],
        path="cache",
        data_home=None,
        device="cpu",
    ):
        self.data_id = data_id
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.backbone = backbone
        self.head = head
        dataset = sklearn.datasets.fetch_openml(
            data_id=data_id, data_home=data_home, as_frame=False, parser="liac-arff"
        )
        dataset = ScikitLearnDataset(dataset, StandardScaler())
        trainer = ClassificationTrainer()
        model_factory = MLPModelFactory(
            len(dataset.feature_names), dataset.n_classes, self.backbone, self.head
        )
        evaluation_metric = CrossValidation(n_splits=n_splits, batch_size=batch_size)
        pipeline = Pipeline(dataset, trainer, model_factory, evaluation_metric)
        args_string = f"{data_id}_{n_splits}_{batch_size}_{backbone}_{head}"
        args_hash = hashlib.md5(args_string.encode('utf-8')).hexdigest()
        path = os.path.join(path, args_hash)
        super().__init__(pipeline=pipeline, path=path, device=device)

    def __str__(self):
        return (
            f"class: {self.__class__.__name__}\n"
            f"data_id: {self.data_id}\n"
            f"n_splits: {self.n_splits}\n"
            f"batch_size: {self.batch_size}\n"
            f"backbone: {self.backbone}\n"
            f"head: {self.head}\n"
        )

class NOSBench(BaseBenchmark):
    @deterministic(seed=42)
    def __init__(self, path="cache", device="cpu"):
        num_features = 1
        seq_len = 20
        steps_per_epoch = 100
        batch_size = 8
        dataset_size = int(steps_per_epoch * batch_size)

        _, y = sample_from_prior(100000, seq_len, num_features)
        limits = get_bucket_limits(num_outputs=100, ys=y.transpose(0, 1).to(device))
        criterion = FullSupportBarDistribution(limits)

        model_factory = PFNModelFactory(
            ninp=256,
            nout=criterion.num_bars,
            nhead=4,
            nhid=512,
            nlayers=4,
            num_features=num_features,
        )
        dataset = RidgeRegressionDataset(
            dataset_size=dataset_size, seq_len=seq_len, num_features=num_features
        )
        trainer = PFNTrainer(criterion)
        evaluation_metric = PFNEvaluation(batch_size=batch_size)
        pipeline = Pipeline(dataset, trainer, model_factory, evaluation_metric)
        path = os.path.join(path, "pfn")
        super().__init__(pipeline=pipeline, path=path, device=device)
