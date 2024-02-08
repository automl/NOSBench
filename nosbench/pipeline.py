import time
from abc import abstractmethod, ABC
from itertools import count, zip_longest
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import sklearn.model_selection
import sklearn.datasets

from nosbench.utils import deterministic
from nosbench.device import Device
from nosbench.pfns.model import PFNModel
from nosbench.pfns.utils import sample_from_prior, torch_nanmean


class ScikitLearnDataset(Dataset):
    def __init__(self, dataset, scaler=None):
        data = dataset.data
        if scaler is not None:
            scaler.fit(data)
            data = scaler.transform(data)
        self.data = torch.from_numpy(data).float()
        target = []
        counter = count(start=0, step=1)
        self.target_map = defaultdict(lambda: next(counter))
        for t in dataset.target:
            target.append(self.target_map[t])
        self.target = torch.tensor(target, dtype=torch.long)
        self.feature_names = dataset.feature_names
        self.target_names = dataset.target_names
        self.n_classes = next(counter)

    def __getitem__(self, i):
        return self.data[i], self.target[i]

    def __len__(self):
        return len(self.target)


class RidgeRegressionDataset(Dataset):
    def __init__(self, dataset_size, seq_len, num_features):
        self.dataset_size = dataset_size
        self.seq_len = seq_len
        self.num_features = num_features

    def loader(self, start_epoch, batch_size):
        dataset_size = self.dataset_size
        seq_len = self.seq_len
        num_features = self.num_features

        class RidgeRegressionDataLoader(DataLoader):
            def __init__(self):
                self.epoch = 0

            def __iter__(self):
                size = dataset_size // batch_size
                for i in range(size):
                    f = deterministic(size * (start_epoch + self.epoch) + i)(
                        sample_from_prior
                    )
                    yield f(batch_size, seq_len, num_features)

            def step(self):
                self.epoch += 1

            @staticmethod
            def __len__():
                return self.dataset_size // self.batch_size

        return RidgeRegressionDataLoader()


class Result(ABC):
    @abstractmethod
    def concat(self, other):
        """Concatenate given results"""

    def empty_like(self):
        return self.__class__()


@dataclass(frozen=True)
class ClassificationResult(Result):
    training_losses: list = field(default_factory=list)
    validation_losses: list = field(default_factory=list)
    accuracies: list = field(default_factory=list)
    training_costs: list = field(default_factory=list)

    def concat(self, other):
        other_costs = other.training_costs
        if len(self.training_costs) > 0:
            other_costs = [a + self.training_costs[-1] for a in other.training_costs]

        return ClassificationResult(
            self.training_losses + other.training_losses,
            self.validation_losses + other.validation_losses,
            self.accuracies + other.accuracies,
            self.training_costs + other_costs,
        )


@dataclass(frozen=True)
class PFNResult(Result):
    training_losses: list = field(default_factory=list)

    def concat(self, other):
        return PFNResult(
            self.training_losses + other.training_losses,
        )


class Trainer(ABC):
    @abstractmethod
    def train(self, model, optimizer, loaders, epochs):
        """Train and validate given model on train/val for epochs"""


@dataclass
class ClassificationTrainer(Trainer):
    target_weights: Optional[float] = None

    @deterministic(seed=42)
    def train(self, model, optimizer, loaders, epochs):
        device = Device.get()
        training_losses = []
        validation_losses = []
        accuracies = []
        training_costs = []
        training_cost = 0.0
        train_loader, val_loader = loaders
        for epoch in range(epochs):
            minibatch_losses = []
            prev_time = time.monotonic()
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                loss = nn.functional.nll_loss(
                    model(data), target, weight=self.target_weights
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                minibatch_losses.append(loss.item())
            training_losses.append(minibatch_losses)
            training_cost += time.monotonic() - prev_time
            training_costs.append(training_cost)

            validation_loss = 0.0
            accuracy = 0.0
            size = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    loss = nn.functional.nll_loss(output, target, reduction="sum")
                    validation_loss += loss.item()
                    accuracy += (output.max(dim=1)[1] == target).sum().item()
                    size += len(data)
                accuracies.append(accuracy / size)
                validation_losses.append(validation_loss / size)

        return ClassificationResult(
            training_losses=training_losses,
            validation_losses=validation_losses,
            accuracies=accuracies,
            # TODO: Remove training_cost
            training_costs=training_costs,
        )


@dataclass
class PFNTrainer(Trainer):
    criterion: callable

    @deterministic(seed=42)
    def train(self, model, optimizer, loader, epochs):
        device = Device.get()
        training_losses = []
        for epoch in range(epochs):
            minibatch_losses = []
            prev_time = time.monotonic()
            total_loss = 0.0
            for x, y in loader:
                x, y = x.squeeze(0), y.squeeze(0)
                seq_len = x.shape[1]
                single_eval_pos = torch.randint(seq_len, []).numpy()
                y = y.transpose(0, 1).to(device)
                logits = model(
                    (x.transpose(0, 1).to(device), y), single_eval_pos=single_eval_pos
                )["standard"]
                targets = y[single_eval_pos:]
                losses = self.criterion(logits, targets)
                losses = losses.view(-1, logits.shape[1])
                loss, nan_share = torch_nanmean(losses.mean(0), return_nanshare=True)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                minibatch_losses.append(loss.item())
            loader.step()
            training_losses.append(minibatch_losses)

        return PFNResult(
            training_losses=training_losses,
        )


class EvaluationMetric(ABC):
    @abstractmethod
    def evaluations(self, dataset, start_epoch):
        """Yield training and validation data loader generator"""

    @abstractmethod
    def evaluate(self, results, epoch):
        """Calculate final evaluation result from results"""


@dataclass
class PFNEvaluation(EvaluationMetric):
    batch_size: float = -1

    @deterministic(seed=42)
    def evaluations(self, dataset, start_epoch):
        yield dataset.loader(start_epoch, self.batch_size)

    def evaluate(self, results, epoch):
        return np.mean(results[0].training_losses[epoch])


@dataclass
class TrainValidationSplit(EvaluationMetric):
    training_percentage: float = 0.8
    batch_size: float = -1

    @deterministic(seed=42)
    def evaluations(self, dataset, start_epoch):
        split = [self.training_percentage, 1 - self.training_percentage]
        split = [int(np.ceil(s * len(dataset))) for s in split]
        train, val = random_split(dataset, split)
        train_loader = DataLoader(
            train,
            batch_size=self.batch_size if self.batch_size > 0 else len(train),
            drop_last=True,
        )
        val_loader = DataLoader(val, batch_size=len(val))
        yield train_loader, val_loader

    def evaluate(self, results, epoch):
        return results[0].validation_losses[epoch]


@dataclass
class CrossValidation(EvaluationMetric):
    n_splits: int = 10
    batch_size: float = -1

    @deterministic(seed=42)
    def evaluations(self, dataset, start_epoch):
        splits = sklearn.model_selection.KFold(
            n_splits=self.n_splits, shuffle=True, random_state=np.random.RandomState(42)
        ).split(dataset)

        for train_split, val_split in splits:
            train_sampler = torch.utils.data.SubsetRandomSampler(train_split)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_split)
            train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size if self.batch_size > 0 else len(train_split),
                sampler=train_sampler,
                drop_last=True,
            )
            val_loader = DataLoader(
                dataset, batch_size=len(val_split), sampler=val_sampler
            )
            yield train_loader, val_loader

    def evaluate(self, results, epoch):
        validation_loss = 0.0
        for result in results:
            validation_loss += result.validation_losses[epoch]
        return validation_loss / len(results)


class ModelFactory(ABC):
    @abstractmethod
    def create_model(self):
        """Return initialized model"""


@dataclass
class ToyMLPModelFactory(ModelFactory):
    n_features: int
    hidden_layers: list
    n_classes: int

    @deterministic(seed=42)
    def create_model(self):
        module_list = []
        prev_layer = self.n_features
        for layer in self.hidden_layers:
            module_list.append(nn.Linear(prev_layer, layer))
            module_list.append(nn.ReLU())
            prev_layer = layer
        module_list.append(nn.Linear(prev_layer, self.n_classes))
        module_list.append(nn.LogSoftmax(-1))
        return nn.Sequential(*module_list)


@dataclass
class PFNModelFactory(ModelFactory):
    ninp: int
    nout: int
    nhead: int
    nhid: int
    nlayers: int
    num_features: int
    dropout: int = 0.0

    @deterministic(seed=42)
    def create_model(self):
        return PFNModel(
            ninp=self.ninp,
            nout=self.nout,
            nhead=self.nhead,
            nhid=self.nhid,
            nlayers=self.nlayers,
            num_features=self.num_features,
            dropout=self.dropout,
        )


class _Linear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=np.sqrt(2))
        if self.bias is not None:
            nn.init.zeros_(self.bias)


@dataclass
class MLPModelFactory(ModelFactory):
    n_features: int
    n_classes: int
    backbone: list
    head: list

    @staticmethod
    def _backbone(input_size, layer_sizes):
        prev = input_size
        layers = []
        for size in layer_sizes[:-1]:
            layers.append(_Linear(prev, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            prev = size
        layers.append(_Linear(prev, layer_sizes[-1]))
        return nn.Sequential(*layers)

    @staticmethod
    def _head(layer_sizes, output_size):
        layers = []
        prev = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(_Linear(prev, size))
            layers.append(nn.ReLU())
            prev = size
        layers.append(_Linear(prev, output_size))
        return nn.Sequential(*layers)

    @deterministic(seed=42)
    def create_model(self):
        layers = []
        backbone = self._backbone(self.n_features, self.backbone)
        layers.append(backbone)
        if len(self.head):
            layers.append(_Linear(self.backbone[-1], self.head[0]))
            layers.append(nn.ReLU())
            head = self._head(self.head, self.n_classes)
            layers.append(head)
        else:
            layers.append(_Linear(self.backbone[-1], self.n_classes))
        layers.append(nn.LogSoftmax(-1))
        return nn.Sequential(*layers)


@dataclass
class Pipeline:
    dataset: Dataset
    trainer: Trainer
    model_factory: ModelFactory
    evaluation_metric: EvaluationMetric

    @deterministic(seed=42)
    def evaluate(self, program, start_epoch, end_epoch, states=[], **kwargs):
        device = Device.get()
        results = []
        new_states = []
        for (loaders), state in zip_longest(
            self.evaluation_metric.evaluations(self.dataset, start_epoch), states
        ):
            model = self.model_factory.create_model().to(device)
            optimizer_class = program.optimizer()
            lr = kwargs.get("lr", 1.0)
            optimizer = optimizer_class(model.parameters(), lr=lr)
            if state is not None:
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
            result = self.trainer.train(
                model, optimizer, loaders, end_epoch - start_epoch
            )
            results.append(result)
            new_states.append(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
            )
        return results, new_states
