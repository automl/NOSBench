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


class Trainer(ABC):
    @abstractmethod
    def train(self, model, optimizer, train_loader, val_loader, epochs):
        """Train and validate given model on train/val for epochs"""


class ClassificationTrainer(Trainer):
    target_weights: Optional[float] = None

    @deterministic(seed=42)
    def train(self, model, optimizer, train_loader, val_loader, epochs):
        device = Device.get()
        training_losses = []
        validation_losses = []
        accuracies = []
        training_costs = []
        training_cost = 0.0
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


class EvaluationMetric(ABC):
    @abstractmethod
    def evaluations(self, dataset):
        """Yield training and validation data loader generator"""

    @abstractmethod
    def evaluate(self, results, epoch):
        """Calculate final evaluation result from results"""


@dataclass
class TrainValidationSplit(EvaluationMetric):
    training_percentage: float = 0.8
    batch_size: float = -1

    @deterministic(seed=42)
    def evaluations(self, dataset):
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
    def evaluations(self, dataset):
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
    def evaluate(self, program, epochs, states=[]):
        device = Device.get()
        results = []
        new_states = []
        for (train, val), state in zip_longest(
            self.evaluation_metric.evaluations(self.dataset), states
        ):
            model = self.model_factory.create_model().to(device)
            optimizer_class = program.optimizer()
            optimizer = optimizer_class(model.parameters())
            if state is not None:
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
            result = self.trainer.train(model, optimizer, train, val, epochs)
            results.append(result)
            new_states.append(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
            )
        return results, new_states
