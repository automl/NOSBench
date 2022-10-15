import time
import copy
from dataclasses import dataclass
from abc import abstractmethod
from functools import lru_cache
from itertools import count
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import sklearn.model_selection
import sklearn.datasets


class ScikitLearnDataset(Dataset):
    def __init__(self, dataset):
        self.data = torch.from_numpy(dataset.data).float()
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


class BasePipeline:
    def __init__(
        self,
        dataset,
        batch_size,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.loader_generator = self.get_loader_generator()

    def _query(self, state_dict, n_epochs):
        torch.manual_seed(42)
        model = self.create_model()
        program = state_dict["program"]
        optimizer_class = program.optimizer()
        optimizer = optimizer_class(model.parameters())
        if state_dict["torch_state"] is not None:
            model.load_state_dict(state_dict["torch_state"]["model"])
            optimizer.load_state_dict(state_dict["torch_state"]["optimizer"])
        training_losses = state_dict["training_losses"]
        validation_losses = state_dict["validation_losses"]
        costs = state_dict["costs"]
        total_cost = costs[-1] if state_dict["n_epochs"] > 0 else 0
        train_loader, val_loader = next(self.loader_generator)
        for epoch in range(state_dict["n_epochs"], n_epochs):
            prev_time = time.time()
            minibatch_losses = []
            for data, target in train_loader:
                loss = self.eval(model, data, target, optimizer, train=True)
                minibatch_losses.append(loss.item())
            training_losses.append(minibatch_losses)
            total_cost += time.time() - prev_time
            costs.append(total_cost)

            with torch.no_grad():
                for data, target in val_loader:
                    loss = self.eval(model, data, target, optimizer, train=False)
                    validation_losses.append(loss.item())

        torch_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        state_dict["n_epochs"] = n_epochs
        return {
            "program": program,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "torch_state": torch_state,
            "costs": costs,
            "n_epochs": n_epochs,
        }

    @staticmethod
    def initial_state(program):
        return {
            "program": program,
            "training_losses": [],
            "validation_losses": [],
            "test_losses": [],
            "torch_state": None,
            "costs": [],
            "n_epochs": 0,
        }

    def query(self, state_dict, n_epochs):
        return self._query(state_dict, n_epochs)

    @abstractmethod
    def get_loader_generator(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def performance(state_dict, epoch):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def eval(model, data, target, optimizer, train=True):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_model():
        raise NotImplementedError


class TrainValidationSplitMixin:
    def __init__(self, split, **kwargs):
        self.split = split
        super().__init__(**kwargs)

    def get_loader_generator(self):
        generator = torch.Generator().manual_seed(42)
        train, val = random_split(self.dataset, self.split, generator=generator)
        train_loader = DataLoader(
            train, batch_size=self.batch_size if self.batch_size > 0 else len(train)
        )
        val_loader = DataLoader(val, batch_size=len(val))
        while True:
            yield train_loader, val_loader

    @staticmethod
    def performance(state_dict, epoch):
        return state_dict["validation_losses"][epoch]


class KFoldMixin:
    def __init__(self, n_fold, **kwargs):
        self.n_fold = n_fold
        super().__init__(**kwargs)

    def get_loader_generator(self):
        splits = sklearn.model_selection.KFold(
            n_splits=self.n_fold, shuffle=True, random_state=np.random.RandomState(42)
        ).split(self.dataset)
        loaders = []
        for train_split, val_split in splits:
            train_sampler = torch.utils.data.SubsetRandomSampler(train_split)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_split)
            train_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size if self.batch_size > 0 else len(train_split),
                sampler=train_sampler,
            )
            val_loader = DataLoader(
                self.dataset, batch_size=len(val_split), sampler=val_sampler
            )
            loaders.append((train_loader, val_loader))
        while True:
            for fold in range(self.n_fold):
                yield loaders[fold]

    def performance(self, state_dict, epoch):
        total_validation_loss = 0
        for fold in range(self.n_fold):
            total_validation_loss += state_dict[fold]["validation_losses"][epoch]
        return total_validation_loss

    def query(self, state_dict, n_epochs):
        for fold in range(self.n_fold):
            self._query(state_dict[fold], n_epochs)
        return state_dict

    def initial_state(self, program):
        states = []
        for fold in range(self.n_fold):
            states.append(super().initial_state(program))
        return states


class MLPClassificationPipeline(BasePipeline):
    def __init__(self, dataset, hidden_layers, **kwargs):
        input_size = len(dataset.feature_names)
        self.layers = [input_size, *hidden_layers, dataset.n_classes]
        super().__init__(dataset=dataset, **kwargs)

    def create_model(self):
        module_list = []
        prev_layer = self.layers[0]
        for layer in self.layers[1:-1]:
            module_list.append(nn.Linear(prev_layer, layer))
            module_list.append(nn.ReLU())
            prev_layer = layer
        module_list.append(nn.Linear(prev_layer, self.layers[-1]))
        module_list.append(nn.LogSoftmax(-1))
        return nn.Sequential(*module_list)

    @staticmethod
    def eval(model, data, target, optimizer, train=True):
        loss = nn.functional.nll_loss(model(data), target)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss


class ToyDatasetPipeline(TrainValidationSplitMixin, MLPClassificationPipeline):
    def __init__(self, split, **kwargs):
        iris = sklearn.datasets.load_iris()
        dataset = ScikitLearnDataset(iris)
        split = [int(s * len(dataset)) for s in split]
        super().__init__(dataset=dataset, split=split, **kwargs)


class OpenMLTabularPipeline(KFoldMixin, MLPClassificationPipeline):
    def __init__(self, data_id, data_home=None, **kwargs):
        dataset = sklearn.datasets.fetch_openml(
            data_id=data_id, data_home=data_home, as_frame=False
        )
        dataset = ScikitLearnDataset(dataset)
        super().__init__(dataset=dataset, **kwargs)
