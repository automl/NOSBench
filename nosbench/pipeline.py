import time
import copy
from dataclasses import dataclass
from abc import abstractmethod
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import KFold


class ScikitLearnDataset(Dataset):
    def __init__(self, dataset):
        self.data = torch.from_numpy(dataset.data).float()
        self.target = torch.from_numpy(dataset.target).long()
        self.feature_names = dataset.feature_names
        self.target_names = dataset.target_names

    def __getitem__(self, i):
        return self.data[i], self.target[i]

    def __len__(self):
        return len(self.target)


class Pipeline:
    def __init__(
        self,
        dataset,
        batch_size,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.loader_generator = self.get_loader_generator()

    def get_loader_generator(self):
        raise NotImplementedError

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
    def get_epoch_data(state_dict, epoch):
        return {
            "minibatch_losses": state_dict["training_losses"][epoch],
            "validation_loss": state_dict["validation_losses"][epoch],
            "cost": state_dict["costs"][epoch],
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

    @staticmethod
    @abstractmethod
    def eval(model, data, target, optimizer, train=True):
        pass

    @staticmethod
    @abstractmethod
    def create_model():
        pass


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


class KFoldMixin:
    def __init__(self, n_fold, **kwargs):
        self.n_fold = n_fold
        super().__init__(**kwargs)

    def get_loader_generator(self):
        splits = KFold(
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

    def get_epoch_data(self, state_dict, epoch):
        total_validation_loss = 0
        cost = 0
        for fold in range(self.n_fold):
            total_validation_loss += state_dict[fold]["validation_losses"][epoch]
            cost += state_dict[fold]["costs"][epoch]
        return {"validation_loss": total_validation_loss / self.n_fold,
                "cost": cost}

    def query(self, state_dict, n_epochs):
        for fold in range(self.n_fold):
            self._query(state_dict[fold], n_epochs)
        return state_dict

    def initial_state(self, program):
        states = []
        for fold in range(self.n_fold):
            states.append(super().initial_state(program))
        return states


class MLPClassificationPipeline(KFoldMixin, Pipeline):
    def __init__(self, dataset, hidden_layers, **kwargs):
        input_size = len(dataset.feature_names)
        output_size = len(dataset.target_names)
        self.layers = [input_size, *hidden_layers, output_size]
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
