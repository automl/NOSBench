import time
import copy
from dataclasses import dataclass
from abc import abstractmethod

import torch
from torch import nn
from torch.utils.data import Dataset, random_split


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


@dataclass
class Pipeline:
    save_program: bool = True
    save_training_losses: bool = True
    save_validation_losses: bool = True
    save_test_losses: bool = True
    save_torch_state: bool = True

    def query(self, state_dict, epochs):
        torch.manual_seed(42)
        train, val, test = self.dataset
        model = self.create_model()
        program = state_dict["program"]
        optimizer_class = program.optimizer()
        optimizer = optimizer_class(model.parameters(), **self.optimizer_kwargs)
        if state_dict["torch_state"] is not None:
            model.load_state_dict(state_dict["torch_state"]["model"])
            optimizer.load_state_dict(state_dict["torch_state"]["optimizer"])
        training_losses = state_dict["training_losses"]
        validation_losses = state_dict["validation_losses"]
        test_losses = state_dict["test_losses"]
        total_cost = state_dict["cost"]
        for epoch in range(state_dict["epoch"], epochs):
            prev_time = time.time()
            data, target = train[:]
            loss = self.eval(model, data, target, optimizer, train=True)
            if self.save_training_losses:
                training_losses.append(loss.item())
            total_cost += time.time() - prev_time

            if self.save_validation_losses:
                with torch.no_grad():
                    loss = self.eval(model, data, target, optimizer, train=False)
                    validation_losses.append(loss.item())

            if self.save_test_losses:
                with torch.no_grad():
                    data, target = test[:]
                    loss = self.eval(model, data, target, optimizer, train=False)
                    test_losses.append(loss.item())

        if self.save_torch_state:
            torch_state = {
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }

        return {
            "program": program,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "test_losses": test_losses,
            "torch_state": torch_state,
            "cost": total_cost,
            "epoch": epoch + 1,
        }

    @staticmethod
    @abstractmethod
    def eval(model, data, target, optimizer, train=True):
        pass

    @staticmethod
    @abstractmethod
    def create_model():
        pass


class MLPClassificationPipeline(Pipeline):
    def __init__(
        self,
        dataset,
        hidden_layers,
        split=[0.8, 0.1, 0.1],
        optimizer_kwargs={"lr": 0.0001},
        **kwargs
    ):
        super().__init__(**kwargs)
        split = [int(s * len(dataset)) for s in split]
        generator = torch.Generator().manual_seed(42)
        self.dataset = random_split(dataset, split, generator=generator)
        input_size = len(dataset.feature_names)
        output_size = len(dataset.target_names)
        self.layers = [input_size, *hidden_layers, output_size]
        self.optimizer_kwargs = optimizer_kwargs

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
