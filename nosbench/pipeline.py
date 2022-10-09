import time
import copy
from abc import ABC, abstractmethod

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


class Pipeline(ABC):
    @abstractmethod
    def query(self, state_dict, epochs):
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

    @staticmethod
    def classification_mlp_model(layers):
        module_list = []
        prev_layer = layers[0]
        for layer in layers[1:-1]:
            module_list.append(nn.Linear(prev_layer, layer))
            module_list.append(nn.ReLU())
            prev_layer = layer
        module_list.append(nn.Linear(prev_layer, layers[-1]))
        module_list.append(nn.LogSoftmax(-1))
        return nn.Sequential(*module_list)

    def query(self, state_dict, epochs):
        torch.manual_seed(42)
        train, val, test = self.dataset
        model = self.classification_mlp_model(self.layers)
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
        start = len(state_dict["training_losses"])
        for epoch in range(start, epochs):
            prev_time = time.time()
            data, target = train[:]
            loss = nn.functional.nll_loss(model(data), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())
            total_cost += time.time() - prev_time

            with torch.no_grad():
                data, target = val[:]
                output = model(data)
                loss = nn.functional.nll_loss(output, target)
                validation_losses.append(loss.item())

                data, target = val[:]
                output = model(data)
                loss = nn.functional.nll_loss(output, target)
                test_losses.append(loss.item())

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
        }
