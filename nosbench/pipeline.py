import time
import copy
from dataclasses import dataclass
from abc import abstractmethod

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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
        train,
        val,
        test,
        batch_size,
        save_program: bool = True,
        save_training_losses: bool = True,
        save_validation_losses: bool = True,
        save_test_losses: bool = True,
        save_torch_state: bool = True,
        save_costs: bool = True,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.loader = DataLoader(
            train, batch_size=batch_size if batch_size > 0 else len(train)
        )
        self.save_program = save_program
        self.save_training_losses = save_training_losses
        self.save_validation_losses = save_validation_losses
        self.save_test_losses = save_test_losses
        self.save_torch_state = save_torch_state
        self.save_costs = save_costs

    def query(self, state_dict, n_epochs):
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
        test_losses = state_dict["test_losses"]
        costs = state_dict["costs"]
        total_cost = costs[-1] if state_dict["n_epochs"] > 0 else 0
        for epoch in range(state_dict["n_epochs"], n_epochs):
            prev_time = time.time()
            minibatch_losses = []
            for data, target in self.loader:
                loss = self.eval(model, data, target, optimizer, train=True)
                if self.save_training_losses:
                    minibatch_losses.append(loss.item())
            if self.save_training_losses:
                training_losses.append(minibatch_losses)
            total_cost += time.time() - prev_time
            if self.save_costs:
                costs.append(total_cost)

            if self.save_validation_losses:
                with torch.no_grad():
                    data, target = self.val[:]
                    loss = self.eval(model, data, target, optimizer, train=False)
                    validation_losses.append(loss.item())

            if self.save_test_losses:
                with torch.no_grad():
                    data, target = self.test[:]
                    loss = self.eval(model, data, target, optimizer, train=False)
                    test_losses.append(loss.item())

        if self.save_torch_state:
            torch_state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

        state_dict["n_epochs"] = n_epochs
        return {
            "program": program,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "test_losses": test_losses,
            "torch_state": torch_state,
            "costs": costs,
            "n_epochs": n_epochs,
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
    def __init__(self, input_size, hidden_layers, output_size, **kwargs):
        self.layers = [input_size, *hidden_layers, output_size]
        super().__init__(**kwargs)

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
