import argparse
from functools import partial
from collections import namedtuple
import math
import pathlib
import json
import pickle

import nosbench
from nosbench.program import NamedProgram
from nosbench.optimizers import *
from ConfigSpace import Configuration
import torch
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt
import numpy as np


_Element = namedtuple("_Element", "cls fitness")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CosineAdamW(torch.optim.AdamW):
    def __init__(self, warmup_epochs, epochs, *args, **kwargs):
        print(args, kwargs)
        super().__init__(*args, lr=0.001)
        self.scheduler = get_cosine_schedule_with_warmup(self, warmup_epochs, epochs)
        self.counter = 0

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        self.counter += 1
        if self.counter % 100 == 0:
            self.scheduler.step()
            print(self.counter)


class Program:
    def __init__(self, optimizer, *args, **kwargs):
        self.optim = optimizer
        self.args = args
        self.kwargs = kwargs

    def optimizer(self):
        print(self.args, self.kwargs)
        return partial(self.optim, *self.args, **self.kwargs)

    def __hash__(self):
        return 42

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", type=str, required=True)
    args = parser.parse_args()

    # benchmark = nosbench.create("toy")
    benchmark = nosbench.create("pfn", path="asdfasdfasdf", device="cuda")

    colors = [
        "#6929c4",
        "#1192e8",
        "#005d5d",
        "#9f1853",
        "#fa4d56",
        "#570408",
        "#198038",
        "#002d9c",
        "#ee538b",
        "#b28600",
        "#009d9a",
        "#012749",
        "#8a3800",
        "#a56eff",
        "#ff7f00",
        "#8b8b00",
    ]

    # algorithms = {}
    # max_len = 0
    # ys = []
    # for path in map(pathlib.Path, args.paths):
    #     with path.open("r") as f:
    #         metadata = json.load(f)
    #         results = pickle.load((path.parent / path.stem).with_suffix(".pickle").open("rb"))
    #         losses = []
    #         min_loss = np.inf
    #         max_len = max(len(results), max_len)
    #         print(metadata["search_algorithm"])
    #         for result in results:
    #             if isinstance(result.cls, Configuration):
    #                 loss = -result.fitness
    #             else:
    #                 loss = benchmark.query(result.cls, metadata["args"]["benchmark_epochs"])
    #                 # print(loss)
    #             if not np.isnan(min_loss) and loss < min_loss:
    #                 print(result.cls)
    #                 min_loss = loss
    #             losses.append(min_loss)
    #         ys.append(losses)
    #         algorithms[metadata["search_algorithm"]] = losses
    
    # for algorithm, losses in algorithms.items():
    #     if len(losses) == 1:
    #         plt.axhline(losses[0], xmax=20, linewidth=2, color=colors.pop(), label=algorithm)
    #     else:
    #         plt.plot(losses, label=algorithm, color=colors.pop())


    # for i, program in enumerate(NamedProgram.get_instances()):
    #     loss = benchmark.query(program, 19)
    #     ys.append([loss])
    #     plt.axhline(loss, xmax=20, linewidth=2, color=colors.pop(), label=program.name)


    lr = 0.001
    epochs=19
    # warmup_epochs=epochs//4
    warmup_epochs = 4
    program = Program(CosineAdamW, warmup_epochs, epochs, lr=0.000001)
    loss = benchmark.query(program, epochs, skip_cache=True)
    print(loss)
    plt.axhline(loss, xmax=20, linewidth=2, color=colors.pop(), label="CosineAdamW")

    # for y in ys:
    #     plt.annotate('%0.2f' % np.max(y), xy=(1, np.max(y)), xytext=(8, 0), 
    #                  xycoords=('axes fraction', 'data'), textcoords='offset points')

    # plt.ylim(-1, 3.0)
    # plt.xscale('log')
    plt.legend()
    plt.savefig('anytime_curves.png', bbox_inches='tight')
    # plt.show()
