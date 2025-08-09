import torch
import os
from pfns import priors, encoders, utils, bar_distribution, train
from ConfigSpace import hyperparameters as CSH
from functools import partial
import random
import time
import warnings
from datetime import datetime
from functools import partial
from typing import Callable, NewType, Type, List
from dataclasses import dataclass
from itertools import chain
from functools import cached_property
import copy
from collections import defaultdict
import weakref
import torch
import math


import warnings

warnings.filterwarnings("ignore")

get_batch_function = priors.get_batch_sequence(
    priors.hebo_prior.get_batch,
    priors.utils.sample_num_feaetures_get_batch,
)

config_heboplus = {
    "priordataloader_class_or_get_batch": get_batch_function,
    "encoder_generator": encoders.get_normalized_uniform_encoder(
        encoders.get_variable_num_features_encoder(encoders.Linear)
    ),
    "emsize": 512,
    "nhead": 4,
    "warmup_epochs": 5,
    "y_encoder_generator": encoders.Linear,
    "batch_size": 128,
    # "scheduler": None,
    "scheduler": utils.get_cosine_schedule_with_warmup,
    "extra_prior_kwargs_dict": {
        "num_features": 18,
        "hyperparameters": {
            "lengthscale_concentration": 1.2106559584074301,
            "lengthscale_rate": 1.5212245992840594,
            "outputscale_concentration": 0.8452312502679863,
            "outputscale_rate": 0.3993553245745406,
            "add_linear_kernel": False,
            "power_normalization": False,
            "hebo_warping": False,
            "unused_feature_likelihood": 0.3,
            "observation_noise": True,
        },
    },
    "epochs": 50,
    "lr": 0.0001,
    "seq_len": 60,
    "single_eval_pos_gen": utils.get_uniform_single_eval_pos_sampler(
        50, min_len=1
    ),  # <function utils.get_uniform_single_eval_pos_sampler.<locals>.<lambda>()>,
    "aggregate_k_gradients": 2,
    "nhid": 1024,
    "steps_per_epoch": 1024,
    "weight_decay": 0.0,
    "train_mixed_precision": False,
    "efficient_eval_masking": True,
    "nlayers": 12,
}


# now let's add the criterions, where we decide the border positions based on the prior
def get_ys(config, device="cuda:0"):
    bs = 128
    all_targets = []
    for num_hps in [
        2,
        8,
        12,
    ]:  # a few different samples in case the number of features makes a difference in y dist
        b = get_batch_function(
            bs,
            1000,
            num_hps,
            epoch=0,
            device=device,
            hyperparameters={
                **config["extra_prior_kwargs_dict"]["hyperparameters"],
                "num_hyperparameter_samples_per_batch": -1,
            },
        )

        all_targets.append(b.target_y.flatten())
    return torch.cat(all_targets, 0)


def add_criterion(config, device="cuda:0"):
    return {
        **config,
        "criterion": bar_distribution.FullSupportBarDistribution(
            bar_distribution.get_bucket_limits(1000, ys=get_ys(config, device).cpu())
        ),
    }

optimizer_cls = partial(
    torch.optim.AdamW,
    lr=config_heboplus["lr"],
    weight_decay=config_heboplus["weight_decay"],
)

results = train.train(
    **add_criterion(config_heboplus, device="cuda"), optimizer_cls=optimizer_cls
)
if 'LOCAL_RANK' in os.environ:
    rank = int(os.environ["LOCAL_RANK"])
else:
    rank = 0

if rank == 0:
    torch.save(results[2], "model_adamw.pt")
    torch.save(results[4], "results_adamw.pt")

