import random
import os
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

import torch

import numpy as np

from tabpfn.scripts.model_builder import get_model, get_default_spec, save_model, load_model
from tabpfn.utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler

from tabpfn.scripts.model_configs import *

from tabpfn.priors.utils import uniform_int_sampler_f

large_datasets = True
max_samples = 10000 if large_datasets else 5000
bptt = 10000 if large_datasets else 3000
suite='cc'

device = 'cuda'
base_path = '.'
max_features = 100
maximum_runtime = 24


def train_function(config_sample, i, optimizer, scheduler, add_name=''):
    start_time = time.time()
    N_epochs_to_save = 5
    # counter = 0
    
    def save_callback(model, epoch):
        if not hasattr(model, 'last_saved_epoch'):
            model.last_saved_epoch = 0
        save_model(model, base_path, f'models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{model.last_saved_epoch}.cpkt', config_sample)
        # if counter % N_epochs_to_save == 0:
        #     print('Saving model..')
        #     config_sample['epoch_in_training'] = epoch
        model.last_saved_epoch = model.last_saved_epoch + 1 # TODO: Rename to checkpoint
        # counter += 1
    
    results = get_model(config_sample
                      , device
                      , optimizer
                      , scheduler
                      , should_train=True
                      , verbose=2
                      , epoch_callback = save_callback)
    
    if 'LOCAL_RANK' in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0

    if rank == 0:
        torch.save(results[2], "model_adamw.pt")
        torch.save(results[4], "results_adamw.pt")
    

    return

def reload_config(config_type='causal', task_type='multiclass', longer=0):
    config = get_prior_config(config_type=config_type)
    
    config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    
    model_string = ''
    
    config['epochs'] = 150
    config['recompute_attn'] = True

    config['max_num_classes'] = 10
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    return config, model_string

config, model_string = reload_config(longer=1)

config['bptt_extra_samples'] = None

# diff
config['output_multiclass_ordered_p'] = 0.
del config['differentiable_hyperparameters']['output_multiclass_ordered_p']

config['multiclass_type'] = 'rank'
del config['differentiable_hyperparameters']['multiclass_type']

config['sampling'] = 'normal' # vielleicht schlecht?
del config['differentiable_hyperparameters']['sampling']

config['pre_sample_causes'] = True
# end diff

config['multiclass_loss_type'] = 'nono' # 'compatible'
config['normalize_to_ranking'] = False # False

config['categorical_feature_p'] = .2 # diff: .0

# turn this back on in a random search!?
config['nan_prob_no_reason'] = .0
config['nan_prob_unknown_reason'] = .0 # diff: .0
config['set_value_to_nan'] = .1 # diff: 1.

config['normalize_with_sqrt'] = False

config['new_mlp_per_example'] = True
config['prior_mlp_scale_weights_sqrt'] = True
config['batch_size_per_gp_sample'] = None

config['normalize_ignore_label_too'] = False

config['differentiable_hps_as_style'] = False
config['max_eval_pos'] = 1000

config['random_feature_rotation'] = True
config['rotate_normalized_labels'] = True

config["mix_activations"] = False # False heisst eig True

config['emsize'] = 512
config['nhead'] = config['emsize'] // 128
config['bptt'] = 1024+128
config['canonical_y_encoder'] = False

    
config['aggregate_k_gradients'] = 8
config['batch_size'] = 8*config['aggregate_k_gradients']
config['num_steps'] = 1024//config['aggregate_k_gradients']
config['epochs'] = 150
config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

config['train_mixed_precision'] = True
config['efficient_eval_masking'] = True

config['lr'] = 0.001

config_sample = evaluate_hypers(config)

optimizer = partial(torch.optim.AdamW, lr=0.001, weight_decay=0.0)
scheduler = get_cosine_schedule_with_warmup
train_function(config_sample, i=1, optimizer=optimizer, scheduler=scheduler, add_name="adamw")

