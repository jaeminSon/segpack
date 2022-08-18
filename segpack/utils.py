import os
import warnings
from pathlib import Path

import torch

import airszoo


def ignore_error(func):
    def fun(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            pass
    return fun


def load_network(pretrained_model: str, cuda: bool = True):
    try:
        network = airszoo.get_pretrained_network(pretrained_model)
        warnings.warn("Load checkpoint {}.".format(pretrained_model), UserWarning)
        if cuda:
            return network.cuda()
        else:
            return network
    except:
        network = airszoo.instantiate_network(pretrained_model)
        warnings.warn("Instantiate architecture without loading checkpoint ({}).".format(pretrained_model), UserWarning)
        if cuda:
            return network.cuda()
        else:
            return network


def get_optimizer(params, init_lr: float, config: str):
    optimizer = torch.optim.AdamW(params, init_lr)
    try:
        optimizer.load_state_dict(airszoo.get_optimizer_state_dict(config))
        return optimizer
    except:
        return optimizer


def save(state: dict, save_path: Path, epoch: int, save_interval: int):
    if epoch % save_interval == 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(state, save_path)


def cudafy(*list):
    if torch.cuda.is_available():
        return [l.cuda() for l in list]
    else:
        return list
