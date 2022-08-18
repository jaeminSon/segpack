from typing import Iterable, Callable
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch

from .loss import Loss
from .scheduler import Scheduler
from .measure import Evaluator
from .utils import cudafy, load_network, get_optimizer, save

import airszoo

__all__ = ['loop', 'train']

def loop(network: torch.nn.Module, dataloader: Iterable, loop_type: str, evaluator: Evaluator, criterion=None, optimizer=None):

    def set_network_state(network, loop_type):
        if loop_type == "train":
            network.train()
        elif loop_type == "eval":
            network.eval()
        else:
            raise ValueError(
                "loop_type {} not known. Only 'train' or 'eval' available.".format(loop_type))

    def gradient_descent(image: torch.tensor, target: torch.tensor, network: torch.nn.Module, criterion: Callable, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()
        output = network(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        return output, loss.item()

    def update_measure_per_itr(output: torch.tensor, target: torch.tensor, evaluator: Evaluator, loop_type: str, loss: float = None):
        if "loss" in evaluator.get_measure(loop_type):
            if loop_type == "train":
                evaluator.update({"loss": loss})
            elif loop_type == "eval":
                loss = criterion(output, target)
                evaluator.update({"loss": loss.item()})
            evaluator.write2tensorboard("{}/loss".format(loop_type), loss, "iteration")

        if "IoU" in evaluator.get_measure(loop_type):
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            evaluator.update_confusion_matrix(pred, target)

        evaluator.increment_iteration()

    def update_measure_per_epoch(evaluator: Evaluator, loop_type: str):
        if "IoU" in evaluator.get_measure(loop_type):
            iou_per_class = evaluator.IoU()
            # ignore background (class=0)
            evaluator.update(
                {"mIoU": np.nanmean([v for i, v in enumerate(iou_per_class) if i != 0])})
            for i in range(1, len(iou_per_class)):
                evaluator.update({"IoU-class{}".format(i): iou_per_class[i]})
            
            evaluator.write2tensorboard("{}/mIoU".format(loop_type), evaluator.average("mIoU"), "epoch")
            for i in range(1, len(iou_per_class)):
                evaluator.write2tensorboard("{}/IoU-class{}".format(loop_type, i), evaluator.average("IoU-class{}".format(i)), "epoch")
            
        evaluator.increment_epoch()

    set_network_state(network, loop_type)
    evaluator.reset()
    tbar = tqdm(dataloader)
    for batch in tbar:
        image, target = cudafy(batch[0], batch[1])
        if loop_type == "train":
            output, loss = gradient_descent(
                image, target, network, criterion, optimizer)
            tbar.set_description('train loss: {}'.format(loss))
        elif loop_type == "eval":
            with torch.no_grad():
                output = network(image)
        update_measure_per_itr(
            output, target, evaluator, loop_type, loss=loss if loop_type == "train" else None)

    update_measure_per_epoch(evaluator, loop_type)
    evaluator.visualize_image(image, output, target, loop_type)


def train(train_data, val_data, pretrained_network: str, config: str):
    def pathfy(data):
        if type(data) == list:
            data = [Path(d) if Path(d).exists() else d for d in data]
        elif Path(data).exists():
            data = Path(data)
        return data
    
    train_data = pathfy(train_data)
    val_data = pathfy(val_data)
    
    hyperparam = airszoo.get_hyperparam(config)
    assert not Path(hyperparam["save"]["checkpoint"]).exists(), "{} already exists.".format(hyperparam["save"]["checkpoint"])
    assert not Path(hyperparam["save"]["tensorboard"]).exists(), "{} already exists.".format(hyperparam["save"]["tensorboard"])
    network = load_network(pretrained_network)
    optimizer = get_optimizer(network.parameters(), hyperparam["lr_scheduler"]["init_lr"], config)
    lr_scheduler = Scheduler(hyperparam["lr_scheduler"])
    criterion = Loss(hyperparam["loss"])
    evaluator = Evaluator(hyperparam["measure"], hyperparam["save"]["tensorboard"] if "tensorboard" in hyperparam["save"] else None)

    preprocess = airszoo.get_preprocess_name_used_for_train(pretrained_network)
    augment_train = hyperparam.get("augment_train")
    augment_val = hyperparam.get("augment_val")
    train_dataloader = airszoo.get_dataloader(train_data, 
                                              preprocess, 
                                              augment_train,
                                              **{"num_workers": hyperparam["num_workers"], 
                                                 "pin_memory": True, 
                                                 "batch_size": hyperparam["batch_size"]["train"], 
                                                 "shuffle": True})
    val_dataloader = airszoo.get_dataloader(val_data, 
                                            preprocess, 
                                            augment_val,
                                            **{"num_workers": hyperparam["num_workers"], 
                                               "pin_memory": True, 
                                               "batch_size": hyperparam["batch_size"]["eval"], 
                                               "shuffle": False})

    total_epochs = 2**31 if hyperparam["epochs"] is None else hyperparam["epochs"]

    for epoch in range(total_epochs):
        # train loop
        loop(network, train_dataloader, "train", evaluator, criterion, optimizer)
        
        # val loop
        loop(network, val_dataloader, "eval", evaluator)
        
        # update learning rate scheduler
        lr_scheduler(optimizer, epoch, evaluator.average("mIoU"))
        
        # checkpoint (or save) model
        save({"state_dict": network.state_dict(), "optimizer": optimizer.state_dict()},
             Path(hyperparam["save"]["checkpoint"]) /
             "epoch-{}.pth".format(str(epoch).zfill(5)),
             epoch,
             hyperparam["save"]["interval"])
