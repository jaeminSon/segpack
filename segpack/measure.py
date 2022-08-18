import numpy as np

import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from skimage import color

__all__ = ['Evaluator']

class Evaluator(object):
    def __init__(self, config, tb_home_dir=None) -> None:
        assert "train" in config, "config has no 'train' tag." 
        assert "eval" in config, "config has no 'eval' tag." 
        assert type(config["train"]) == list, "'train' tag of config should be a list."
        assert type(config["eval"]) == list, "'eval' tag of config should be a list."
        
        self.measure = {"train": set(config["train"]),
                        "eval": set(config["eval"])}
        self.reset()
        
        self.tb_writer = SummaryWriter(log_dir="tensorboard" if tb_home_dir is None else tb_home_dir)
        self.epoch_counter = 0
        self.iteration_counter = 0
        
    def get_measure(self, loop_type) -> set:
        return self.measure[loop_type]
    
    def IoU(self) -> np.array:
        return np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
    
    def reset(self) -> None:
        self.metrics = {}
        self.confusion_matrix = None
        
    def update(self, dict_metrics) -> None:
        for metric, val in dict_metrics.items():
            self.metrics.setdefault(metric, {"numerator":0, "denominator":0, "average":0})
            self.metrics[metric]["numerator"] += val
            self.metrics[metric]["denominator"] += 1
            self.metrics[metric]["average"] = 1. * self.metrics[metric]["numerator"] / self.metrics[metric]["denominator"]
        
    def _compute_confusion_matrix(self, pred, target) -> np.array:
        n_classes = max(pred.astype('int').max(), target.astype('int').max()) + 1
        label = n_classes * target.astype('int') + pred.astype('int')
        count = np.bincount(label.flatten(), minlength=n_classes**2)
        return count.reshape(n_classes, n_classes)
        
    def _upate_confusion_matrix(self, incoming_confusion_matrix) -> None:
        if self.confusion_matrix is None:
            self.confusion_matrix = incoming_confusion_matrix
        else:
            if self.confusion_matrix.shape == incoming_confusion_matrix.shape:
                self.confusion_matrix += incoming_confusion_matrix
            else:
                new = np.zeros((max(self.confusion_matrix.shape[0], incoming_confusion_matrix.shape[0]), 
                                max(self.confusion_matrix.shape[1], incoming_confusion_matrix.shape[1])))
                def add(a,b):
                    for i in range(b.shape[0]):
                        for j in range(b.shape[1]):
                            a[i][j] += b[i][j]
                add(new, self.confusion_matrix)
                add(new, incoming_confusion_matrix)
                self.confusion_matrix = new
                    
    def update_confusion_matrix(self, pred:np.array, target:np.array) -> None:
        self._upate_confusion_matrix(self._compute_confusion_matrix(pred, target))

    def average(self, metric) -> float:
        return self.metrics[metric]["average"]

    def summary(self) -> dict:
        return self.metrics

    def increment_epoch(self) -> None:
        self.epoch_counter+=1

    def increment_iteration(self) -> None:
        self.iteration_counter+=1

    def write2tensorboard(self, varname:str, value:float, time_scale:str) -> None:
        if time_scale == "iteration":
            x = self.iteration_counter
        elif time_scale == "epoch":
            x = self.epoch_counter
        else:
            raise ValueError("Unknown time scale {}".format(time_scale))
        
        self.tb_writer.add_scalar(varname, value, x)
        
    def visualize_image(self, image:torch.tensor, output:torch.tensor, target:torch.tensor, loop_type:str, n_display=2) -> None:
        def segmap2colormap(segmap:torch.tensor):
            colormap = []
            for s in segmap:
                rgb = color.label2rgb(s.detach().cpu().numpy())
                colormap.append(rgb)
            colormap = torch.from_numpy(np.array(colormap).transpose([0, 3, 1, 2]))
            return colormap
        
        self.tb_writer.add_image('{}/Image'.format(loop_type), make_grid(image[:n_display].clone().cpu().data, n_display, normalize=True), self.epoch_counter)
        self.tb_writer.add_image('{}/Prediction'.format(loop_type), make_grid(segmap2colormap(torch.max(output[:n_display], 1)[1]), n_display, normalize=True) , self.epoch_counter)
        self.tb_writer.add_image('{}/Groundtruth'.format(loop_type), make_grid(segmap2colormap(torch.squeeze(target[:n_display], 1)), n_display, normalize=True), self.epoch_counter)
