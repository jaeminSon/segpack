import torch

__all__ = ['Loss', 'CrossEntropy']

class Loss(object):
    
    def __new__(self, config):
        if config["type"] == "cross_entropy":
            return CrossEntropy()
        else: # add loss function class here
            raise NotImplementedError()


class CrossEntropy(object):
    def __init__(self) -> None:
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        try:
            criterion = criterion.cuda()
        except:   
            pass
        
    def __call__(self, logit:torch.tensor, target:torch.tensor) -> torch.tensor:
        return self.criterion(logit, target.long()) / logit.size()[0]
