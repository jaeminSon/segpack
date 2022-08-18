__all__ = ['Scheduler', 'ReduceLROnPlateauAnnealing']

class Scheduler(object):
    
    def __new__(self, config):
        if config["type"] == "ReduceLROnPlateauAnnealing":
            return ReduceLROnPlateauAnnealing(config["init_lr"],
                                            config["min_lr"],
                                            config["n_epochs_plateau"])
        else: # add new learning rate scheduler class here
            raise NotImplementedError()

        
class ReduceLROnPlateauAnnealing(object):
    """
        lr = baselr * 0.1 if validation measure plateaus (measure does not increase)
        lr = init_lr if lr < min_lr
    """

    def __init__(self, init_lr, min_lr, n_epochs_plateau, decay_factor=0.1) -> None:

        self.lr = init_lr
        self.n_epochs_plateau = n_epochs_plateau
        self.n_epochs_no_updates = 0
        self.epochs = -1
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.decay_factor = decay_factor
        self.best_measure = -float("inf")

    def __call__(self, optimizer, epoch, curr_measure) -> None:
        if self.epochs != epoch:
            self.epochs = epoch
            if self.n_epochs_plateau <= self.n_epochs_no_updates:
                self.n_epochs_no_updates = 0
                if self.lr * self.decay_factor < self.min_lr:
                    self.lr = self.init_lr
                else:
                    self.lr *= self.decay_factor
            else:
                if self.best_measure < curr_measure:
                    self.best_measure = curr_measure
                    self.n_epochs_no_updates = 0
                else:
                    self.n_epochs_no_updates += 1
            print("best:{}, curr:{}, n_epochs_no_updates: {}, lr: {}".format(
                self.best_measure, curr_measure, self.n_epochs_no_updates, self.lr))
        self._adjust_learning_rate(optimizer)

    def _adjust_learning_rate(self, optimizer) -> None:
        assert 0 < self.lr
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = self.lr
