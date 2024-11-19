import torch
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
import pytorch_warmup as warmup
import sys

sys.dont_write_bytecode = True


class Optimizer:

    def __init__(self, network, optimizer, learning_rate, weight_decay, amsgrad):

        self.network = network
        self.optimizer, self.learning_rate = self.set_optimizer(optimizer=optimizer, learning_rate=learning_rate,
                                                                    weight_decay=weight_decay, amsgrad=amsgrad)
        self.lr_scheduler = None
        self.warmup = None

    def get_optimizer(self):
        return self.optimizer

    def set_optimizer(self, optimizer=torch.optim.Adam, learning_rate=1e-5, weight_decay=1e-4,
                        amsgrad=False):
        if optimizer.__class__.__name__ == "Adam" or optimizer.__class__.__name__ == "AdamW":
            optimizer = optimizer(self.network.parameters(),
                                            lr=learning_rate, weight_decay=weight_decay,
                                            amsgrad=amsgrad)
        else:
            optimizer = optimizer(self.network.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay, decoupled_weight_decay=True)
        return optimizer, learning_rate

    def set_scheduler(self, scheduler_type, patience=None, milestones=None, gamma=None,
                             end_lr=None, steps=None, epochs=None):
        if scheduler_type == "OneCycleLR":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                max_lr=self.learning_rate, final_div_factor=end_lr,
                                                epochs=epochs, steps_per_epoch=steps)
        elif scheduler_type == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", patience=20)
        elif scheduler_type == "MultiStepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 15, 55, 70, 80, 90, 95], gamma=0.5)
        else:
            raise ValueError("The scheduler type {} is not available".format(scheduler_type))


    def set_warmup(self, warmup_period):
        self.warmup = {"scheduler": warmup.LinearWarmup(self.optimizer, warmup_period), "period": warmup_period}

    def scheduler_step(self, value=False):
        if value is not False:
            self.lr_scheduler.step(value)
        else:
            self.lr_scheduler.step()

    def update_scheduler(self, value=False):
        if self.warmup is not None:
            with self.warmup["scheduler"].dampening():
                if self.warmup["scheduler"].last_step + 1 >= self.warmup["period"]:
                    self.scheduler_step(value=value)
        else:
            self.scheduler_step(value=value)

