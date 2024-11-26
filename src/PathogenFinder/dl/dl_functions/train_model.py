import torch
import time

from dl.utils.optimizer_utils import Optimizer
from dl.utils.data_utils import NN_Data
from dl.utils.nn_utils import Network_Module




class Train_NeuralNetwork:


    def __init__(self, network_module, model_storage=False, model_report=False, model_params=None):

        assert model_storage in ["last_epoch", "early_stopping", "best_epoch"]

        self.network_module = network_module

        self.train_loader = None
        self.val_loader = None
        self.train_steps = None
        self.batch_size = None
        self.asynchronity = False

        if model_storage == "early_stopping":
            self.early_stopping = EarlyStopping(patience=7, delta=0.001)
        else:
            self.early_stopping = None
        self.model_storage = model_storage
        self.model_report = model_report

        self.optimizer = None


    def set_dataloaders(self, train_dataset, batch_size, val_dataset=None, num_workers=2,
                stratified=False, bucketing=False, asynchronity=False, max_batch_size=45):

        if batch_size > max_batch_size:
            batch_size, accumulate_gradient = Network_Module.set_sensible_batch_size(batch_size)
        else:
            accumulate_gradient = 1
        #  creating dataloaders
        self.train_loader = NN_Data.load_data(train_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=stratified)
        if val_dataset is not None:
            self.val_loader = NN_Data.load_data(val_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=False)
        else:
            self.val_loader = None

        self.train_steps = len(self.train_loader)
        self.batch_size = batch_size
        self.asynchronity = asynchronity

    def set_optimizer(self, optimizer_class=None, optimizer=None, learning_rate=None, weight_decay=None, amsgrad=False, scheduler_type=False,
                        warmup_period=False, patience=None, milestones=None, gamma=None, end_lr=None,
                        steps=None, epochs=None):

        if optimizer is not None and optimizer_class is None:
            optimizer_instance = optimizer
        elif optimizer is None and optimizer_class is not None:
            optimizer_instance = Optimizer(network=self.network_module.network, optimizer=optimizer_class, learning_rate=learning_rate,
                                        weight_decay=weight_decay, amsgrad=amsgrad)
        elif optimizer is not None and optimizer_class is not None:
            raise ValueError("An optimizer and an optimizer class are trying to be set.")
        else:
            raise ValueError("Or an optimizer or an optimizer class have to be set.")

        if scheduler_type:
            optimizer_instance.set_scheduler(scheduler_type=scheduler_type, patience=patience, milestones=milestones,
                                        gamma=gamma,  end_lr=end_lr, steps=self.train_steps, epochs=epochs)
        if warmup_period:
            optimizer_instance.set_warmup(warmup_period=warmup_period)

        self.optimizer = optimizer_instance

    def __call__(self, epochs, model_params=None):

        if self.train_loader is None:
            raise ValueError("Please set dataloaders before training")

        self.model_report.start_train_report(model=self.network_module.network,
                                                criterion=self.network_module.loss_function)

        max_mcc_val = 0

        if model_params is None:
            init_epoch = 0
        else:
            init_epoch = model_params["Epoch"]

        for epoch in range(epochs):
            epoch += init_epoch
            print(f'Epoch {epoch+1}/{epochs}')
            start_e_time = time.time()
            #  training
            print('training...')
            loss_train, mcc_t, lr_rate, self.optimizer = self.network_module.train_pass(train_loader=self.train_loader, batch_size=self.batch_size,
                                                                           results_module=self.model_report, optimizer=self.optimizer, asynchronity=self.asynchronity)

            if self.val_loader is not None:
                #  validation
                print('validating...')
                loss_val, mcc_v = self.network_module.validation_pass(val_loader=self.val_loader, asynchronity=self.asynchronity, batch_size=self.batch_size)

            if self.model_report:
                self.model_report.add_epoch_info(epoch=epoch, loss_t=loss_train, loss_v=loss_val,
                                                   mcc_t=mcc_t, mcc_v=mcc_v, epoch_duration=time.time()-start_e_time)

            if self.optimizer.lr_scheduler is not None:
                if self.optimizer.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.optimizer.update_scheduler(value=loss_val)
                elif self.optimizer.lr_scheduler.__class__.__name__ == "MultiStepLR":
                    self.optimizer.update_scheduler()

            if self.model_storage != "last_epoch":
                if max_mcc_val < mcc_v:
                    self.network_module.save_model(optimizer=self.optimizer.optimizer, loss=loss_train, mcc_val=mcc_v, epoch=epoch)
                    max_mcc_val = mcc_v

            if self.early_stopping is not None:
                stop = self.early_stopping(val_measure=mcc_v)
                if stop:
                    break

        if self.model_storage == "last_epoch":
            self.network_module.save_model(optimizer=self.optimizer.optimizer, loss=loss_train, mcc_val=mcc_v, epoch=epoch)
    
        if self.network_module.memory_profiler:
            self.network_module.memory_profiler.stop_memory_reports()
        if self.model_report:
            self.model_report.finish_report()
        return self.network_module

