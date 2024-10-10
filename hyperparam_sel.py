import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

class HyperParameter_Opt:

    def __init__(self, network, configuration=None, loss_function=None, results_dir=None, memory_report=False,
                compiler=False, mixed_precision=False, results_module=None, wandb_report=False):

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, garbage_collection_threshold:0.6'
        torch.cuda.empty_cache()

        self.results_dir = results_dir
        if memory_report:
            self.profiler = self.start_memory_reports()
        else:
            self.profiler = None


        self.device = NNUtils.get_device()
        print("Training on {}".format(self.device))
        network = network.to(self.device)
        if not compiler:
            self.network = network
        else:
            self.network = torch.compile(network, mode=compiler, dynamic=True)
        self.loss = loss_function

        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.network = None
        self.validation_set = None
        slef.train_set = None

    def get_datasets(self, train_dataset, val_dataset):
        self.validation_set = val_dataset
        self.train_set = train_dataset

    def set_frozen_params(self, config):
        self.asynchronization = config.train_parameters["asyncrhonization"]
        self.epochs = config.train_parameters["epochs"]
        self.optimizer_module = config.train_parameters["optimizer"]
        self.weight_decay = config.train_parameters["weight_decay"]
        self.num_workers = config.train_parameters["num_workers"]
        self.stratified = config.train_parameters["stratified"]
        self.bucketing = config.train_parameters["bucketing"]
        self.warm_up = config.train_parameters["warm_up"]
        self.early_stopping = config.train_parameters["early_stopping"]

    def train_pass(self, train_loader, batch_size, profiler=None, asynchronity=False, clipping=False):

        self.network.train()

        loss_lst = []
        lr_rate_lst = []
        loss_pass = 0.
        count = 0
        len_dataloader = len(train_loader)
        labels_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device, dtype=int)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device)
        batch_n = 0

        for batch in tqdm(train_loader):
            pos_first, pos_last = count, count+batch_size
            if self.profiler is not None:
                self.profiler.step()
            embeddings = batch["Input"]
            labels = batch["PathoPhenotype"]
            lengths = batch["Protein Count"]
            #  sending data to device
            embeddings = embeddings.to(self.device, non_blocking=asynchronity)
            labels = labels.to(self.device, non_blocking=asynchronity)
            lengths = lengths.to(self.device, non_blocking=asynchronity)
            #  resetting gradients
            self.optimizer.zero_grad(set_to_none=True)
            #  making predictions
            with torch.autocast(device_type=self.device, enabled=self.mixed_precision):
                predictions_logit, attentions = self.network(embeddings, lengths)
                predictions, loss = self.calculate_loss(
                                        predictions_logit=predictions_logit, labels=labels)
            #  computing gradients
            self.scaler.scale(loss).backward()
            if clipping:
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), clipping)
            #  updating weights
            if isinstance(self.optimizer, dict):
                pass
            else:
                self.scaler.step(self.optimizer)
            lr_rate_lst.append(self.optimizer.param_groups[-1]['lr'])
            self.scaler.update()
            #  computing loss
            loss_c = loss.detach()
            pred_c = predictions.detach()
            labels = labels.detach()
            labels_tensor[pos_first:pos_last,:] = labels
            pred_tensor[pos_first:pos_last,:] = pred_c

            loss_pass += loss_c
            self.results_training.add_step_info(loss_train=loss_c, lr=self.optimizer.param_groups[-1]['lr'], batch_n=batch_n,
                                             len_dataloader=len_dataloader)
            #  clean gpu (maybe unnecessary)
            count += batch_size
            batch_n += 1
        loss_pass = loss_pass/batch_n
        if pred_tensor.size()[1] == 2:
            pred_tensor = pred_tensor[:,0] - pred_tensor[:,1]
            pred_tensor = (pred_tensor+1)/2
            labels_tensor = labels_tensor[:,0] - labels_tensor[:, 1]
            labels_tensor = (labels_tensor+1)/2

        mcc_pass = Metrics.calculate_MCC(labels=labels_tensor, predictions=pred_tensor, device=self.device)
        if self.lr_scheduler is not None and self.lr_scheduler.__class__.__name__ == "OneCycleLR":
            self.update_scheduler()
        return loss_pass, mcc_pass, lr_rate_lst, profiler
    
    def val_pass(self, val_loader, batch_size, profiler=None, asynchronity=False):

        self.network.eval()

        loss_lst = []

        loss_pass = 0.
        count = 0
        len_dataloader = len(val_loader)
        labels_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device, dtype=int)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device)
        batch_n = 0

        with torch.inference_mode():
            for batch in tqdm(val_loader):
                pos_first, pos_last = count, count+batch_size
                if self.profiler is not None:
                    self.profiler.step()
                embeddings = batch["Input"]
                labels = batch["PathoPhenotype"]
                lengths = batch["Protein Count"]
                #  sending data to device
                embeddings = embeddings.to(self.device, non_blocking=asynchronity)
                labels = labels.to(self.device, non_blocking=asynchronity)
                lengths = lengths.to(self.device, non_blocking=asynchronity)
                #  making predictions
                predictions_logit, attentions = self.network(embeddings, lengths)
                predictions, loss = self.calculate_loss(
                                         predictions_logit=predictions_logit, labels=labels)
                #  computing loss
                loss_c = loss.detach()
                pred_c = predictions.detach()
                labels = labels.detach()
                loss_pass += loss_c

                labels_tensor[pos_first:pos_last,:] = labels
                pred_tensor[pos_first:pos_last,:] = pred_c

                #  clean gpu (maybe unnecessary
                batch_n += 1
                count += batch_size
        loss_pass = loss_pass/batch_n
        if pred_tensor.size()[1] == 2:
            pred_tensor = pred_tensor[:,0] - pred_tensor[:,1]
            pred_tensor = (pred_tensor+1)/2
            labels_tensor = labels_tensor[:,0] - labels_tensor[:, 1]
            labels_tensor = (labels_tensor+1)/2

        mcc_pass = Metrics.calculate_MCC(labels=labels_tensor, predictions=pred_tensor, device=self.device)
        return loss_pass, mcc_pass, profiler

    def train_model(self, epochs, batch_size, optimizer=torch.optim.Adam, learning_rate=1e-5, weight_decay=1e-4,
            lr_schedule=False, end_lr=3/2, amsgrad=False, num_workers=2, asynchronity=False, stratified=False,
            fused_OptBack=False, clipping=False, bucketing=False, warmup_period=False, early_stopping=True, keep_model=False):


        pos_weight = train_dataset.get_weights()

        self.loss_function = self.loss()

        #  creating dataloaders
        train_loader = NN_Data.load_data(self.train_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=stratified)
        val_loader = NN_Data.load_data(self.val_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=False)


        steps = steps=len(train_loader)
        self.set_optimizer(epochs=epochs, steps=steps, optimizer=optimizer, learning_rate=learning_rate,
                weight_decay=weight_decay, lr_schedule=lr_schedule, end_lr=end_lr, amsgrad=amsgrad, fused_OptBack=fused_OptBack,
                warmup_period=warmup_period)
        if early_stopping:
            early_stopping_method = EarlyStopping(patience=7, delta=0.001)
        else:
            early_stopping_method = None

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            start_e_time = time.time()
            #  training
#            with torch.autograd.set_detect_anomaly(True):
            loss_train, mcc_t, lr_rate, profiler = self.train_pass(train_loader=train_loader, batch_size=batch_size,
                                                                        clipping=clipping, asynchronity=asynchronity)
            #  validation
            print('validating...')
            loss_val, mcc_v, profiler = self.val_pass(val_loader=val_loader, asynchronity=asynchronity, batch_size=batch_size)

            self.results_training.add_epoch_info(epoch=epoch, loss_t=loss_train, loss_v=loss_val,
                                                   mcc_t=mcc_t, mcc_v=mcc_v)

            if keep_model == "best_epoch":
                self.saved_model = self.best_epoch_retain(new_val=mcc_v, optimizer=self.optimizer, model=self.network, epoch=epoch, loss=loss_train)

            if self.lr_scheduler is not None and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.update_scheduler(value=loss_val)
            end_e_time = time.time()
            self.results_training.add_time_info(end_e_time-start_e_time)
            print("training_loss: {}, validation_loss: {} // training_mcc: {}, validation_mcc: {} ".format(loss_train, loss_val, mcc_t, mcc_v))
            if early_stopping:
                stop = early_stopping_method(val_measure=mcc_v)
                if stop:
                    break
        if keep_model == "last_epoch":
            self.saved_model = {"epoch": epoch, 'model_state_dict': self.network.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': loss_train, "val_measure": mcc_v}
        if self.swa_iter:
            swa_model, swa_loss, mcc_v_swa = self.swa_pass(state_dict_model=self.network.state_dict(), optimizer=self.optimizer,
                                                            swa_iter=self.swa_iter, train_loader=train_loader, val_loader=val_loader, batch_size=batch_size,
                                                            asynchronity=asynchronity)
            self.saved_model = {"epoch": epoch+self.swa_iter, "model_state_dict": swa_model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": swa_loss, "val_measure":mcc_v_swa}
        if self.profiler is not None:
            self.stop_memory_reports()
        return mcc_v

    def hyperparam_pass(self, config):
        should_checkpoint = config.get("should_checkpoint", False)
        use_cuda =torch.cuda.isavailable()
        device = torch.device("cuda" if use_cuda else "cpu")

        self.network = self.model_carcass(input_dim=self.config.model_parameters["input_dim"],
                                    num_classes=self.config.model_parameters["out_dim"],
                                    num_blocks=config["num_blocks"],
                                    block_dims=config["block_dims"],
                                    stochastic_depth_prob=config["stochastic_depth_prob"],
                                    attention_dim=config["att_dim"],
                                    attention_norm=self.config.model_parameters["model_structure"]["att_norm"],
                                    dropout_att=config["att_dropout"],
                                    layer_scale=self.config.train_parameters["norm_scale"],
                                    sequence_dropout=config["sequence_dropout"],
                                    norm=self.config.model_parameters["norm"],
                                    length_information=self.config.model_parameters["length_information"],
                                    length_dim=self.config.model_parameters["length_dim"])
        while True:
            mcc_v = train_model(self.epochs, config["batch_size"], optimizer=self.optimizer_module, learning_rate=config["lr"], weight_decay=self.weight_decay,
                        lr_schedule=self.lr_schedule, end_lr=3/2, amsgrad=False, num_workers=self.num_workers, asynchronity=self.asynchronity, stratified=self.stratified,
                        fused_OptBack=False, clipping=False, bucketing=self.bucketing, warmup_period=self.warm_up, early_stopping=self.early_stopping, keep_model=False)
            metrics = {"mean_accuracy": acc}
            # Report metrics (and possibly a checkpoint)
            if should_checkpoint:
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(model.state_dict(), os.path.join(tempdir, "model.pt"))
                    train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
            else:
                train.report(metrics)




    def __call__(self, config, num_samples, max_num_epochs, gpus_per_trial):
        scheduler = ASHAScheduler(max_t=max_num_epochs, grace_period=1,
                                    reduction_factor=2)

        tuner = tune.Tuner(tune.with_resources(
                                tune.with_parameters(self.hyperparam_pass),
                                resources={"cpu": 2, "gpu": gpus_per_trial}),
                           tune_config=tune.TuneConfig(
                                metric="loss",
                                mode="min",
                                scheduler=scheduler,
                                num_samples=num_samples,),
                           param_space=config,)
        results = tuner.fit()

        best_result = results.get_best_result("loss", "min")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
        print("Best trial final validation accuracy: {}".format(best_result.metrics["accuracy"]))
