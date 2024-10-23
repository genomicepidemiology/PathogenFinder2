import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import ray
import sys
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
import pytorch_warmup as warmup
from tqdm import tqdm
import time
import logging
import joblib
import pickle
import wandb
import matplotlib.pyplot as plt
from optuna.integration.wandb import WeightsAndBiasesCallback

from data_utils import ProteomeDataset, ToTensor, Normalize_Data, BucketSampler, FractionEmbeddings, PhenotypeInteger, NN_Data
from results_record import Json_Results
from utils import NNUtils, EarlyStopping, Metrics


# wandb might cause an error without this.
os.environ["WANDB_START_METHOD"] = "thread"

CPU_per_Actor = 8
GPU_per_Actor = 1

class Hyper_Optimizer:

    def __init__(self, network, config, study_name, device, NN, group,
                    results_folder, load_study=False, storage=False,
                    min_epochs=2, sampler=optuna.samplers.TPESampler(),
                    pruner=optuna.pruners.SuccessiveHalvingPruner()):
        self.study_name = study_name
        self.study = self.create_study(pruner, sampler, storage, load_study=load_study)
        self.results_folder = results_folder
        self.device = device
        self.num_NN = NN
        self.group = group
        self.wandbc = self.init_wandb(study_name)
        self.parameters = None
        self.min_epochs = min_epochs
        self.network_base = network
        self.config = config


    def create_study(self, pruner, sampler, storage, load_study=False):
        if storage:
            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            storage_module = optuna.storages.JournalStorage(
                                optuna.storages.journal.JournalFileBackend(storage))
            if load_study:
                study = optuna.create_study(direction="maximize", study_name=self.study_name, pruner=pruner,
                                                storage=storage_module, load_if_exists=True)
            else:
                study = optuna.create_study(direction="maximize", study_name=self.study_name, pruner=pruner,
                                                storage=storage_module)
        else:
            if load_study:
                study = joblib.load(load_study)
            else:
                study = optuna.create_study(direction="maximize", study_name=self.study_name, pruner=pruner)
        return study

    def init_wandb(self, name):
        wandb_kwargs = {"project": name, "group":self.group, "reinit": True}
        wandbc = WeightsAndBiasesCallback(
                        metric_name="final validation accuracy",
                        wandb_kwargs=wandb_kwargs, as_multirun=True)
        return wandbc

    def define_parameters(self, trial):
        parameters = {}
        parameters["optimizer"] = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        parameters["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        parameters["n_layers"] =  trial.suggest_int("n_layers", 1, 5)
        parameters["dropout_p"] = trial.suggest_float("dropout_p", 0.2, 0.5)
        parameters["n_hidden"] =  trial.suggest_int("n_hidden", 4, 300)

        return parameters

    def run_optimization(self, n_trials, timeout):

        self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout, callbacks=[self.wandbc],
                                gc_after_trial=True)

    def report_optimization(self):
        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def create_NNSet(self, parameters):
        train_NNs = []
        for n in range(len()):
            
            train_NN1 = self.create_actor(trainloader1, valloader1, parameters)

            train_NNs.append(train_NN1)
        return train_NNs

    def create_actor(self, trainloader, valloader, parameters):
        train_NN = Trainable_NN.remote(trainloader, valloader, self.device)
        train_NN.set_NN.remote(parameters)
        train_NN.set_optimizer.remote(parameters)
        return train_NN

    def create_name(self, parameters):
        name = ""
        for k,v in parameters.items():
            name += "{}_{}".format(k,v)
        return name

    def add_advice(self, advice):
        self.study.enqueue_trial(advice)

    def objective(self, trial):
        # Define Parameters
        parameters = self.define_parameters(trial)
        self.parameters = parameters.keys()
        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb_run = wandb.init(project=self.study_name, name=self.create_name(parameters),# NOTE: this entity depends on your wandb account.
                                group=self.group, config=config,reinit=True)
        # Get the FashionMNIST dataset.
        train_NNs = self.create_NNSet(parameters)
        # Training of the model.
        for epoch in tqdm(range(EPOCHS)):
            acc_loss = ray.get([trains.train_epoch.remote() for trains in train_NNs])
            acc = 0
            for n in acc_loss:
                acc+=n
            acc/=self.num_NN
            wandb_run.log(data={"validation accuracy": acc}, step=epoch)
            trial.report(acc, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune() and epoch < self.min_epochs:
                wandb_run.summary["state"] = "pruned"
                wandb_run.finish(quiet=True)
                for actor in train_NNs:
                    ray.kill(actor)
                raise optuna.exceptions.TrialPruned()
        for actor in train_NNs:
            ray.kill(actor)
        wandb_run.summary["final accuracy"] = acc
        wandb_run.summary["state"] = "complated"
        wandb_run.finish(quiet=True)
        return acc

    def run_save(self, name=None):
        if name is None:
            name = "{}/{}_{}".format(self.results_folder, self.study_name, self.group)
        joblib.dump(self.study, "{}.pkl".format(name))
        with open("{}_sampler.pkl".format(name), "wb") as fout:
            pickle.dump(self.study.sampler, fout)


    def visualization(self):
        wandb_run = wandb.init(project=self.study_name, name="{}_resume".format(self.group),
                                group=self.group, # NOTE: this entity depends on your wandb account.
                                reinit=True)
        fig = optuna.visualization.plot_param_importances(self.study)
        plt.title("Lable Importance")
        wandb.log({"chart": fig})
        fig = optuna.visualization.plot_optimization_history(self.study)
        wandb.log({"chart": fig})
        fig = optuna.visualization.plot_intermediate_values(self.study)
        wandb.log({"chart":fig})
        fig = optuna.visualization.plot_parallel_coordinate(self.study, params=self.parameters)
        wandb.log({"chart":fig})


@ray.remote(num_gpus=GPU_per_Actor, num_cpus=CPU_per_Actor)
class Trainable_NN:

    def __init__(self, network, config, results_dir, mixed_precision, loss_function, memory_report, wandb_report, swa_iter):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, garbage_collection_threshold:0.6'
        torch.cuda.empty_cache()

        self.results_dir = results_dir
        if memory_report:
            self.profiler = self.start_memory_reports()
        else:
            self.profiler = None

        self.device = NNUtils.get_device()

        self.network_base = network
        self.config = config

        self.loss = loss_function

        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.swa_iter = swa_iter

        self.results_training = wandb_report
        self.results_training.start_train_report(model=self.network, criterion=self.loss)

        self.saved_model = {"epoch": None, 'model_state_dict': None, 'optimizer_state_dict': None,
                                'loss': None, "val_measure": None}


    def set_dataloader(self, train_dataset, val_dataset, batch_size, num_workers, stratified, bucketing, asynchonity):
        #  creating dataloaders
        self.train_dataloader = NN_Data.load_data(train_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=stratified)
        self.val_dataloader = NN_Data.load_data(val_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=False)

    def set_NN(self, parameters):

        blocks = [i for i in parameters["block_dims"] if i != 0]

        self.network = self.network_base(input_dim=self.config.model_parameters["input_dim"],
                                    num_classes=self.config.model_parameters["out_dim"],
                                    block_dims=blocks,
                                    stochastic_depth_prob=parameters["stochastic_depth_prob"],
                                    attention_dim=parameters["att_dim"],
                                    attention_norm=self.config.model_parameters["model_structure"]["att_norm"],
                                    dropout_att=self.config.model_parameters["model_structure"]["att_dropout"],
                                    layer_scale=self.config.train_parameters["norm_scale"],
                                    residual_attention=self.config.model_parameters["model_structure"]["residual_attention"],
                                    sequence_dropout=parameters["sequence_dropout"],
                                    norm=self.config.model_parameters["norm"],
                                    length_information=self.config.model_parameters["length_information"],
                                    length_dim=self.config.model_parameters["length_dim"])
        self.network.to(self.device)

    def set_optimizer(self, epochs, steps, optimizer=torch.optim.Adam, learning_rate=1e-5, weight_decay=1e-4,
                        lr_schedule=False, end_lr=3/2, amsgrad=False, fused_OptBack=False, warmup_period=False):
        if fused_OptBack:
            self.optimizer = {p: optimizer([p], foreach=False, lr=learning_rate, weight_decay=weight_decay, amsgrad=amsgrad
                                            ) for p in self.network.parameters()}
            for p in self.network.parameters():
                p.register_post_accumulate_grad_hook(Train_NeuralNet.optimizer_hook)
        else:
            if optimizer.__class__.__name__ == "Adam" or optimizer.__class__.__name__ == "AdamW":
                self.optimizer = optimizer(self.network.parameters(),
                                            lr=learning_rate, weight_decay=weight_decay,
                                            amsgrad=amsgrad)
            else:
                self.optimizer = optimizer(self.network.parameters(),
                                            lr=learning_rate, weight_decay=weight_decay, decoupled_weight_decay=True)
        if lr_schedule:
            self.lr_scheduler = self.set_schedule_lr(optimizer=self.optimizer, end_lr=end_lr,
                                            scheduler_type=lr_schedule,
                                            epochs=epochs, steps=steps, max_lr=learning_rate)
        else:
            self.lr_scheduler = None
        if warmup_period:
            self.warmup = {"scheduler": warmup.LinearWarmup(self.optimizer, warmup_period), "period": warmup_period}
        else:
            self.warmup = None

    def set_schedule_lr(self, scheduler_type, optimizer, epochs, steps, max_lr, end_lr=3/2):
        if scheduler_type == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                    max_lr=max_lr, final_div_factor=end_lr,
                                    epochs=epochs, steps_per_epoch=steps, pct_start=0.2,
                                    )
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=15, min_lr=1e-6)
        else:
            raise ValueError("The scheduler type {} is not available".format(scheduler_type))
        return scheduler

    def train_pass(self, batch_size, profiler=None, asynchronity=False, clipping=False):

        self.network.train()

        loss_lst = []
        lr_rate_lst = []
        loss_pass = 0.
        count = 0
        len_dataloader = len(self.train_dataloader)
        labels_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device, dtype=int)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device)
        batch_n = 0

        for batch in tqdm(self.train_loader):
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
            if self.lr_scheduler is not None and self.lr_scheduler.__class__.__name__ == "OneCycleLR":
                self.update_scheduler()
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
        return loss_pass, mcc_pass, lr_rate_lst, profiler

    def val_pass(self, batch_size, profiler=None, asynchronity=False):

        self.network.eval()

        loss_lst = []

        loss_pass = 0.
        count = 0
        len_dataloader = len(self.val_dataloader)
        labels_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device, dtype=int)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device)
        batch_n = 0

        with torch.inference_mode():
            for batch in tqdm(self.val_dataloader):
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
#                predictions_logit, attentions = self.network(embeddings, lengths)
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


