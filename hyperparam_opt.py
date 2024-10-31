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
import numpy as np
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

    def __init__(self, network, config, study_name, group, dual_pred, memory_report,
                    results_folder, loss_function, mixed_precision, load_study=False, storage=False,
                    min_epochs_prune=4, min_epochs_count=2, sampler=optuna.samplers.TPESampler(),
                    pruner=optuna.pruners.SuccessiveHalvingPruner()):

        assert min_epochs_prune > min_epochs_count
        self.study_name = study_name
        self.study = self.create_study(pruner, sampler, storage, load_study=load_study)
        self.results_folder = results_folder
        self.device = NNUtils.get_device()
        self.num_NN = len(config.hyperopt_parameters["train_df"])
        self.group = group
        self.dual_pred = dual_pred
        self.wandbc = self.init_wandb(study_name)
        self.parameters = None
        self.min_epochs_prune = min_epochs_prune
        self.min_epochs_count = min_epochs_count
        self.network_base = network
        self.config = config
        self.loss_function = loss_function
        self.mixed_precision = mixed_precision
        self.memory_report = memory_report


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

    def define_parameters(self, trial, trial_parameters):
        parameters = {}
        for key, val in trial_parameters.items():
            parameters[key] = trial.suggest_categorical(key, val)
        return parameters

    def define_parameters_old(self, trial):
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
        for n in range(self.num_NN):
            traindf = self.config.hyperopt_parameters["train_df"][n]
            trainloc = self.config.hyperopt_parameters["train_loc"][n]
            valdf = self.config.hyperopt_parameters["val_df"][n]
            valloc = self.config.hyperopt_parameters["val_loc"][n]
            train_NN1 = self.create_actor(traindf=traindf, valdf=valdf, trainloc=trainloc,
                                            valloc=valloc, parameters=parameters)
            train_NNs.append(train_NN1)
        return train_NNs

    def create_actor(self, traindf, valdf, trainloc, valloc, parameters):
        train_NN = Trainable_NN.remote(network=self.network_base, config=self.config,
                                        results_dir=self.results_folder, mixed_precision=self.mixed_precision,
                                        loss_function=self.loss_function, memory_report=self.memory_report,
                                        )
        train_NN.create_dataloader.remote(traindf=traindf, trainloc=trainloc, valdf=valdf, valloc=valloc,
                                            parameters=parameters, dual_pred=self.dual_pred)
        train_NN.set_NN.remote(parameters)
        train_NN.set_optimizer.remote(epochs=self.config.train_parameters["epochs"],
                                optimizer=self.config.train_parameters["optimizer"],
                                learning_rate=parameters["learning_rate"],
                                weight_decay=self.config.train_parameters["weight_decay"],
                                lr_schedule=self.config.train_parameters["lr_scheduler"],
                                end_lr=self.config.train_parameters["lr_end"],
                                amsgrad=False, fused_OptBack=False, warmup_period=self.config.train_parameters["warm_up"])
        return train_NN

    def create_name(self, parameters):
        name = ""
        for k,v in parameters.items():
            name += "{}_{}".format(k,v)
        return name

    def add_advice(self, advice):
        self.study.enqueue_trial(advice)

    def record_data(self, wandb_run, results_array):
        wandb_run.summary["final accuracy"] = results_array[1,-1]
        wandb_run.summary["final epoch"] = results_array[0, -1]
        wandb_run.summary["best accuracy"] = max(results_array[1,:])
        wandb_run.summary["best epoch"] = results_array[0, np.argmax(results_array[1,:])]
        last15 = np.sort(results_array[1,-15:])[::-1]
        wandb_run.summary["mean (5) best 15 last epochs"] = np.mean(last15[0:5])
        wandb_run.summary["state"] = "complated"
        wandb_run.finish(quiet=True)

    def objective(self, trial):
        # Define Parameters
        parameters = self.define_parameters(trial=trial,
                                            trial_parameters=self.config.hyperopt_parameters["optimizing_parameters"])
        self.parameters = parameters.keys()
        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb_run = wandb.init(project=self.study_name, name=self.create_name(parameters),# NOTE: this entity depends on your wandb account.
                                group=self.group, config=config,reinit=True)
        # Get the FashionMNIST dataset.
        train_NNs = self.create_NNSet(parameters)
        max_mcc_value = 0
        results_mcc = []
        # Training of the model.
        for epoch in tqdm(range(self.config.train_parameters["epochs"])):
            print("epoch starts")
            acc_mccs = ray.get([trains.train_epoch.remote(self.config.train_parameters["asynchronity"]) for trains in train_NNs])
            print("epoch finishes")
            acc = 0
            for n in range(self.num_NN):
                val_mcc = float(acc_mccs[n]["val_mcc"])
                lr = float(acc_mccs[n]["learning_rate"])
                wandb_run.log(data={"validation accuracy NN{}".format(n):val_mcc,
                                    "learning rate NN{}".format(n):lr,
                                    "epoch": epoch}, step=epoch)
#                wandb_run.log(data={"learning rate NN{}".format(n):lr}, step=epoch)
                acc += val_mcc
            acc /= self.num_NN
            results_mcc.append([epoch, acc])
            if epoch >= self.min_epochs_count:
                if max_mcc_value < acc:
                    max_mcc_value = acc
                trial.report(max_mcc_value, epoch)
            else:
                trial.report(acc, epoch)

            wandb_run.log(data={"validation accuracy": acc, "max validation accuracy": max_mcc_value, "epoch":epoch}, step=epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune() and epoch < self.min_epochs_prune:
                wandb_run.summary["state"] = "pruned"
                wandb_run.finish(quiet=True)
                for actor in train_NNs:
                    ray.kill(actor)
                raise optuna.exceptions.TrialPruned()
        for actor in train_NNs:
            ray.kill(actor)
        self.record_data(wandb_run, np.array(results_mcc))
        return acc

    def run_save(self, name=None):
        if name is None:
            name = "{}/{}_Group{}".format(self.results_folder, self.study_name, self.group)
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
    MAX_BATCH_SIZE = 45
    MIN_BATCH_SIZE = 8

    def __init__(self, network, config, results_dir, mixed_precision, loss_function, memory_report):
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
        self.loss_function = loss_function()

        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.saved_model = {"epoch": None, 'model_state_dict': None, 'optimizer_state_dict': None,
                                'loss': None, "val_measure": None}

    def set_sensible_batch_size(self, batch_size):
        if batch_size < Trainable_NN.MIN_BATCH_SIZE:
            raise ValueError("The batch size {} is smaller than the minimum allowed ({})".format(
                                                                batch_size, Trainable_NN.MIN_BATCH_SIZE))

        min_batch = 0 + Trainable_NN.MIN_BATCH_SIZE
        accumulated_gradient = batch_size / min_batch
        pre_min_batch = min_batch
        pre_accumul = accumulated_gradient
        while True:
            if pre_min_batch > Trainable_NN.MIN_BATCH_SIZE:
                break
            elif not pre_accumul.is_integer():
                pass
            elif pre_min_batch == Trainable_NN.MIN_BATCH_SIZE:
                accumulated_gradient = pre_accumul
                min_batch = pre_min_batch
                break
            else:
                accumulated_gradient = pre_accumul
                min_batch = pre_min_batch
            pre_min_batch += Trainable_NN.MIN_BATCH_SIZE
            pre_accumul = batch_size / pre_min_batch
        if accumulated_gradient is None or not accumulated_gradient.is_integer():
            raise ValueError("Batch size {} is not a multiple of 8".format(batch_size))
        return min_batch, accumulated_gradient


    def create_dataloader(self, traindf, trainloc, valdf, valloc, parameters, dual_pred):
        batch_size = self.config.model_parameters["batch_size"]
        if batch_size > Trainable_NN.MAX_BATCH_SIZE:
            batch_size, accumulate_gradient = self.set_sensible_batch_size(batch_size)
        else:
            accumulate_gradient = False

        self.batch_size = batch_size
        self.accumulate_gradient = accumulate_gradient

        # Create Train data
        train_dataset = NN_Data.create_dataset(input_type=self.config.model_parameters["input_type"],
                            data_df=traindf, data_loc=trainloc, data_type="train",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred,
                            weighted=self.config.train_parameters["imbalance_weight"],
                            normalize=self.config.train_parameters["normalize"],
                            fraction_embeddings=self.config.train_parameters["prot_dim_split"])
        # Create Val data
        val_dataset = NN_Data.create_dataset(input_type=self.config.model_parameters["input_type"],
                            data_df=valdf, data_loc=valloc, data_type="prediction",
                            cluster_sample=self.config.train_parameters["data_sample"],
                            cluster_tsv=self.config.train_parameters["cluster_tsv"],
                            dual_pred=dual_pred, normalize=self.config.train_parameters["normalize"],
                            fraction_embeddings=self.config.train_parameters["prot_dim_split"])
        self.train_dataloader = NN_Data.load_data(train_dataset, batch_size,
                                        num_workers=self.config.train_parameters["num_workers"],
                                        shuffle=True, pin_memory=self.config.train_parameters["asynchronity"],
                                        bucketing=self.config.train_parameters["bucketing"],
                                        stratified=self.config.train_parameters["stratified"])
        self.val_dataloader = NN_Data.load_data(val_dataset, batch_size,
                                        num_workers=self.config.train_parameters["num_workers"],
                                        shuffle=True, pin_memory=self.config.train_parameters["asynchronity"],
                                        bucketing=self.config.train_parameters["bucketing"], stratified=False)
        self.accumulate_gradient = accumulate_gradient
        


    def set_data(self, trainloader, valloader):
        self.train_dataloader = trainloader
        self.val_dataloader = valloader

    def set_blocks_old(self, parameters):
        blocks = []
        count = 1
        while True:
            try:
                value_block = parameters["block_dim{}".format(count)]
            except KeyError:
                break
            if value_block != 0:
                blocks.append(value_block)
            count += 1
        return blocks

    def set_blocks(self, parameters):
        list_blocks = [parameters["block_dims"]] * (parameters["num_blocks"]-1)
        list_blocks.append(int(parameters["block_dims"]/2))
        return list_blocks

    def set_NN(self, parameters):

        blocks = self.set_blocks(parameters)
        print(blocks)

        self.network = self.network_base(input_dim=self.config.model_parameters["input_dim"],
                                    num_classes=self.config.model_parameters["out_dim"],
                                    block_dims=blocks,
                                    stochastic_depth_prob=parameters["stochastic_depth_prob"],
                                    attention_dim=parameters["att_dim"],
                                    attention_norm=self.config.model_parameters["model_structure"]["att_norm"],
                                    dropout_att=parameters["att_dropout"],
                                    layer_scale=self.config.train_parameters["norm_scale"],
                                    residual_attention=self.config.model_parameters["model_structure"]["residual_attention"],
                                    sequence_dropout=parameters["sequence_dropout"],
                                    norm=self.config.model_parameters["norm"],
                                    length_information=self.config.model_parameters["length_information"],
                                    length_dim=self.config.model_parameters["length_dim"])
        self.network.to(self.device)

    def set_optimizer(self, epochs, optimizer=torch.optim.Adam, learning_rate=1e-5, weight_decay=1e-4,
                        lr_schedule=False, end_lr=3/2, amsgrad=False, fused_OptBack=False, warmup_period=False):
        steps = len(self.train_dataloader)
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

    def train_epoch(self, asynchronity):
        batch_size = self.batch_size
        loss_tr, mcc_tr, lr_rate_lst, _ = self.train_pass(batch_size=batch_size, asynchronity=asynchronity)
        loss_val, mcc_val, _ = self.val_pass(batch_size=batch_size, asynchronity=asynchronity)
        if self.lr_scheduler is not None:
            if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.update_scheduler(value=loss_val)
            elif self.lr_scheduler.__class__.__name__ == "MultiStepLR":
                self.update_scheduler()
        return {"val_mcc": mcc_val, "learning_rate": np.mean(lr_rate_lst)}

    def calculate_loss(self, predictions_logit, labels):
        predictions = torch.sigmoid(predictions_logit)
        if self.loss == torch.nn.modules.loss.BCELoss:
            loss = self.loss_function(predictions, labels)
        elif self.loss == torch.nn.modules.loss.BCEWithLogitsLoss:
            loss = self.loss_function(predictions_logit, labels)
        else:
            raise KeyError("The loss function {} is not available".format(self.loss))
        return predictions, loss

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

    def train_pass(self, batch_size, accumulate_gradient=False, profiler=None, asynchronity=False, clipping=False):

        self.network.train()
        train_loader = self.train_dataloader

        loss_lst = []
        lr_rate_lst = []
        loss_pass = 0.
        count = 0
        len_dataloader = len(train_loader)
        labels_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device, dtype=int)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device)
        batch_n = 0

        self.optimizer.zero_grad(set_to_none=True)

        for idx, batch in tqdm(enumerate(train_loader)):
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
        #    self.optimizer.zero_grad(set_to_none=True)
            #  making predictions
            with torch.autocast(device_type=self.device, enabled=self.mixed_precision):
                predictions_logit, attentions = self.network(embeddings, lengths)
                predictions, loss = self.calculate_loss(
                                        predictions_logit=predictions_logit, labels=labels)

            if self.accumulate_gradient:
                loss = loss/self.accumulate_gradient
            #  computing gradients
            self.scaler.scale(loss).backward()
            if clipping:
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), clipping)
            if not self.accumulate_gradient or ((idx + 1) % self.accumulate_gradient == 0) or (idx + 1 == len(train_loader)):
                #  updating weights
                if isinstance(self.optimizer, dict):
                    pass
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            lr_rate_lst.append(self.optimizer.param_groups[-1]['lr'])
        #   self.scaler.update()
            if self.lr_scheduler is not None and self.lr_scheduler.__class__.__name__ == "OneCycleLR":
                self.update_scheduler()
            #  computing loss
            loss_c = loss.detach()
            pred_c = predictions.detach()
            labels = labels.detach()
            labels_tensor[pos_first:pos_last,:] = labels
            pred_tensor[pos_first:pos_last,:] = pred_c

            loss_pass += loss_c
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
        val_loader = self.val_dataloader

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


