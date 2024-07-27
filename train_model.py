import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
import pytorch_warmup as warmup
from tqdm import tqdm
import os
import numpy as np
import sys
import time
from torchvision import transforms
from torch.optim import swa_utils

sys.dont_write_bytecode = True

from data_utils import ProteomeDataset, ToTensor, Normalize_Data, BucketSampler, FractionEmbeddings, PhenotypeInteger
from results_record import Json_Results
from utils import NNUtils


class Train_NeuralNet():
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000

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
        self.train_dataset = None
        self.val_dataset = None

        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)


        self.results_training = Json_Results(wandb_results=wandb_report, configuration=configuration,
                                           name=os.path.basename(self.results_dir), model=self.network,
                                           criterion=self.loss)
        self.saved_model = {"epoch": None, 'model_state_dict': None, 'optimizer_state_dict': None,
				'loss': None, "val_measure": None}

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
                                    epochs=epochs, steps_per_epoch=steps,
                                    )
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, min_lr=1e-6)
        else:
            raise ValueError("The scheduler type {} is not available".format(scheduler_type))
        return scheduler

    def start_memory_reports(self):
        memory_report = "{}/memory_report".format(self.results_dir)
        torch.cuda.memory._record_memory_history(
            max_entries=Train_NeuralNet.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)
        prof = torch.profiler.profile(
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler("{}".format(memory_report)),
                        record_shapes=True, with_stack=False, profile_memory=True)
        prof.start()
        return prof

    def stop_memory_reports(self):
        if self.profiler is not None:
            self.profiler.stop()
            torch.cuda.memory._dump_snapshot("{}/memory_record.pkl".format(self.results_dir))
            torch.cuda.memory._record_memory_history(enabled=None)


    def create_dataset(self, data_df, data_loc, data_type="train", dual_pred=False, cluster_sample=False,
                        cluster_tsv=None, weighted=False, normalize=False, fraction_embeddings=False):
        transform_data = []

        if normalize:
            transform_data.append(Normalize_Data(normalize))
        else:
            pass

        if fraction_embeddings:
            transform_data.append(FractionEmbeddings(fraction_embeddings))
        else:
            pass

        if dual_pred:
            transform_data.append(PhenotypeInteger(prediction="Dual"))
        else:
            transform_data.append(PhenotypeInteger(prediction="Single"))

        transform_compose = transforms.Compose(transform_data)

        if data_type == "validation":
            dataset = ProteomeDataset(csv_file=data_df, root_dir=data_loc, transform=transform_compose)
            self.val_dataset = dataset
        elif data_type == "train":
            if not cluster_sample:
                dataset = ProteomeDataset(csv_file=data_df, root_dir=data_loc,
                                        transform=transform_compose, weighted=weighted)
            else:
                dataset = ProteomeDataset(csv_file=data_df, root_dir=data_loc,
                            transform=transform_compose, cluster_sampling=cluster_sample,
                            cluster_tsv=cluster_tsv, weighted=weighted)
            self.train_dataset = dataset
        else:
            raise ValueError("The data_type {} is not an option (choose between train and val)".format(
                                data_type))

    def load_data(self, data_set, batch_size, num_workers=4, shuffle=True, pin_memory=False,
                  bucketing=None, stratified=False):
        if bucketing:
            bucketing_sampler = BucketSampler(data_set.landmarks_frame, batch_size=batch_size,
                                              num_buckets=bucketing, stratified=stratified, drop_last=True)
            data_loader = DataLoader(data_set, num_workers=num_workers,
                              collate_fn=ProteomeDataset.collate_fn_mask, batch_sampler=bucketing_sampler,
                              persistent_workers=False, pin_memory=pin_memory)
        else:
            data_loader = DataLoader(data_set, batch_size=batch_size, num_workers=num_workers,
                              collate_fn=ProteomeDataset.collate_fn_mask, drop_last=True,
                              shuffle=shuffle, persistent_workers=False, pin_memory=pin_memory)

        return data_loader

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

    @staticmethod
    def optimizer_hook(parameter):
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()
    
    def train_pass(self, train_loader, profiler=None, asynchronity=False, clipping=False):

        loss_lst = []
        mcc_lst = []
        lr_rate_lst = []
        loss_pass = 0.
        mcc_pass = 0.
        count = 0
        len_dataloader = len(train_loader) 
        for batch in tqdm(train_loader):
            if self.profiler is not None:
                self.profiler.step()
            embeddings = batch["Embeddings"]
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
            if torch.isnan(loss_c):
                print("FILE WRONG: ", batch["File_Names"])
            pred_c = predictions.detach()
            label_c = labels.detach()
            loss_pass += loss_c
            mcc = Json_Results.calculate_metrics_GPU(labels=label_c, predictions=pred_c)
            mcc_pass += mcc
            self.results_training.step_log(loss_train=loss_c, lr=self.optimizer.param_groups[-1]['lr'], batch_n=count,
                                             len_dataloader=len_dataloader)
            #  clean gpu (maybe unnecessary)
            count += 1
        loss_pass = loss_pass/count
        mcc_pass = mcc_pass/count
        return loss_pass, mcc_pass, lr_rate_lst, profiler

    def val_pass(self, val_loader, last_epoch=False, profiler=None, asynchronity=False):

        loss_lst = []
        mcc_lst = []
        
        loss_pass = 0.
        mcc_pass = 0.
        count = 0

        if last_epoch:
            att_results = {"Genomes": [], "Proteins":[], "Attentions": []}
        else:
            att_results = None

        with torch.inference_mode():
            for batch in tqdm(val_loader):
                if self.profiler is not None:
                    self.profiler.step()
                embeddings = batch["Embeddings"]
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
                label_c = labels.detach()
                loss_pass += loss_c
                if torch.isnan(loss_c):
                    print("FILE WRONG: ", batch["File_Names"])
                mcc = Json_Results.calculate_metrics_GPU(labels=label_c, predictions=pred_c)
                mcc_pass += mcc

                if last_epoch:
                    genome_names = batch["File_Names"]
                    att_results["Genomes"].extend(genome_names)
                    prot_names = batch["Protein_IDs"]
                    att_results["Proteins"].extend(prot_names)
                    attentions = attentions.detach().cpu().numpy()
                    att_results["Attentions"].append(attentions)
                #  clean gpu (maybe unnecessary
                count += 1
        loss_pass = loss_pass/count
        mcc_pass = loss_pass/count
        if last_epoch:
            return loss_lst, mcc_lst, att_results, profiler
        else:
            return loss_pass, mcc_pass, profiler

    def best_epoch_retain(self, new_val, optimizer, model, epoch, loss):
        if self.saved_model["val_measure"] is not None and self.saved_model["val_measure"] > new_val:
            return {"epoch": epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss, "val_measure": new_val}
        else:
            return self.saved_model

    def early_stopping(self):
        pass
        

    def train(self, epochs, batch_size, optimizer=torch.optim.Adam, learning_rate=1e-5, weight_decay=1e-4,
            lr_schedule=False, end_lr=3/2, amsgrad=False, num_workers=2, asynchronity=False, stratified=False,
            fused_OptBack=False, clipping=False, bucketing=False, warmup_period=False, stop_method="best_epoch"):


        pos_weight = self.train_dataset.get_weights()

        self.loss_function = self.loss()

        #  creating dataloaders
        train_loader = self.load_data(self.train_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=stratified)
        val_loader = self.load_data(self.val_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing)

        steps = steps=len(train_loader)
        self.set_optimizer(epochs=epochs, steps=steps, optimizer=optimizer, learning_rate=learning_rate, 
                weight_decay=weight_decay, lr_schedule=lr_schedule, end_lr=end_lr, amsgrad=amsgrad, fused_OptBack=fused_OptBack,
                warmup_period=warmup_period)

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}') 
            start_e_time = time.time()
            #  training
            loss_train, mcc_t, lr_rate, profiler = self.train_pass(train_loader=train_loader, clipping=clipping)

            #  validation
            print('validating...')
            loss_val, mcc_v, profiler = self.val_pass(val_loader=val_loader)

            self.results_training.add_epoch_report(epoch=epoch, loss_t=loss_train, loss_v=loss_val, 
                                                   lr=lr_rate, mcc_t=mcc_t, mcc_v=mcc_v)
            if stop_method == "early_stopping":
                pass
            elif stop_method == "best_epoch":
                self.saved_model = self.best_epoch_retain(new_val=mcc_v, optimizer=self.optimizer, model=self.network, epoch=epoch, loss=loss_train)
            else:
                pass

            if self.lr_scheduler is not None and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.update_scheduler(value=loss_val)
            end_e_time = time.time()
            self.results_training.time_log(end_e_time-start_e_time)
            print("training_loss: {}, validation_loss: {}".format(loss_train, loss_val))
            
        if self.profiler is not None:
            self.stop_memory_reports()
        return self.saved_model

    def swa_pass(self, state_dict_model, optimizer, swa_iter, train_loader):
        self.network = self.network.load_state_dict(state_dict)
        swa_model = swa_utils.AveragedModel(self.network)
        swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=0.05)
        for swa_epoch in range(swa_iter):
            loss_train, prediction_train, labels_train, lr_rate, profiler = self.train_pass(train_loader=train_loader,
										              clipping=False)
            swa_model.update_parameters(model)
            swa_scheduler.step()

        swa_utils.update_bn(train_loader, swa_model)
        return swa_model
   
    def save_model(self, model, type_save="state_dict"):
        path = "{}/model.pth".format(self.results_dir)
        if type_save == "state_dict":
            torch.save(self.saved_model["model_state_dict"], path)
        elif type_save == "checkpoint":
            torch.save(self.saved_model, path)
        else:
            torch.save(self.network, path)

    def load_model(self, state_dict):
        return self.network.load_state_dict(state_dict)

    def get_network(self):
        return self.network.cpu()

    def predict(self, x):
        with torch.inference_mode():
            logits = self.network(x)
        return torch.sigmoid(logits)

