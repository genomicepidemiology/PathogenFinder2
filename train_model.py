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

from data_utils import ProteomeDataset, ToTensor, Normalize_Data, BucketSampler, FractionEmbeddings, PhenotypeInteger, NN_Data
from results_record import Json_Results
from utils import NNUtils, EarlyStopping, Metrics


class Train_NeuralNet():
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000
    MAX_BATCH_SIZE = 64
    MIN_BATCH_SIZE = 8

    def __init__(self, network, configuration=None, loss_function=None, results_dir=None, memory_report=False, 
		compiler=False, mixed_precision=False, results_module=None, wandb_report=False, swa_iter=False):

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

        self.swa_iter = swa_iter


        self.results_training = wandb_report
        self.results_training.start_train_report(model=self.network, criterion=self.loss)

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
                                    epochs=epochs, steps_per_epoch=steps, pct_start=0.2,
                                    )
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=20)
        elif scheduler_type == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 55, 70, 80, 90, 95], gamma=0.5)
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
    
    def train_pass(self, train_loader, batch_size, accumulate_gradient=False, profiler=None, asynchronity=False, clipping=False):
        
        self.network.train()
        
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

            if accumulate_gradient:
                loss = loss/accumulate_gradient
            #  computing gradients
            self.scaler.scale(loss).backward()
            if clipping:
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), clipping)
            if not accumulate_gradient or ((idx + 1) % accumulate_gradient == 0) or (idx + 1 == len(dataloader)):
                #  updating weights
                if isinstance(self.optimizer, dict):
                    pass
                else:
                    print("UPDATED WEIGHTS", idx)
                    self.scaler.step(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)
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

    def best_epoch_retain(self, new_val, optimizer, model, epoch, loss):
        if self.saved_model["val_measure"] is None or self.saved_model["val_measure"] < new_val:
            print("SAVED")
            return {"epoch": epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss, "val_measure": new_val}
        else:
            return self.saved_model


    def set_sensible_batch_size(self, batch_size):
        if batch_size < Train_NeuralNet.MIN_BATCH_SIZE:
            raise ValueError("The batch size {} is smaller than the minimum allowed ({})".format(
                                                                batch_size, Train_NeuralNet.MIN_BATCH_SIZE))

        min_batch = 0 + Train_NeuralNet.MIN_BATCH_SIZE
        accumulated_gradient = batch_size / min_batch
        pre_min_batch = min_batch
        pre_accumul = accumulated_gradient
        while True:
            if pre_min_batch > Train_NeuralNet.MAX_BATCH_SIZE:
                break
            elif not pre_accumul.is_integer():
                pass
            elif pre_min_batch == Train_NeuralNet.MAX_BATCH_SIZE:
                accumulated_gradient = pre_accumul
                min_batch = pre_min_batch
                break
            else:
                accumulated_gradient = pre_accumul
                min_batch = pre_min_batch
            pre_min_batch += Train_NeuralNet.MIN_BATCH_SIZE
            pre_accumul = batch_size / pre_min_batch
        if accumulated_gradient is None or not accumulated_gradient.is_integer():
            raise ValueError("Batch size {} is not a multiple of 8".format(batch_size))
        return min_batch, accumulated_gradient

                   
       

    def __call__(self, train_dataset, val_dataset, epochs, batch_size, optimizer=torch.optim.Adam, learning_rate=1e-5, weight_decay=1e-4,
            lr_schedule=False, end_lr=3/2, amsgrad=False, num_workers=2, asynchronity=False, stratified=False,
            fused_OptBack=False, clipping=False, bucketing=False, warmup_period=False, early_stopping=True, keep_model=False, evaluate_train=True):


        pos_weight = train_dataset.get_weights()

        self.loss_function = self.loss()

        if batch_size > Train_NeuralNet.MAX_BATCH_SIZE:
            batch_size, accumulate_gradient = self.set_sensible_batch_size(batch_size)
        else:
            accumulate_gradient = False

        #  creating dataloaders
        train_loader = NN_Data.load_data(train_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=stratified)
        val_loader = NN_Data.load_data(val_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=False)
        if evaluate_train:
            train_loader_eval = NN_Data.load_data(train_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing, stratified=False)


        steps = len(train_loader)
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
            if evaluate_train:
                loss_train_eval, mcc_train_eval, _ = self.val_pass(val_loader=train_loader_eval, asynchronity=asynchronity, batch_size=batch_size)
            else:
                loss_train_eval, mcc_train_eval = None, None

            self.results_training.add_epoch_info(epoch=epoch, loss_t=loss_train, loss_v=loss_val, 
                                                   mcc_t=mcc_t, mcc_v=mcc_v, loss_t_eval=loss_train_eval,
                                                   mcc_t_eval=mcc_train_eval)

            if keep_model == "best_epoch":
                self.saved_model = self.best_epoch_retain(new_val=mcc_v, optimizer=self.optimizer, model=self.network, epoch=epoch, loss=loss_train)
            print(self.lr_scheduler.__class__.__name__, self.optimizer.param_groups[-1]['lr'])
            if self.lr_scheduler is not None:
                if self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.update_scheduler(value=loss_val)
                elif self.lr_scheduler.__class__.__name__ == "MultiStepLR":
                    self.update_scheduler()
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
        return self.saved_model

    def swa_pass(self, state_dict_model, optimizer, swa_iter, train_loader, val_loader, asynchronity, batch_size):
        self.network.load_state_dict(state_dict_model)
        swa_model = swa_utils.AveragedModel(self.network)
        swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=0.05)
        for swa_epoch in range(swa_iter):
            loss_train, mcc_t, lr_rate, profiler = self.train_pass(train_loader=train_loader, batch_size=batch_size,
                                    clipping=False, asynchronity=asynchronity)
            loss_val, mcc_v, profiler = self.val_pass(val_loader=val_loader, asynchronity=asynchronity, batch_size=batch_size)

            swa_model.update_parameters(swa_model)
            swa_scheduler.step()

        swa_utils.update_bn(train_loader, swa_model)
        return swa_model, loss_train, mcc_v
   
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

