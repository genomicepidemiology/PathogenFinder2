import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR,SequentialLR, OneCycleLR
from tqdm import tqdm
from torchmetrics.classification import BinaryMatthewsCorrCoef
import h5py
import pickle
import gc
import os
import numpy as np
import sys
from datetime import datetime
from torchvision import transforms

sys.dont_write_bytecode = True

from data_utils import ProteomeDataset, ToTensor, Normalize_Data, BucketSampler, FractionEmbeddings, PhenotypeInteger


class Train_NeuralNet():
    MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000

    def __init__(self, network, optimizer=torch.optim.Adam, results_train="dictionary", learning_rate=1e-5,
                weight_decay=1e-4, amsgrad=False, loss_function=None, results_dir=None,
                memory_report=False, train_results="dictionary", compiler=False):

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, garbage_collection_threshold:0.6' 
        torch.cuda.empty_cache()

        self.results_dir = self.set_results_files(results_dir=results_dir)
        if memory_report:
            self.profiler = self.start_memory_reports()
        else:
            self.profiler = None


        self.device = self.get_device()
        print("Training on {}".format(self.device))
        network = network.to(self.device)
        if not compiler:
            self.network = network
        else:
            self.network = torch.compile(network, mode=compiler, dynamic=True)
        self.loss = loss_function
        self.train_dataset = None
        self.val_dataset = None

    def set_optimizer(self, epochs, steps, optimizer=torch.optim.Adam, learning_rate=1e-5, weight_decay=1e-4,
                        lr_schedule=False, end_lr=3/2, amsgrad=False, fused_OptBack=False):
        if fused_OptBack:
            self.optimizer = {p: optimizer([p], foreach=False, lr=learning_rate, weight_decay=weight_decay, amsgrad=amsgrad
                                            ) for p in self.network.parameters()}
            for p in self.network.parameters():
                p.register_post_accumulate_grad_hook(Train_NeuralNet.optimizer_hook)
        else:
            self.optimizer = optimizer(self.network.parameters(),
                                            lr=learning_rate, weight_decay=weight_decay,
                                            amsgrad=amsgrad)
        if lr_schedule:
            self.lr_scheduler = self.set_schedule_lr(optimizer=self.optimizer, end_lr=end_lr,
                                            scheduler_type=lr_schedule,
                                            epochs=epochs, steps=steps, max_lr=learning_rate)
        else:
            self.lr_scheduler = None


    def set_schedule_lr(self, scheduler_type, optimizer, epochs, steps, max_lr, end_lr=3/2):
        if scheduler_type == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                    max_lr=max_lr, final_div_factor=end_lr,
                                    epochs=epochs, steps_per_epoch=steps,
                                    )
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
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


    def set_results_files(self, results_dir):

        timestamp = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

        results_dir = "{}_{}".format(results_dir, timestamp)
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        return results_dir

    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

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
                  bucketing=None):
        if bucketing:
            bucketing_sampler = BucketSampler(data_set.landmarks_frame, batch_size=batch_size,
                                              num_buckets=bucketing)
            data_loader = DataLoader(data_set, num_workers=num_workers,
                              collate_fn=ProteomeDataset.collate_fn_mask, batch_sampler=bucketing_sampler,
                              persistent_workers=False, pin_memory=pin_memory)
        else:
            data_loader = DataLoader(data_set, batch_size=batch_size, num_workers=num_workers,
                              collate_fn=ProteomeDataset.collate_fn_mask,
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

    @staticmethod
    def optimizer_hook(parameter):
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad()
    
    def train_pass(self, train_loader, scaler, mixed_precision=False, profiler=None, asynchronity=False, clipping=False):
        mcc_calc = BinaryMatthewsCorrCoef()
        loss_lst = []
        mcc_lst = []
        prediction_lst = []
        labels_lst = []
        lr_rate_lst = []
        genome_names = []
        for batch in tqdm(train_loader):
            if self.profiler is not None:
                self.profiler.step()
            embeddings = batch["Embeddings"]
            print(embeddings.shape)
            labels = batch["PathoPhenotype"]
            lengths = batch["Proteome_Length"]
            #  sending data to device
            embeddings = embeddings.to(self.device, non_blocking=asynchronity)
            labels = labels.to(self.device, non_blocking=asynchronity)
            lengths = lengths.to(self.device, non_blocking=asynchronity)
            #  resetting gradients
            self.optimizer.zero_grad(set_to_none=True)
            #  making predictions
            with torch.autocast(device_type=self.device, enabled=mixed_precision):
                predictions_logit, attentions = self.network(embeddings, lengths)
                predictions, loss = self.calculate_loss(
                                        predictions_logit=predictions_logit, labels=labels)
            #  computing gradients
            scaler.scale(loss).backward()
            if clipping:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(self.optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), clipping)
            #  updating weights
            if isinstance(self.optimizer, dict):
                pass
            else:
                scaler.step(self.optimizer)
            lr_rate_lst.append(self.optimizer.param_groups[-1]['lr'])
            scaler.update()
            if self.lr_scheduler is not None and self.lr_scheduler.__class__.__name__ == "OneCycleLR":
                self.lr_scheduler.step()          
            #  computing loss
            loss_c = loss.detach().cpu().tolist()
            pred_c = predictions.detach().cpu()
            label_c = labels.detach().cpu()
            genome_names.append(batch["File_Names"])
            mcc_c = mcc_calc(pred_c, label_c)
            loss_lst.append(loss_c)
            mcc_lst.append(mcc_c)
            prediction_lst.extend(pred_c.tolist())
            labels_lst.extend(label_c.tolist())
            #  clean gpu (maybe unnecessary)
            print("Train Loss step: {} // Train MCC step: {}".format(loss_c, mcc_c))
        return loss_lst, mcc_lst, prediction_lst, labels_lst, lr_rate_lst, profiler, genome_names

    def val_pass(self, val_loader, last_epoch=False, profiler=None, asynchronity=False):
        mcc_calc = BinaryMatthewsCorrCoef()
        loss_lst = []
        mcc_lst = []
        prediction_lst = []
        labels_lst = []
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
                lengths = batch["Proteome_Length"]
                #  sending data to device
                embeddings = embeddings.to(self.device, non_blocking=asynchronity)
                labels = labels.to(self.device, non_blocking=asynchronity)
                lengths = lengths.to(self.device, non_blocking=asynchronity)
                #  making predictions
                predictions_logit, attentions = self.network(embeddings, lengths)
                predictions, loss = self.calculate_loss(
                                        predictions_logit=predictions_logit, labels=labels)
                #  computing loss
                loss_c = loss.detach().cpu().tolist()
                pred_c = predictions.detach().cpu()
                label_c = labels.detach().cpu()
                mcc_c = mcc_calc(pred_c, label_c)
                loss_lst.append(loss_c)
                mcc_lst.append(mcc_c)
                prediction_lst.extend(pred_c.tolist())
                labels_lst.extend(label_c.tolist())
                if last_epoch:
                    genome_names = batch["File_Names"]
                    att_results["Genomes"].extend(genome_names)
                    prot_names = batch["Protein_IDs"]
                    att_results["Proteins"].append(prot_names)
                    attentions = attentions.detach().cpu().numpy()
                    att_results["Attentions"].append(attentions)
                #  clean gpu (maybe unnecessary
                print("Val Loss step: {} // Val MCC step: {}".format(loss_c, mcc_c))
        if last_epoch:
            return loss_lst, mcc_lst, prediction_lst, labels_lst, att_results, profiler
        else:
            return loss_lst, mcc_lst, prediction_lst, labels_lst, profiler       

    def train(self, epochs, batch_size, optimizer=torch.optim.Adam, learning_rate=1e-5, weight_decay=1e-4,
            lr_schedule=False, end_lr=3/2, amsgrad=False, mixed_precision=False, num_workers=2, asynchronity=False,
            fused_OptBack=False, clipping=False, bucketing=False):

        log_dict = {"Epochs": dict()}
        pos_weight = self.train_dataset.get_weights()

        self.loss_function = self.loss()

        #  creating dataloaders
        train_loader = self.load_data(self.train_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing)
        val_loader = self.load_data(self.val_dataset, batch_size, num_workers=num_workers,
                                        shuffle=True, pin_memory=asynchronity, bucketing=bucketing)

        steps = steps=len(train_loader)
        self.set_optimizer(epochs=epochs, steps=steps, optimizer=optimizer, learning_rate=learning_rate, 
                weight_decay=weight_decay, lr_schedule=lr_schedule, end_lr=end_lr, amsgrad=amsgrad, fused_OptBack=fused_OptBack)

        scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)


        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}') 
            log_dict["Epochs"][epoch] = {"Training": dict(), "Validation": dict()}
            log_dict["Epochs"][epoch]["Training"]["Loss"] = list()
            log_dict["Epochs"][epoch]["Training"]["Prediction"] = list()
            log_dict["Epochs"][epoch]["Training"]["Labels"] = list()
            log_dict["Epochs"][epoch]["Validation"]["Loss"] = list()
            log_dict["Epochs"][epoch]["Validation"]["Prediction"] = list()
            log_dict["Epochs"][epoch]["Validation"]["Labels"] = list()
            log_dict["Epochs"][epoch]["Learning rate"] = list()

            if epoch >= epochs-1:
                log_dict["Epochs"][epoch]["Validation"]["Protein Names"] = list()
                log_dict["Epochs"][epoch]["Validation"]["Genome Names"] = list()
                log_dict["Epochs"][epoch]["Validation"]["Attentions"] = list()
                log_dict["Epochs"][epoch]["Training"]["Genome Names"] = list()

            #  training
            loss_lst_train, mcc_lst, prediction_lst, labels_lst, lr_rate_lst, profiler, genome_names = self.train_pass(
                                                                    train_loader=train_loader,
                                                                    scaler=scaler,
                                                                    mixed_precision=mixed_precision,
                                                                    clipping=clipping)
            if epoch >= epochs-1:
                log_dict["Epochs"][epoch]["Training"]["Genome Names"].extend(genome_names)
            log_dict["Epochs"][epoch]["Training"]["Loss"].extend(loss_lst_train)
            log_dict["Epochs"][epoch]["Training"]["Prediction"].extend(prediction_lst)
            log_dict["Epochs"][epoch]["Training"]["Labels"].extend(labels_lst)
            log_dict["Epochs"][epoch]["Learning rate"].extend(lr_rate_lst)

            #  validation
            print('validating...')
            if epoch >= epochs-1:
                loss_lst_val, mcc_lst, prediction_lst, labels_lst, att_results, profiler = self.val_pass(
                                                                val_loader=val_loader, last_epoch=True)
                log_dict["Epochs"][epoch]["Validation"]["Protein Names"].extend(att_results["Proteins"])                
                log_dict["Epochs"][epoch]["Validation"]["Genome Names"].extend(att_results["Genomes"])                
                log_dict["Epochs"][epoch]["Validation"]["Attentions"].extend(att_results["Attentions"])                
            else:
                loss_lst_val, mcc_lst, prediction_lst, labels_lst, profilerr = self.val_pass(
                                                                    val_loader=val_loader)
            log_dict["Epochs"][epoch]["Validation"]["Loss"].extend(loss_lst_val)
            log_dict["Epochs"][epoch]["Validation"]["Prediction"].extend(prediction_lst)
            log_dict["Epochs"][epoch]["Validation"]["Labels"].extend(labels_lst)
            if self.lr_scheduler is not None and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.lr_scheduler.step(np.nanmean(loss_lst_val))
            #print("training_loss: {}, validation_loss: {}, valClust_loss".format(
             #   round(np.mean(log_dict["Epochs"][epoch]["Training"]['Loss']), 4), round(np.mean(log_dict["Epochs"][epoch]["Validation"]["Loss"]), 4))
            #)
            print("training_loss: {}, validation_loss: {}, valClust_loss".format(
                round(np.mean(loss_lst_train), 4), round(np.mean(loss_lst_val), 4))
            )
        if self.profiler is not None:
            self.stop_memory_reports()
        return log_dict

    def savedict_train_results(self, train_res):
        with open("{}/train_results.pkl".format(self.results_dir), 'wb') as f:
            pickle.dump(train_res, f)

    def get_network(self):
        return self.network.cpu()

    def predict(self, x):
        with torch.inference_mode():
            logits = self.network(x)
        return torch.sigmoid(logits)

