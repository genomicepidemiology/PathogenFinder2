import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR,SequentialLR, OneCycleLR
from tqdm import tqdm
import h5py
import argparse
import pickle
import gc
import os
import pandas as pd
import numpy as np
import sys

sys.dont_write_bytecode = True

from data_utils import ProteomeDataset, ToTensor


class Train_NeuralNet():
    def __init__(self, network, learning_rate=1e-5, weight_decay=1e-4, amsgrad=False,
                loss_function=None):

        self.device = self.get_device()
        self.network = network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                            lr=learning_rate, weight_decay=weight_decay,
                                        amsgrad=amsgrad)

        self.loss = loss_function
        self.train_dataset = None
        self.val_dataset = None

    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def create_dataset(self, data_df, data_loc, data_type="train", dual_pred=False, cluster_sample=False,
                        cluster_tsv=None, weighted=False):
        if data_type == "validation":
            dataset = ProteomeDataset(csv_file=data_df, root_dir=data_loc,
                                        transform=ToTensor(), dual_pred=dual_pred)
        else:
            if not cluster_sample:
                dataset = ProteomeDataset(csv_file=data_df, root_dir=data_loc,
                                        transform=ToTensor(), dual_pred=dual_pred, weighted=weighted)
            else:
                dataset = ProteomeDataset(csv_file=data_df, root_dir=data_loc,
                            transform=ToTensor(), cluster_sampling=cluster_sample,
                            cluster_tsv=cluster_tsv, dual_pred=dual_pred, weighted=weighted)
        if data_type == "train":
            self.train_dataset = dataset
        elif data_type == "validation":
            self.val_dataset = dataset
        else:
            raise ValueError("The data_type {} is not an option (choose between train and val)".format(
                                data_type))

    def load_data(self, data_set, batch_size, num_workers=4, shuffle=True):
        return DataLoader(data_set, batch_size=batch_size, num_workers=num_workers,
                              collate_fn=ProteomeDataset.collate_fn_mask,
                              shuffle=shuffle, persistent_workers=False, pin_memory=False)

    def set_schedule_lr(self, epochs, steps, end_lr=3/2):
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                    max_lr=self.optimizer.param_groups[-1]['lr'],
                                    epochs=epochs, steps_per_epoch=steps, 
                                    )
        return scheduler

    def train_pass(self, train_loader):
        loss_lst = []
        prediction_lst = []
        labels_lst = []
        lr_rate_lst = []
        for batch in tqdm(train_loader):
            embeddings = batch["Embeddings"]
            labels = batch["Pathogen_Label"]
            lengths = batch["Length_Proteome"]
            #  sending data to device
            embeddings, labels, lengths = embeddings.to(self.device), labels.to(self.device), lengths.to(self.device)
            #  resetting gradients
            self.optimizer.zero_grad()
            #  making predictions
            predictions_logit, attentions = self.network(embeddings, lengths)
            predictions = torch.sigmoid(predictions_logit)
            if self.loss_function == nn.BCELoss():
                loss = self.loss_function(predictions, labels)
            else:
                loss = self.loss_function(predictions_logit, labels)
            #  computing loss
            loss_lst.append(loss.detach().cpu().tolist())
            prediction_lst.extend(predictions.detach().cpu().tolist())
            labels_lst.extend(labels.detach().cpu().tolist())
            #  computing gradients
            loss.backward()
            #  updating weights
            self.optimizer.step()
            lr_rate_lst.append(self.optimizer.param_groups[-1]['lr'])
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            #  clean gpu (maybe unnecessary)
            torch.cuda.empty_cache()
            gc.collect()
        return loss_lst, prediction_lst, labels_lst, lr_rate_lst

    def val_pass(self, val_loader, last_epoch=False):
        loss_lst = []
        prediction_lst = []
        labels_lst = []
        if last_epoch:
            att_results = {"Genomes": [], "Proteins":[], "Attentions": []}
        else:
            att_results = None
        with torch.no_grad():
            for batch in tqdm(val_loader):
                embeddings = batch["Embeddings"]
                labels = batch["Pathogen_Label"]
                lengths = batch["Length_Proteome"]
                #  sending data to device
                embeddings, labels, lengths = embeddings.to(self.device), labels.to(self.device), lengths.to(self.device)
                #  making predictions
                predictions_logit, attentions = self.network(embeddings, lengths)
                predictions = torch.sigmoid(predictions_logit)
                #  computing loss
                loss = self.loss_function(predictions, labels)
                loss_lst.append(loss.cpu().tolist())
                prediction_lst.extend(predictions.cpu().tolist())
                labels_lst.extend(labels.detach().cpu().tolist())
                if last_epoch:
                    genome_names = batch["File_Name"]
                    att_results["Genomes"].extend(genome_names)
                    prot_names = batch["Protein_Names"]
                    att_results["Proteins"].append(prot_names)
                    attentions = attentions.detach().cpu().numpy()
                    att_results["Attentions"].append(attentions)
                #  clean gpu (maybe unnecessary)
                torch.cuda.empty_cache()
                gc.collect()
        if last_epoch:
            return loss_lst, prediction_lst, labels_lst, att_results
        else:
            return loss_lst, prediction_lst, labels_lst        

    def train(self, epochs, batch_size, lr_schedule=False, end_lr=3/2, save_model=None):

        log_dict = {"Epochs": dict()}
        pos_weight = self.train_dataset.get_weights()

        #self.loss_function = self.loss(pos_weight=pos_weight.to(self.device))
        self.loss_function = self.loss()

        #  creating dataloaders
        train_loader = self.load_data(self.train_dataset, batch_size, num_workers=4,
                                        shuffle=True)
        val_loader = self.load_data(self.val_dataset, batch_size, num_workers=4,
                                        shuffle=True)

        if lr_schedule:
            self.lr_scheduler = self.set_schedule_lr(end_lr=end_lr, epochs=epochs,
                                                        steps=len(train_loader))
        else:
            self.lr_scheduler = None



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

            #  training
            loss_lst, prediction_lst, labels_lst, lr_rate_lst = self.train_pass(
                                                        train_loader=train_loader)

            log_dict["Epochs"][epoch]["Training"]["Loss"].extend(loss_lst)
            log_dict["Epochs"][epoch]["Training"]["Prediction"].extend(prediction_lst)
            log_dict["Epochs"][epoch]["Training"]["Labels"].extend(labels_lst)
            log_dict["Epochs"][epoch]["Learning rate"].extend(lr_rate_lst)
            #  validation
            print('validating...')
            if epoch >= epochs-1:
                loss_lst, prediction_lst, labels_lst, att_results = self.val_pass(
                                            val_loader=val_loader, last_epoch=True)
                log_dict["Epochs"][epoch]["Validation"]["Protein Names"].extend(att_results["Proteins"])                
                log_dict["Epochs"][epoch]["Validation"]["Genome Names"].extend(att_results["Genomes"])                
                log_dict["Epochs"][epoch]["Validation"]["Attentions"].extend(att_results["Attentions"])                
            else:
                loss_lst, prediction_lst, labels_lst = self.val_pass(
                                                        val_loader=val_loader)
            log_dict["Epochs"][epoch]["Validation"]["Loss"].extend(loss_lst)
            log_dict["Epochs"][epoch]["Validation"]["Prediction"].extend(prediction_lst)
            log_dict["Epochs"][epoch]["Validation"]["Labels"].extend(labels_lst)

            print("training_loss: {}, validation_loss: {}, valClust_loss".format(
                round(np.mean(log_dict["Epochs"][epoch]["Training"]['Loss']), 4), round(np.mean(log_dict["Epochs"][epoch]["Validation"]["Loss"]), 4))
            )
        
        if save_model is not None:
            torch.save(self.network, '{}'.format(save_model))
        return log_dict, self.network

    def predict(self, x):
        logits = self.network(x)
        return torch.sigmoid(logits)

