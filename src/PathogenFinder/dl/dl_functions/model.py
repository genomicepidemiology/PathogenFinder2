import random
import torch
import numpy as np

from dl.dl_functions.train_model import Train_NeuralNetwork
from dl.dl_functions.inference_model import Inference_NeuralNetwork
from dl.utils.nn_utils import Network_Module
from dl.utils.data_utils import NN_Data
from dl.utils.report_utils import ReportNN, Memory_Report



class Pathogen_DLModel:

    def __init__(self, model_parameters, misc_parameters, seed=False):

        if seed:
            Pathogen_DLModel.init_seed(seed)

        self.model_type = Pathogen_DLModel.get_model(model_parameters["Model Name"])
        self.misc_parameters = misc_parameters
        self.model_parameters = model_parameters

        if misc_parameters["Report Results"] == "wandb":
            self.reportNN = ReportNN(report_wandb=True, report_dict=False,
                                        configuration=model_parameters,
                                        name=misc_parameters["Name"],
                                        path_dir=misc_parameters["Results Folder"],
                                        project=misc_parameters["Project Name"])
        elif misc_parameters["Report Results"] == "file":
            self.reportNN = ReportNN(report_wandb=False, report_dict=True,
                                        configuration=model_parameters,
                                        name=misc_parameters["Name"],
                                        path_dir=misc_parameters["Results Folder"],
                                        modes=misc_parameters["Actions"])
        else:
            raise ValueError("Reporting results as {} is not available".format(misc_parameters["Report Results"]))

        if model_parameters["Memory Report"]:
            self.memory_report = Memory_Report(results_dir=misc_parameters["Results Folder"], process="neuralnetworkinference")
            self.memory_report.start_memory_reports()
        else:
            self.memory_report = None



    @staticmethod
    def get_model(model_type):
        if model_type == "Conv1D-AddAtt":
            from dl.models.conv1d_addatt import Conv1D_AddAtt_Net
            return Conv1D_AddAtt_Net
        elif model_type == "Conv1D":
            from dl.models.conv1d import Conv1D_Net
            return Conv1D_Net
        elif model_type == "FNN":
            from dl.models.fnn import FNN_Net
            return FNN_Net
        elif model_type == "AddAtt":
            from dl.models.addatt import AddAtt_Net
            return AddAtt_Net
        elif model_type == "DenseNet":
            from dl.models.densenet import DenseNet_Net
            return DenseNet_Net
        elif model_type == "DenseNet-AddAtt":
            from dl.models.densenet_addatt import DenseNet_AddAtt_Net
            return DenseNet_AddAtt_Net
        elif model_type == "ConvNext":
            from dl.models.convnext import ConvNext_Net
            return ConvNext_Net
        elif model_type == "ConvNext-AddAtt":
            from dl.models.convnext_addatt import ConvNext_AddAtt_Net
            return ConvNext_AddAtt_Net
        else:
            raise ValueError(
                    "The model {} is not available for Pathogenicity prediction".format(model_type))

    @staticmethod
    def init_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


    def train_model(self, train_parameters):

        network_module = Network_Module(model_type=self.model_type,
                                        out_folder=self.misc_parameters["Results Folder"],
                                        model_parameters=self.model_parameters,
                                        mixed_precision=self.model_parameters["Mixed Precision"],
                                        results_module=self.reportNN,
                                        memory_profiler=self.memory_report,
                                        loss_type=self.model_parameters["Loss Function"])

        train_instance = Train_NeuralNetwork(network_module=network_module,
                                            model_report=self.reportNN,
                                            model_storage=train_parameters["Save Model"])


        train_df = NN_Data.create_dataset(input_type="protein_embeddings",
                                    data_df=train_parameters["Train DF"],
                                    data_loc=train_parameters["Train Loc"],
                                    data_type="train", dual_pred=False)
        val_df = NN_Data.create_dataset(input_type="protein_embeddings",
                                    data_df=train_parameters["Validation DF"],
                                    data_loc=train_parameters["Validation Loc"],
                                    data_type="prediction", dual_pred=False)

        train_instance.set_dataloaders(train_dataset=train_df, val_dataset=val_df, batch_size=self.model_parameters["Batch Size"],
                                        num_workers=self.model_parameters["Data Parameters"]["num_workers"],
                                        asynchronity=self.model_parameters["Data Parameters"]["asynchronity"],
                                        bucketing=self.model_parameters["Data Parameters"]["bucketing"],
                                        stratified=self.model_parameters["Data Parameters"]["stratified"])

        train_instance.set_optimizer(optimizer_class=train_parameters["Optimizer Parameters"]["optimizer"],
                                        learning_rate=train_parameters["Optimizer Parameters"]["learning_rate"],
                                        weight_decay=train_parameters["Optimizer Parameters"]["weight_decay"],
                                        amsgrad=False, scheduler_type=train_parameters["Optimizer Parameters"]["lr_scheduler"],
                                        warmup_period=train_parameters["Optimizer Parameters"]["warm_up"],
                                        patience=None, milestones=None, gamma=None, end_lr=None,
                                        epochs=train_parameters["Epochs"])
        
        if self.model_parameters["Network Weights"] is not None:
            model_params = network_module.load_model(self.model_parameters["Network Weights"],
                                                        optimizer=train_instance.optimizer.optimizer)
        else:
            model_params = None

        train_instance(epochs=train_parameters["Epochs"], model_params=model_params)

    def test_model(self):
        pass

    def predict_model(self, inference_parameters):
        network_module = Network_Module(model_type=self.model_type,
                                        out_folder=self.misc_parameters["Results Folder"],
                                        model_parameters=self.model_parameters,
                                        mixed_precision=self.model_parameters["Mixed Precision"],
                                        results_module=self.reportNN,
                                        memory_profiler=self.memory_report,
                                        loss_type=self.model_parameters["Loss Function"])

        inference_df = NN_Data.create_dataset(input_type="protein_embeddings",
                                                data_df=inference_parameters["Input Metadata"],
                                                data_type="prediction", dual_pred=False)

        inference_instance = Inference_NeuralNetwork(network_module=network_module,
                                                    model_report=self.reportNN,
                                                    model_weights=self.model_parameters["Network Weights"],
                                                    out_folder=self.misc_parameters["Results Folder"])
        inference_instance.set_dataloader(inference_dataset=inference_df,
                                    num_workers=self.model_parameters["Data Parameters"]["num_workers"],
                                    asynchronity=self.model_parameters["Data Parameters"]["asynchronity"])

        inference_instance()


    def hyperparamOpt_model(self):
        pass


