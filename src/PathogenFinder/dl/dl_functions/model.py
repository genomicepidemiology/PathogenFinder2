import random
import torch
import numpy as np
import pandas as pd
import os
from collections import defaultdict

from .train_model import Train_NeuralNetwork
from .inference_model import Inference_NeuralNetwork
from .test_model import Test_NeuralNetwork
from ..utils.nn_utils import Network_Module
from ..utils.data_utils import NN_Data
from ..utils.report_utils import ReportNN, Memory_Report, Inference_Report



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
        if self.model_parameters["Network Weights"] is not None:
            model_params = network_module.load_model(self.model_parameters["Network Weights"])
        else:
            model_params = None
        
        if model_params is None:
            train_instance.set_optimizer(optimizer_class=train_parameters["Optimizer Parameters"]["optimizer"],
                                        learning_rate=train_parameters["Optimizer Parameters"]["learning_rate"],
                                        weight_decay=train_parameters["Optimizer Parameters"]["weight_decay"],
                                        amsgrad=False, scheduler_type=train_parameters["Optimizer Parameters"]["lr_scheduler"],
                                        warmup_period=train_parameters["Optimizer Parameters"]["warm_up"],
                                        patience=None, milestones=None, gamma=None, end_lr=None,
                                        epochs=train_parameters["Epochs"])
        else:
            train_instance.set_optimizer(optimizer=optimizer["Optimizer"],
                                        learning_rate=optimizer["Optimizer"].param_groups[-1]['lr'],
                                        weight_decay=train_parameters["Optimizer Parameters"]["weight_decay"],
                                        amsgrad=False, scheduler_type=train_parameters["Optimizer Parameters"]["lr_scheduler"],
                                        warmup_period=False,
                                        patience=None, milestones=None, gamma=None, end_lr=None,
                                        epochs=train_parameters["Epochs"])

        train_instance(epochs=train_parameters["Epochs"], model_params=model_params)

    def test_model(self, predicted_data, test_parameters):
        label_df = pd.read_csv(test_parameters["Label File"], sep="\t")
        
        test_instance = Test_NeuralNetwork(out_folder=self.misc_parameters["Results Folder"])

        results_df = test_instance.create_results(label_df=label_df, predicted_data=predicted_data)

        measures = test_instance.calculate_measures(results_df=results_df)
        cm_display, roc_display, curve_cal = test_instance.graph_objects(results_df=results_df)
        test_instance.save_results(measures=measures, cm_display=cm_display, roc_display=roc_display, curve_cal=curve_cal)


    def predict_model(self, inference_parameters):

        pathopred_ensemble = []
        protein_features_ensemble = []
        embedding_maps_ensemble = []
        ensemble_results = False
        input_metadata = pd.read_csv(inference_parameters["Input Metadata"], sep="\t")
        for ens in range(len(self.model_parameters["Network Weights"])):
            network_module = Network_Module(model_type=self.model_type,
                                        out_folder=self.misc_parameters["Results Folder"],
                                        model_parameters=self.model_parameters,
                                        mixed_precision=self.model_parameters["Mixed Precision"],
                                        results_module=self.reportNN,
                                        memory_profiler=self.memory_report,
                                        loss_type=self.model_parameters["Loss Function"],
                                        )
            inference_df = NN_Data.create_dataset(input_type="protein_embeddings",
                                                data_df=inference_parameters["Input Metadata"],
                                                data_type="prediction", dual_pred=False)

            inference_instance = Inference_NeuralNetwork(network_module=network_module,
                                                    model_report=self.reportNN,
                                                    model_weights=self.model_parameters["Network Weights"][ens],
                                                    out_folder=self.misc_parameters["Results Folder"])
            inference_instance.set_dataloader(inference_dataset=inference_df,
                                        num_workers=self.model_parameters["Data Parameters"]["num_workers"],
                                        asynchronity=self.model_parameters["Data Parameters"]["asynchronity"])

            results_inference = inference_instance(return_attentions=inference_parameters["Produce Attentions"],
                                                    return_embeddings=inference_parameters["Produce Embeddings"])
            
            if not ensemble_results:
                ensemble_results = {}
                for i, j in results_inference.items():
                    name = input_metadata.loc[input_metadata["File_Embedding"]==i, "Input Files"].values[0]
                    ensemble_results[name] = {"Output":{}, "Features":{}}
                    ensemble_results[name]["Features"]["Filepath"] = j["Features"]["Filename"]
                    ensemble_results[name]["Features"]["Proteome Length"] = j["Features"]["Proteome Length"]
                    ensemble_results[name]["Features"]["ProtIDs"] = j["Features"]["ProtIDs"]
                    ensemble_results[name]["Features"]["Filename"] = os.path.basename(name)
                    for io, jo in j["Output"].items():
                        ensemble_results[name]["Output"][io] = jo
            else:
                for i, j in results_inference.items():
                    name = input_metadata.loc[input_metadata["File_Embedding"]==i, "Input Files"].values[0]
                    for io, jo in j["Output"].items():
                        if io == "Prediction":
                            ensemble_results[name]["Output"][io].extend(jo)
                        else:
                            ensemble_results[name]["Output"][io] = np.concatenate((ensemble_results[name]["Output"][io], jo))
                    assert j["Features"]["Filename"] == ensemble_results[name]["Features"]["Filepath"]
                    assert j["Features"]["Proteome Length"] == ensemble_results[name]["Features"]["Proteome Length"]

        results_module = Inference_Report(out_folder=self.misc_parameters["Results Folder"])
        ensemble_results = Inference_Report.get_predictions(ensemble_results=ensemble_results)
        results_module.save_report(results_ensemble=ensemble_results,
                                    save_attentions=inference_parameters["Produce Attentions"],
                                    save_embeddings=inference_parameters["Produce Embeddings"])
        return ensemble_results




    def hyperparamOpt_model(self):
        pass


