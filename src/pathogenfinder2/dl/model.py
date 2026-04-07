"""
Deep learning model wrapper for PathogenFinder2.

:class:`PathogenDLModel` wraps the PyTorch model with helpers for training,
testing, and ensemble inference.  It selects the correct network architecture
(ConvNext-AddAtt, FNN, …) from the model parameters dict, sets up reporting
(wandb or file), and exposes :meth:`train_model`, :meth:`test_model`, and
:meth:`predict_model`.
"""
import random
import logging
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

from pathogenfinder2.dl.engine import (
    train_loop, run_inference,
    create_test_results, calculate_measures, graph_objects, save_test_results,
)
from pathogenfinder2.dl.utils.network import NetworkModule
from pathogenfinder2.dl.utils.data import EmbeddingData
from pathogenfinder2.dl.utils.reporting import WandbReport, FileReport, MemoryReport
from pathogenfinder2.dl.utils.optimizer import Optimizer



class PathogenDLModel:
    """PyTorch model wrapper for PathogenFinder2 training, testing, and inference.

    Handles architecture selection, report setup, and exposes a uniform
    interface for :meth:`train_model`, :meth:`test_model`, and
    :meth:`predict_model`.

    Parameters
    ----------
    model_parameters:
        Sub-dict from the PF2 configuration containing architecture,
        weights, batch size, and data-loader settings.
    misc_parameters:
        Sub-dict with reporting mode, output folder, run name, etc.
    seed:
        Optional integer random seed for reproducibility.
    """

    def __init__(self, model_parameters: dict, misc_parameters: dict,
                 seed: int | None = None) -> None:

        if seed is not None:
            PathogenDLModel.init_seed(seed)

        self.model_type = PathogenDLModel.get_model(model_parameters["Model Name"])
        self.misc_parameters = misc_parameters
        self.model_parameters = model_parameters

        if misc_parameters["Report Results"] == "wandb":
            self.reportNN = WandbReport(
                name=misc_parameters["Name"],
                wandb_dir=misc_parameters["Results Folder"],
                configuration=model_parameters,
                project=misc_parameters["Project Name"],
            )
        elif misc_parameters["Report Results"] == "file":
            self.reportNN = FileReport(
                name=misc_parameters["Name"],
                configuration=model_parameters,
                dict_dir=misc_parameters["Results Folder"],
                modes=misc_parameters["Actions"],
            )
        else:
            raise ValueError("Reporting results as {} is not available".format(misc_parameters["Report Results"]))

        if model_parameters["Memory Report"]:
            self.memory_report = MemoryReport(results_dir=misc_parameters["Results Folder"], process="neuralnetworkinference")
            self.memory_report.start_memory_reports()
        else:
            self.memory_report = None



    @staticmethod
    def get_model(model_type):
        if model_type == "Conv1D-AddAtt":
            from .models.conv1d_addatt import Conv1D_AddAtt_Net
            return Conv1D_AddAtt_Net
        elif model_type == "Conv1D":
            from .models.conv1d import Conv1D_Net
            return Conv1D_Net
        elif model_type == "FNN":
            from .models.fnn import FNN_Net
            return FNN_Net
        elif model_type == "AddAtt":
            from .models.addatt import AddAtt_Net
            return AddAtt_Net
        elif model_type == "DenseNet":
            from .models.densenet import DenseNet_Net
            return DenseNet_Net
        elif model_type == "DenseNet-AddAtt":
            from .models.densenet_addatt import DenseNet_AddAtt_Net
            return DenseNet_AddAtt_Net
        elif model_type == "ConvNext":
            from .models.convnext import ConvNext_Net
            return ConvNext_Net
        elif model_type == "ConvNext-AddAtt":
            from .models.convnext_addatt import ConvNext_AddAtt_Net
            return ConvNext_AddAtt_Net
        else:
            raise ValueError(
                    "The model {} is not available for Pathogenicity prediction".format(model_type))

    @staticmethod
    def init_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


    def train_model(self, train_parameters: dict) -> None:
        logger.debug("Misc parameters: %s", self.misc_parameters)
        network_module = NetworkModule(
            model_type=self.model_type,
            out_folder=self.misc_parameters["Results Folder"],
            model_parameters=self.model_parameters,
            mixed_precision=self.model_parameters["Mixed Precision"],
            results_module=self.reportNN,
            memory_profiler=self.memory_report,
            loss_type=self.model_parameters["Loss Function"],
        )

        train_df = EmbeddingData.create_dataset(
            input_type="protein_embeddings",
            data_df=train_parameters["Train DF"],
            data_loc=train_parameters["Train Loc"],
            data_type="train", dual_pred=False,
        )
        val_df = EmbeddingData.create_dataset(
            input_type="protein_embeddings",
            data_df=train_parameters["Validation DF"],
            data_loc=train_parameters["Validation Loc"],
            data_type="prediction", dual_pred=False,
        )

        batch_size = self.model_parameters["Batch Size"]
        if batch_size > 45:
            batch_size, accumulate_gradient = NetworkModule.set_sensible_batch_size(batch_size)
        else:
            accumulate_gradient = 1

        num_workers = self.model_parameters["Data Parameters"]["num_workers"]
        asynchronity = self.model_parameters["Data Parameters"]["asynchronity"]

        train_loader = EmbeddingData.load_data(
            train_df, batch_size, num_workers=num_workers, shuffle=True, pin_memory=asynchronity,
            bucketing=self.model_parameters["Data Parameters"]["bucketing"],
            stratified=self.model_parameters["Data Parameters"]["stratified"],
        )
        val_loader = EmbeddingData.load_data(
            val_df, batch_size, num_workers=num_workers, shuffle=True, pin_memory=asynchronity,
            bucketing=False, stratified=False,
        )
        train_steps = len(train_loader)

        if self.model_parameters["Network Weights"] is not None:
            model_params = network_module.load_model(self.model_parameters["Network Weights"])
        else:
            model_params = None

        if model_params is None:
            optimizer = Optimizer(
                network=network_module.network,
                optimizer=train_parameters["Optimizer"],
                learning_rate=train_parameters["Learning Rate"],
                weight_decay=train_parameters["Weight Decay"],
                amsgrad=False,
            )
            if train_parameters["Lr Scheduler"]:
                optimizer.set_scheduler(
                    scheduler_type=train_parameters["Lr Scheduler"],
                    patience=None, milestones=None, gamma=None, end_lr=None,
                    steps=train_steps, epochs=train_parameters["Epochs"],
                )
            if train_parameters["Warm Up"]:
                optimizer.set_warmup(warmup_period=train_parameters["Warm Up"])
        else:
            # Resume from checkpoint: reconstruct the Optimizer wrapper then
            # restore saved state so training continues at the correct LR.
            saved_lr = model_params["Optimizer State"]["param_groups"][-1]["lr"]
            optimizer = Optimizer(
                network=network_module.network,
                optimizer=train_parameters["Optimizer"],
                learning_rate=saved_lr,
                weight_decay=train_parameters["Weight Decay"],
                amsgrad=False,
            )
            optimizer.optimizer.load_state_dict(model_params["Optimizer State"])
            if train_parameters["Lr Scheduler"]:
                optimizer.set_scheduler(
                    scheduler_type=train_parameters["Lr Scheduler"],
                    patience=None, milestones=None, gamma=None, end_lr=None,
                    steps=train_steps, epochs=train_parameters["Epochs"],
                )

        logger.info("Training for %d epochs", train_parameters["Epochs"])
        train_loop(
            network_module=network_module,
            train_loader=train_loader,
            val_loader=val_loader,
            batch_size=batch_size,
            accumulate_gradient=accumulate_gradient,
            asynchronity=asynchronity,
            optimizer=optimizer,
            model_storage=train_parameters["Save Model"],
            model_report=self.reportNN,
            epochs=train_parameters["Epochs"],
            model_params=model_params,
        )

    def test_model(self, predicted_data: dict, test_parameters: dict) -> None:
        label_df = pd.read_csv(test_parameters["Label File"], sep="\t")
        out_folder = "{}/test_results/".format(self.misc_parameters["Results Folder"])

        results_df = create_test_results(
            label_df=label_df, predicted_data=predicted_data, out_folder=out_folder
        )
        measures = calculate_measures(results_df=results_df)
        cm_display, roc_display, curve_cal = graph_objects(results_df=results_df)
        save_test_results(
            measures=measures, out_folder=out_folder,
            cm_display=cm_display, roc_display=roc_display, curve_cal=curve_cal,
        )


    def _aggregate_model_results(
        self,
        ensemble_results: dict,
        results_inference: dict,
        input_metadata: pd.DataFrame,
    ) -> None:
        """Merge one model's inference output into the running ensemble dict."""
        for emb_key, sample in results_inference.items():
            if len(results_inference) > 1:
                fullname = input_metadata.loc[
                    input_metadata["File_Embedding"] == emb_key, "Input_Files"
                ].values[0]
                name = os.path.basename(fullname)
            else:
                name = "PF2"
            if name not in ensemble_results:
                ensemble_results[name] = {"Output": {}, "Features": {}}
                ensemble_results[name]["Features"]["Filepath"] = sample["Features"]["Filename"]
                ensemble_results[name]["Features"]["Proteome Length"] = sample["Features"]["Proteome Length"]
                ensemble_results[name]["Features"]["ProtIDs"] = sample["Features"]["ProtIDs"]
                ensemble_results[name]["Features"]["Filename"] = name
                for key, val in sample["Output"].items():
                    ensemble_results[name]["Output"][key] = val
            else:
                assert sample["Features"]["Filename"] == ensemble_results[name]["Features"]["Filepath"]
                assert sample["Features"]["Proteome Length"] == ensemble_results[name]["Features"]["Proteome Length"]
                for key, val in sample["Output"].items():
                    if key == "Prediction":
                        ensemble_results[name]["Output"][key].extend(val)
                    else:
                        ensemble_results[name]["Output"][key] = np.concatenate(
                            (ensemble_results[name]["Output"][key], val)
                        )

    def predict_model(self, input_metapath: str, save_embeddings: bool = False,
                      save_attentions: bool = False) -> dict:

        ensemble_results = {}
        input_metadata = pd.read_csv(input_metapath, sep="\t")
        inference_df = EmbeddingData.create_dataset(
            input_type="protein_embeddings",
            data_df=input_metapath,
            data_type="prediction", dual_pred=False,
        )
        amount_nets = len(self.model_parameters["Network Weights"])
        for ens in tqdm(range(amount_nets), total=amount_nets, desc="Infering from Neural Network Models",
                        unit="Neural Network", dynamic_ncols=True, smoothing=0.05, mininterval=0.2, position=0):
            network_module = NetworkModule(
                model_type=self.model_type,
                out_folder=self.misc_parameters["Results Folder"],
                model_parameters=self.model_parameters,
                mixed_precision=self.model_parameters["Mixed Precision"],
                results_module=self.reportNN,
                memory_profiler=self.memory_report,
                loss_type=self.model_parameters["Loss Function"],
            )
            results_inference = run_inference(
                network_module=network_module,
                model_weights=self.model_parameters["Network Weights"][ens],
                inference_dataset=inference_df,
                num_workers=self.model_parameters["Data Parameters"]["num_workers"],
                asynchronity=self.model_parameters["Data Parameters"]["asynchronity"],
                return_embeddings=save_embeddings,
                return_attentions=save_attentions,
            )
            self._aggregate_model_results(ensemble_results, results_inference, input_metadata)
        return ensemble_results