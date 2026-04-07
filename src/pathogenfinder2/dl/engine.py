"""
Training loop, inference, and evaluation functions for PathogenFinder2.

This module contains all the execution-level logic that :class:`~model.PathogenDLModel`
delegates to:

- :class:`EarlyStopping` — stateful early-stopping criterion
- :func:`train_loop` — one full training run across N epochs
- :func:`run_inference` — load weights and run a forward pass
- :func:`create_test_results`, :func:`calculate_measures`, :func:`graph_objects`,
  :func:`save_test_results` — evaluation and reporting helpers
"""
import logging
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score, roc_auc_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc,
)
from sklearn.calibration import calibration_curve

from pathogenfinder2.dl.utils.data import EmbeddingData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.counter = 0

    def __call__(self, val_measure):
        if self.best is None or val_measure > self.best + self.delta:
            self.best = val_measure
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_loop(
    network_module,
    train_loader,
    val_loader,
    batch_size,
    accumulate_gradient,
    asynchronity,
    optimizer,
    model_storage,
    model_report,
    epochs,
    model_params=None,
):
    """Run the training loop and return the updated network_module.

    Args:
        network_module: :class:`~network.NetworkModule` instance.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split, or None.
        batch_size: Effective batch size after gradient accumulation adjustment.
        accumulate_gradient: Number of steps to accumulate before a weight update.
        asynchronity: Non-blocking device transfers.
        optimizer: :class:`~optimizer_utils.Optimizer` wrapper instance.
        model_storage: One of ``"last_epoch"``, ``"best_epoch"``, ``"early_stopping"``.
        model_report: Reporting object (``File_Report`` or ``Wandb_Report``).
        epochs: Number of epochs to train.
        model_params: Checkpoint dict from :meth:`~network.NetworkModule.load_model`,
            or ``None`` to start fresh.

    Returns:
        The updated ``network_module``.
    """
    if model_storage not in ["last_epoch", "early_stopping", "best_epoch"]:
        raise ValueError(
            f"model_storage must be one of 'last_epoch', 'early_stopping', 'best_epoch'; got {model_storage!r}"
        )

    if model_report:
        model_report.start_train_report(
            model=network_module.network, criterion=network_module.loss_function
        )

    max_mcc_val = 0
    init_epoch = 0 if model_params is None else model_params["Epoch"]
    early_stopping = (
        EarlyStopping(patience=7, delta=0.001)
        if model_storage == "early_stopping"
        else None
    )

    for epoch in range(init_epoch, init_epoch + epochs):
        logger.info("Epoch %d/%d", epoch + 1, init_epoch + epochs)
        start_e_time = time.time()
        loss_val, mcc_v = None, None
        loss_train, mcc_t, lr_rate, optimizer = network_module.train_pass(
            train_loader=train_loader,
            batch_size=batch_size,
            results_module=model_report,
            optimizer=optimizer,
            asynchronity=asynchronity,
            accumulate_gradient=accumulate_gradient,
        )

        if val_loader is not None:
            loss_val, mcc_v = network_module.validation_pass(
                val_loader=val_loader, asynchronity=asynchronity, batch_size=batch_size
            )

        if model_report:
            log_results = {
                "Training Loss": loss_train,
                "Validation Loss": loss_val,
                "Training MCC": mcc_t,
                "Validation MCC": mcc_v,
                "Epoch": epoch,
                "Epoch Runtime (seconds)": time.time() - start_e_time,
            }
            model_report.add_epoch_info(log_results)

        if optimizer.lr_scheduler is not None:
            if isinstance(optimizer.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                optimizer.update_scheduler(value=loss_val)
            elif isinstance(optimizer.lr_scheduler, lr_scheduler.MultiStepLR):
                optimizer.update_scheduler()

        if model_storage != "last_epoch" and val_loader is not None:
            if max_mcc_val < mcc_v:
                network_module.save_model(
                    optimizer=optimizer.optimizer,
                    loss=loss_train,
                    mcc_val=mcc_v,
                    epoch=epoch,
                )
                max_mcc_val = mcc_v

        if early_stopping is not None and val_loader is not None:
            if early_stopping(val_measure=mcc_v):
                break

    if model_storage == "last_epoch" and epochs > 0:
        network_module.save_model(
            optimizer=optimizer.optimizer, loss=loss_train, mcc_val=mcc_v, epoch=epoch
        )

    if network_module.memory_profiler:
        network_module.memory_profiler.stop_memory_reports()
    if model_report:
        model_report.finish_report()
    return network_module


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    network_module,
    model_weights,
    inference_dataset,
    num_workers: int = 2,
    asynchronity: bool = False,
    max_batch_size: int = 45,
    return_embeddings: bool = False,
    return_attentions: bool = False,
) -> dict:
    """Load *model_weights* and run a forward pass over *inference_dataset*.

    Args:
        network_module: :class:`~network.NetworkModule` instance (fresh per ensemble member).
        model_weights: Path to the weights ``.pt`` file.
        inference_dataset: A dataset object compatible with :func:`~data.EmbeddingData.load_data`.
        num_workers: DataLoader worker processes.
        asynchronity: Non-blocking device transfers (requires ``pin_memory``).
        max_batch_size: If the dataset is larger than this, use ``batch_size=1`` to
            avoid out-of-memory errors during multi-sample inference.
        return_embeddings: Capture and return intermediate embeddings.
        return_attentions: Return per-protein attention weights.

    Returns:
        Dict mapping embedding-file paths to sample result dicts.
    """
    network_module.load_model(model_weights)

    batch_size = 1 if len(inference_dataset) > max_batch_size else len(inference_dataset)
    inference_loader = EmbeddingData.load_data(
        inference_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=asynchronity,
        drop_last=False,
    )

    batches_results = network_module.predictive_pass(
        val_loader=inference_loader,
        asynchronity=asynchronity,
        batch_size=batch_size,
        record_attentions=return_attentions,
        record_embeddings=return_embeddings,
    )

    if network_module.memory_profiler:
        network_module.memory_profiler.step()

    results_inference = {}
    for batch in batches_results:
        results_inference.update(batch.get_samples())

    if network_module.memory_profiler:
        network_module.memory_profiler.stop_memory_reports()

    return results_inference


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def create_test_results(label_df, predicted_data, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    label_df["Predicted Label"] = np.nan
    label_df["Predicted Cont"] = np.nan
    label_df["Predicted Std"] = np.nan
    label_df["Predicted Phenotype"] = None
    for k, val in predicted_data.items():
        label_df.loc[label_df["File Name"]==val["Features"]["Filename"], "Predicted Label"] = val["Ensemble Predictions"]["Binary Prediction Mean"]
        label_df.loc[label_df["File Name"]==val["Features"]["Filename"], "Predicted Cont"] = val["Ensemble Predictions"]["Prediction Mean"]
        label_df.loc[label_df["File Name"]==val["Features"]["Filename"], "Predicted Std"] = val["Ensemble Predictions"]["Prediction STD"]
        label_df.loc[label_df["File Name"]==val["Features"]["Filename"], "Predicted Phenotype"] = val["Ensemble Predictions"]["Phenotype"]
    logger.debug("Rows with nulls:\n%s", label_df[pd.isnull(label_df).any(axis=1)])
    logger.debug("Any nulls: %s", label_df.isnull().values.any())
    assert not label_df.isnull().values.any()
    label_df.to_csv("{}/test_results.tsv".format(out_folder), sep="\t", index=False)
    return label_df


def calculate_measures(results_df):
    mcc = matthews_corrcoef(results_df["PathoPhenotype"], results_df["Predicted Label"])
    bacc = balanced_accuracy_score(results_df["PathoPhenotype"], results_df["Predicted Label"])
    rocauc = roc_auc_score(results_df["PathoPhenotype"], results_df["Predicted Label"])
    sensit = recall_score(results_df["PathoPhenotype"], results_df["Predicted Label"], pos_label=1)
    specifi = recall_score(results_df["PathoPhenotype"], results_df["Predicted Label"], pos_label=0)
    return {"MCC": mcc, "BACC": bacc, "ROCAUC": rocauc, "Sensitivity": sensit, "Specificity": specifi}


def graph_objects(results_df):
    cm = confusion_matrix(results_df["PathoPhenotype"], results_df["Predicted Label"])
    cm_display = ConfusionMatrixDisplay(cm)
    fpr, tpr, thresholds = roc_curve(results_df["PathoPhenotype"], results_df["Predicted Label"])
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        results_df["PathoPhenotype"], results_df["Predicted Cont"], n_bins=10
    )
    curve_cal = {"frac_pos": fraction_of_positives, "mean_pred": mean_predicted_value}
    return cm_display, roc_display, curve_cal


def save_test_results(measures, out_folder, cm_display=None, roc_display=None, curve_cal=None):
    measures_file = "{}/measures.txt".format(out_folder)
    with open(measures_file, "w") as measuresf:
        measuresf.write("======== Test Results ==========\n\n")
        for k, v in measures.items():
            measuresf.write("{}: {}\n".format(k, v))
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    plt.savefig("{}/cfm.png".format(out_folder))
    plt.close()
    fig, ax = plt.subplots()
    roc_display.plot(ax=ax)
    plt.savefig("{}/roc.png".format(out_folder))
    plt.close()
    fig, ax = plt.subplots()
    ax.plot(curve_cal["mean_pred"], curve_cal["frac_pos"], "s-")
    plt.savefig("{}/cal_curve.png".format(out_folder))
    plt.close()
