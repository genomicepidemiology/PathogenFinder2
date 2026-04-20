"""
Neural-network result containers and training/inference reporters for PathogenFinder2.

:class:`BatchResults` stores the raw tensor outputs for one inference batch and
exposes :meth:`~BatchResults.get_samples` to convert them into the nested dict
format used by the rest of the pipeline.

:class:`FileReport` serialises training metrics to a pickle file.
:class:`WandbReport` streams metrics to Weights & Biases (optional dependency).
:class:`MemoryReport` wraps the PyTorch CUDA memory profiler.
"""
import logging
import pickle
import torch
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class BatchResults:
    """Container for the raw outputs of one inference batch.

    Parameters
    ----------
    filenames:
        List of embedding file paths, one per sample in the batch.
    predictions:
        Tensor of sigmoid probabilities, shape ``(batch, 1)``.
    protIDs:
        List of protein-ID arrays, one per sample.
    proteome_lengths:
        Tensor of actual proteome lengths before padding.
    attentions:
        Optional attention weight tensor, or ``None``.
    embeddings1:
        Optional intermediate embeddings from ``classifier.norm_layer``.
    embeddings2:
        Optional intermediate embeddings from ``hook_postatt``.
    """

    def __init__(self, filenames, predictions, protIDs, proteome_lengths, attentions=None, embeddings1=None,
            embeddings2=None):

        self.filenames = filenames
        self.predictions = predictions
        self.protIDs = protIDs
        self.attentions = attentions
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.proteome_lengths = proteome_lengths

        assert len(self.filenames) == len(self.predictions)
        assert len(self.predictions) == len(self.protIDs)
        assert len(self.protIDs) == len(self.proteome_lengths)

        if attentions is not None:
            assert len(self.attentions) == len(self.proteome_lengths)
        if embeddings1 is not None:
            assert len(self.embeddings1) == len(self.proteome_lengths)
        if embeddings2 is not None:
            assert len(self.embeddings2) == len(self.proteome_lengths)

    def __len__(self):
        return len(self.filenames)

    def get_samples(self):
        samples = {}
        cuda_av = torch.cuda.is_available()
        for n in range(len(self)):
            name = str(self.filenames[n])
            samples[name] = {}
            samples[name]["Features"] = {}
            samples[name]["Features"]["Filename"] = [self.filenames[n]]
            samples[name]["Features"]["ProtIDs"] = self.protIDs[n][:int(self.proteome_lengths[n][0])]
            samples[name]["Output"] = {}
            samples[name]["Output"]["Prediction"] = self.predictions[n].tolist()
            samples[name]["Features"]["Proteome Length"] = self.proteome_lengths[n][0]
            if self.attentions is None:
                samples[name]["Output"]["Attention"] = [None]
            else:
                if cuda_av:
                    samples[name]["Output"]["Attention"] = self.attentions[n][:,:int(self.proteome_lengths[n][0])].numpy()
                else:
                    samples[name]["Output"]["Attention"] = self.attentions[n][:,:int(self.proteome_lengths[n][0])].to(torch.float32).numpy()
            if self.embeddings1 is None:
                samples[name]["Output"]["Embeddings1"] = [None]
            else:
                if cuda_av:
                    samples[name]["Output"]["Embeddings1"] = self.embeddings1[n].numpy()
                else:
                    samples[name]["Output"]["Embeddings1"] = self.embeddings1[n].to(torch.float32).numpy()
            if self.embeddings2 is None:
                samples[name]["Output"]["Embeddings2"] = [None]
            else:
                if cuda_av:
                    samples[name]["Output"]["Embeddings2"] = self.embeddings2[n].numpy()
                else:
                    samples[name]["Output"]["Embeddings2"] = self.embeddings2[n].to(torch.float32).numpy()
        return samples



class MemoryReport:
    """Wrapper around the PyTorch CUDA memory profiler.

    Parameters
    ----------
    results_dir:
        Directory where profile traces and snapshots are written.
    process:
        Label used as a filename prefix for all output files.
    """

    def __init__(self, results_dir, process):
        self.results_dir = results_dir
        self.process = process
        self.prof = None

    def start_memory_reports(self, max_num_events_per_snapshot=1):
        memory_report = "{}/{}_memory-report".format(self.results_dir, self.process)
        torch.cuda.memory._record_memory_history(
            max_entries=max_num_events_per_snapshot)
        self.prof = torch.profiler.profile(
                        schedule=torch.profiler.schedule(wait=0, warmup=0, active=2),
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True, with_stack=True, profile_memory=True)
        self.prof.start()

    def step(self):
        self.prof.step()

    def stop_memory_reports(self):
        if self.prof is not None:
            self.prof.stop()
            logger.debug("CUDA memory summary:\n%s", torch.cuda.memory_summary())
            self.prof.export_chrome_trace("{}/{}_memory-record.json".format(self.results_dir, self.process))
            logger.debug("Profiler key averages:\n%s", self.prof.key_averages().table())
            torch.cuda.memory._dump_snapshot("{}/{}_memory-record.pkl".format(self.results_dir, self.process))
            torch.cuda.memory._record_memory_history(enabled=None)
        else:
            raise ValueError("Profiler has not been started")

class FileReport:
    """Serialise training metrics to a pickle file.

    Parameters
    ----------
    configuration:
        Model parameter dict stored in the report for reproducibility.
    name:
        Base filename (without extension) for the output pickle.
    dict_dir:
        Directory where the pickle is written.
    modes:
        Pipeline mode (``"Train"``, ``"Prediction"``, or ``"Test"``).
    """

    def __init__(self, configuration, name, dict_dir, modes):

        self.dict_file = "{}/{}".format(dict_dir, name)
        self.data = {}
        self.data["Configuration"] = configuration
        if modes == "Train":
            self.data["Train"] = {}
        elif modes == "Prediction":
            self.data["Prediction"] = {}
        elif modes == "Test":
            self.data["Test"] = {}

    def start_train_report(self, model, criterion) -> None:
        pass

    def add_step_info(self, loss_train, lr, batch_n, len_dataloader) -> None:
        pass

    def add_epoch_info(self, log_results):
        self.data["Train"]["Epoch {}".format(log_results["Epoch"])] = log_results

    def finish_report(self):
        with open("{}_results.pickle".format(self.dict_file), "wb") as f:
            pickle.dump(self.data, f)



class WandbReport:
    """Stream training metrics to Weights & Biases.

    Requires ``wandb`` to be installed (``pip install pathogenfinder2[wandb]``).
    The ``wandb`` import is deferred to construction time so the rest of the
    package can be imported without it.

    Parameters
    ----------
    configuration:
        Model parameter dict logged to the W&B run config.
    project:
        W&B project name.
    name:
        Optional run name.
    wandb_dir:
        Local directory for W&B run artefacts.
    """

    batch_checkpoint = 30

    def __init__(self, configuration, project, name="", wandb_dir=None):
        import wandb
        self.wandb_run = wandb.init(project=project, name=name,
                                config=configuration, dir=wandb_dir)


    def start_train_report(self, model, criterion, log="all"):
        self.wandb_run.watch(model, criterion, log=log, log_freq=1)
        model_params = self.count_params(model)
        self.wandb_run.summary["Trainable parameters"] = model_params
        self.step_wandb = 0
        self.epoch = 0

    def count_params(self, model):
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return pytorch_total_params

    def add_epoch_info(self, log_results):
        import wandb
        self.epoch = log_results["Epoch"]
        wandb.log(log_results, step=self.step_wandb)

    def add_step_info(self, loss_train, lr, batch_n, len_dataloader):
        import wandb
        if batch_n % WandbReport.batch_checkpoint == 1 and self.wandb_run:
            wandb.log({"Training Loss/Step": loss_train, "Learning Rate": lr, "Epoch": self.epoch + ((batch_n+1)/len_dataloader)}, step=self.step_wandb)
            self.step_wandb += 1

    def finish_report(self):
        self.wandb_run.finish()

    def log_plot(self, fig, name):
        self.wandb_run.log({name: fig})
