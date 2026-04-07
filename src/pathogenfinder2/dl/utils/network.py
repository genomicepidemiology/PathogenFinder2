"""
Neural network training, validation, and inference engine for PathogenFinder2.

:class:`NetworkModule` owns the PyTorch model, the loss function, and the
AMP grad scaler.  It exposes three pass methods:

* :meth:`train_pass` — one full training epoch with gradient accumulation.
* :meth:`validation_pass` — one full validation epoch (no gradients).
* :meth:`predictive_pass` — inference with optional attention / embedding
  capture via forward hooks.
"""
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path
import os
import logging

from torchmetrics.classification import BinaryMatthewsCorrCoef
from pathogenfinder2.dl.utils.reporting import BatchResults


class NetworkModule:

    def __init__(self, model_type, model_parameters, out_folder, device=None, mixed_precision=True,
                    results_module=None, memory_profiler=None, loss_type="bcelogits"):
        
        self.out_folder = out_folder
        if device is None:
            self.device = NetworkModule.get_device()
        else:
            self.device = device
        network = model_type.set_model_params(model_type, model_parameters)
        self.network = network.to(self.device)

        self.mixed_precision = mixed_precision
        self.scaler = torch.amp.GradScaler(self.device, enabled=self.mixed_precision)

        self.results_module = results_module
        self.memory_profiler = memory_profiler
        
        self.loss_type = loss_type
        self.mcc_metric = BinaryMatthewsCorrCoef().to(self.device)
        if self.loss_type == "bcelogits":
            self.loss_function = torch.nn.modules.loss.BCEWithLogitsLoss()
        else:
            self.loss_function = torch.nn.modules.loss.BCELoss()

    def calculate_loss(self, predictions_logit, labels):
        predictions = torch.sigmoid(predictions_logit)
        if self.loss_type == "bce":
            loss = self.loss_function(predictions, labels)
        elif self.loss_type == "bcelogits":
            loss = self.loss_function(predictions_logit, labels)
        else:
            raise KeyError("The loss function {} is not available".format(self.loss_type))
        return predictions, loss

    @staticmethod
    def _trim_and_normalize(pred_tensor, labels_tensor, count):
        """Trim pre-allocated tensors to actual sample count and normalise dual-output mode.

        In dual-output mode the two output columns encode [P(pathogen), P(non-pathogen)].
        Converting to a scalar via (col0 - col1 + 1) / 2 maps the result to [0, 1] so
        that MCC sees a standard binary prediction.
        """
        if pred_tensor.size()[1] == 2:
            pred_tensor = pred_tensor[:count, 0] - pred_tensor[:count, 1]
            pred_tensor = (pred_tensor + 1) / 2
            labels_tensor = labels_tensor[:count, 0] - labels_tensor[:count, 1]
            labels_tensor = (labels_tensor + 1) / 2
        else:
            pred_tensor = pred_tensor[:count]
            labels_tensor = labels_tensor[:count]
        return pred_tensor, labels_tensor

    def load_weights(self, weights):
        self.network.load_state_dict(weights)
        return None

    def load_checkpoint(self, checkpoint):
        self.network.load_state_dict(checkpoint['model_state_dict'])
        metadata = {
            k: checkpoint[k]
            for k in ("optimizer_state_dict", "epoch", "loss", "val_measure")
            if k in checkpoint
        }
        if not metadata:
            return None
        return {
            "Optimizer State": metadata.get("optimizer_state_dict"),
            "Epoch": metadata.get("epoch"),
            "Loss": metadata.get("loss"),
        }

    def load_model(self, weights_path):
        weights = torch.load(weights_path, weights_only=True, map_location=torch.device(self.device))
        if "model_state_dict" in weights:
            model_params = self.load_checkpoint(checkpoint=weights)
        else:
            model_params = self.load_weights(weights=weights)
        return model_params

    def save_model(self, optimizer=None, loss=None, mcc_val=None, epoch=None):
        model_data = {"model_state_dict": self.network.state_dict()}
        if optimizer is not None:
            model_data["optimizer_state_dict"] = optimizer.state_dict()
        if loss is not None:
            model_data["loss"] = loss
        if mcc_val is not None:
            model_data["val_measure"] = mcc_val
        if epoch is not None:
            model_data["epoch"] = epoch
        torch.save(model_data, "{}/weights_model.pt".format(self.out_folder))



    def train_pass(self, train_loader, batch_size, optimizer, results_module, accumulate_gradient=False, asynchronity=False):
        """Run one full training epoch.

        Args:
            train_loader: DataLoader for the training split.
            batch_size: Configured batch size (used for tensor pre-allocation).
            optimizer: :class:`~optimizer_utils.Optimizer` wrapper instance.
            results_module: Reporting object with ``add_step_info()``.
            accumulate_gradient: Number of steps to accumulate before a weight
                update.  ``False`` or ``1`` means update every step.
            asynchronity: If ``True`` use non-blocking device transfers
                (requires ``pin_memory=True`` on the DataLoader).

        Returns:
            Tuple of ``(loss, mcc, lr_rates, optimizer)``.
        """

        self.network.train()

        lr_rate_lst = []
        loss_pass = 0.
        count = 0
        len_dataloader = len(train_loader)
        labels_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device, dtype=int)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device)
        batch_n = 0

        optimizer.optimizer.zero_grad(set_to_none=True)

        for idx, batch in tqdm(enumerate(train_loader)):
            if self.memory_profiler:
                self.memory_profiler.step()
            embeddings = batch["Input"]
            labels = batch["PathoPhenotype"]
            lengths = batch["Protein Count"]
            actual_bs = embeddings.shape[0]
            pos_first, pos_last = count, count + actual_bs
            #  sending data to device
            embeddings = embeddings.to(self.device, non_blocking=asynchronity)
            labels = labels.to(self.device, non_blocking=asynchronity)
            lengths = lengths.to(self.device, non_blocking=asynchronity)
            #  making predictions
            with torch.autocast(device_type=self.device, enabled=self.mixed_precision):
                predictions_logit, attentions = self.network(embeddings, lengths)
                predictions, loss = self.calculate_loss(predictions_logit=predictions_logit,labels=labels)

            if accumulate_gradient:
                loss = loss/accumulate_gradient
            #  computing gradients
            self.scaler.scale(loss).backward()
            if not accumulate_gradient or ((idx + 1) % accumulate_gradient == 0) or (idx + 1 == len_dataloader):
                #  updating weights
                if isinstance(optimizer.optimizer, dict):
                    pass
                else:
                    self.scaler.step(optimizer.optimizer)
                optimizer.optimizer.zero_grad(set_to_none=True)

            lr_rate_lst.append(optimizer.optimizer.param_groups[-1]['lr'])
            self.scaler.update()
            if optimizer.lr_scheduler is not None and optimizer.lr_scheduler.__class__.__name__ == "OneCycleLR":
                optimizer.update_scheduler()
            #  computing loss
            loss_c = loss.detach()
            pred_c = predictions.detach()
            labels = labels.detach()
            labels_tensor[pos_first:pos_last,:] = labels
            pred_tensor[pos_first:pos_last,:] = pred_c

            loss_pass += loss_c
            results_module.add_step_info(loss_train=loss_c, lr=optimizer.optimizer.param_groups[-1]['lr'], batch_n=batch_n,
                                             len_dataloader=len_dataloader)
            count += actual_bs
            batch_n += 1
        loss_pass = loss_pass/batch_n
        pred_tensor, labels_tensor = NetworkModule._trim_and_normalize(pred_tensor, labels_tensor, count)
        mcc_pass = self.mcc_metric(pred_tensor, labels_tensor)
        return loss_pass, mcc_pass, lr_rate_lst, optimizer

    def validation_pass(self, val_loader, batch_size, asynchronity=False):
        """Run one full validation epoch without gradient computation.

        Args:
            val_loader: DataLoader for the validation split.
            batch_size: Configured batch size (used for tensor pre-allocation).
            asynchronity: If ``True`` use non-blocking device transfers.

        Returns:
            Tuple of ``(loss, mcc)``.
        """
        self.network.eval()

        loss_pass = 0.
        count = 0
        len_dataloader = len(val_loader)
        labels_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device, dtype=int)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device)
        batch_n = 0

        with torch.inference_mode():
            for batch in tqdm(val_loader):
                if self.memory_profiler:
                    self.memory_profiler.step()
                embeddings = batch["Input"]
                labels = batch["PathoPhenotype"]
                lengths = batch["Protein Count"]
                actual_bs = embeddings.shape[0]
                pos_first, pos_last = count, count + actual_bs
                #  sending data to device
                embeddings = embeddings.to(self.device, non_blocking=asynchronity)
                labels = labels.to(self.device, non_blocking=asynchronity)
                lengths = lengths.to(self.device, non_blocking=asynchronity)
                #  making predictions
                with torch.autocast(device_type=self.device, enabled=self.mixed_precision):
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

                batch_n += 1
                count += actual_bs
        loss_pass = loss_pass/batch_n
        pred_tensor, labels_tensor = NetworkModule._trim_and_normalize(pred_tensor, labels_tensor, count)
        mcc_pass = self.mcc_metric(pred_tensor, labels_tensor)
        return loss_pass, mcc_pass

    def predictive_pass(self, val_loader, batch_size, asynchronity=False,
                            record_attentions=True, record_embeddings=True):
        """Run inference and optionally capture attention weights and embeddings.

        Embeddings are captured at two points via forward hooks:
        ``classifier.norm_layer`` (post-attention, pre-classifier) and
        ``hook_postatt`` (post-attention identity layer).

        Args:
            val_loader: DataLoader for the inference split.
            batch_size: Configured batch size.
            asynchronity: Non-blocking device transfers.
            record_attentions: Return attention tensors from the model output.
            record_embeddings: Capture intermediate embeddings via hooks.

        Returns:
            List of :class:`~report_utils.Batch_Results` objects, one per batch.
        """

        self.network.eval()

        if record_embeddings:
        
            activation = {}
            def getActivation(name):
                # the hook signature
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            joinlength_hook = self.network.classifier.norm_layer.register_forward_hook(getActivation('norm_layer'))
            postattention_hook = self.network.hook_postatt.register_forward_hook(getActivation("hook_postatt"))

        loss_pass = 0.
        count = 0
        len_dataloader = len(val_loader)

        batch_n = 0
        batches_results = []
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="Predictive pass", unit="batch",
                                dynamic_ncols=True, smoothing=0.05, mininterval=0.2, position=1, leave=False):
                if self.memory_profiler:
                    self.memory_profiler.step()
                file_names = batch["File_Names"]
                protein_ids = batch["Protein_IDs"]
                embeddings = batch["Input"]
                lengths = batch["Protein Count"]

                #  sending data to device
                embeddings = embeddings.to(self.device, non_blocking=asynchronity)
                lengths = lengths.to(self.device, non_blocking=asynchronity)
                #  making predictions
                with torch.autocast(device_type=self.device, enabled=self.mixed_precision):
                    predictions_logit, attentions = self.network(embeddings, lengths)
                    predictions = torch.sigmoid(predictions_logit)

                pred_c = predictions.detach().cpu()
                if record_attentions:
                    attentions = attentions.detach().cpu()
                else:
                    attentions = None
                if record_embeddings:
                    embeddings1 = activation['norm_layer'].detach().cpu().unsqueeze(1)
                    embeddings2 = activation['hook_postatt'].detach().cpu().unsqueeze(1)
                else:
                    embeddings1 = None
                    embeddings2 = None

                batch_results = BatchResults(filenames=file_names, predictions=pred_c,
                                        protIDs=protein_ids, proteome_lengths=lengths.detach().cpu(),
                                        attentions=attentions, embeddings1=embeddings1,
                                        embeddings2=embeddings2)
                batches_results.append(batch_results)

                batch_n += 1
                count += batch_size
        if record_embeddings:
            joinlength_hook.remove()
            postattention_hook.remove()
        return batches_results

    @staticmethod
    def set_sensible_batch_size(batch_size, min_batch_size=8, max_batch_size=45):
        if batch_size < min_batch_size:
            raise ValueError("The batch size {} is smaller than the minimum allowed ({})".format(
                                                                batch_size, min_batch_size))
        min_batch = min_batch_size
        accumulated_gradient = batch_size / min_batch
        pre_min_batch = min_batch
        pre_accumul = accumulated_gradient
        while True:
            if pre_min_batch > max_batch_size:
                break
            elif not pre_accumul.is_integer():
                pass
            elif pre_min_batch == max_batch_size:
                accumulated_gradient = pre_accumul
                min_batch = pre_min_batch
                break
            else:
                accumulated_gradient = pre_accumul
                min_batch = pre_min_batch
            pre_min_batch += min_batch_size
            pre_accumul = batch_size / pre_min_batch
        if accumulated_gradient is None or not accumulated_gradient.is_integer():
            raise ValueError("Batch size {} is not a multiple of 8".format(batch_size))
        return min_batch, accumulated_gradient


    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

