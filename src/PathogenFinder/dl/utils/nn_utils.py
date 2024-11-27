import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.metrics_utils import Metrics

class Network_Module:

    def __init__(self, model_type, model_parameters, out_folder, device=None, mixed_precision=True,
                    results_module=False, memory_profiler=None, loss_type="bcelogits"):
        
        self.out_folder = out_folder
        if device is None:
            self.device = Network_Module.get_device()
        else:
            self.device = device
        network = model_type.set_model_params(model_type, model_parameters)
        self.network = network.to(self.device)

        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.results_module = results_module
        self.memory_profiler = memory_profiler
        
        self.loss_type = loss_type
        if self.loss_type == "bcelogits":
            self.loss_function = torch.nn.modules.loss.BCEWithLogitsLoss()
        else:
            self.loss_function = torch.nn.modules.loss.BCELoss

        

    def calculate_loss(self, predictions_logit, labels):
        predictions = torch.sigmoid(predictions_logit)
        if self.loss_type == "bce":
            loss = self.loss_function(predictions, labels)
        elif self.loss_type == "bcelogits":
            loss = self.loss_function(predictions_logit, labels)
        else:
            raise KeyError("The loss function {} is not available".format(self.loss))
        return predictions, loss

    def load_weights(self, weights):
        self.network.load_state_dict(weights)
        return None

    def load_checkpoint(self, checkpoint, optimizer=None):
        self.network.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return {"Optimizer": optimizer, "Epoch": epoch, "Loss": loss}

    def load_model(self, weights_path, optimizer=None):
        weights = torch.load(weights_path, weights_only=True)
        if "optimizer_state_dict" in weights:
            model_params = self.load_checkpoint(checkpoint=weights, optimizer=optimizer)
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
        torch.save(model_data, "{}/weights_model.pickle".format(self.out_folder))



    def train_pass(self, train_loader, batch_size, optimizer, results_module, accumulate_gradient=False, asynchronity=False):

        self.network.train()

        loss_lst = []
        lr_rate_lst = []
        loss_pass = 0.
        count = 0
        len_dataloader = len(train_loader)
        labels_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device, dtype=int)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device)
        batch_n = 0

        optimizer.optimizer.zero_grad(set_to_none=True)

        for idx, batch in tqdm(enumerate(train_loader)):
            pos_first, pos_last = count, count+batch_size
            if self.memory_profiler:
                self.memory_profiler.step()
            embeddings = batch["Input"]
            labels = batch["PathoPhenotype"]
            lengths = batch["Protein Count"]
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
            if not accumulate_gradient or ((idx + 1) % accumulate_gradient == 0) or (idx + 1 == len(dataloader)):
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
        return loss_pass, mcc_pass, lr_rate_lst, optimizer

    def validation_pass(self, val_loader, batch_size, asynchronity=False):
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
                if self.memory_profiler:
                    self.memory_profiler.step()
                embeddings = batch["Input"]
                labels = batch["PathoPhenotype"]
                lengths = batch["Protein Count"]
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

        return loss_pass, mcc_pass

    def predictive_pass(self, val_loader, batch_size, asynchronity=False):

        self.network.eval()

        loss_lst = []

        loss_pass = 0.
        count = 0
        len_dataloader = len(val_loader)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device="cpu")
        file_tensor = []
        protID_tensor = []
        att_tensor = []
        batch_n = 0

        with torch.inference_mode():
            for batch in tqdm(val_loader):
                pos_first, pos_last = count, count+batch_size
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
                pred_tensor[pos_first:pos_last,:] = pred_c
                file_tensor.extend(file_names)
                protID_tensor.append(protein_ids)

                attentions = attentions.detach().cpu()
                att_tensor.append(attentions)

                #  clean gpu (maybe unnecessary
                batch_n += 1
                count += batch_size

        if pred_tensor.size()[1] == 2:
            pred_tensor = pred_tensor[:,0] - pred_tensor[:,1]
            pred_tensor = (pred_tensor+1)/2

        return pred_tensor.T.tolist(), file_tensor, protID_tensor, att_tensor

    @staticmethod
    def set_sensible_batch_size(batch_size, min_batch_size=8, max_batch_size=45):
        if batch_size < min_batch_size:
            raise ValueError("The batch size {} is smaller than the minimum allowed ({})".format(
                                                                batch_size, Train_NeuralNet.MIN_BATCH_SIZE))
        min_batch = 0 + min_batch_size
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
