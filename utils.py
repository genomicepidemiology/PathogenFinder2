import os
import json
from torch import nn
import types
import torch
from datetime import datetime
from torchmetrics.classification import BinaryMatthewsCorrCoef


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

class NNEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, nn.ReLU) or isinstance(obj, nn.LeakyReLU) or isinstance(obj, nn.Tanh) or isinstance(obj, torch.nn.modules.loss.BCEWithLogitsLoss):
            return str(obj)
        if isinstance(obj, types.FunctionType):
            return obj.__name__
        if obj.__class__.__name__ == "type":
            return obj.__name__
        return super(NNEncoder, self).default(obj)

class NNUtils():
    
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    @staticmethod
    def set_results_files(results_dir):

        timestamp = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

        results_dir = "{}_{}".format(results_dir, timestamp)
        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)
        return results_dir

class Metrics:

    @staticmethod
    def calculate_metrics(predictions, labels):
        predictions_binary = np.where(np.array(predictions) > 0.5, 1, 0)
        acc = balanced_accuracy_score(labels, predictions_binary)
        mcc = matthews_corrcoef(labels, predictions_binary)
        return acc, mcc

    @staticmethod
    def calculate_MCC(predictions, labels, device):
        metric = BinaryMatthewsCorrCoef().to(device)
        mcc = metric(predictions, labels)
        return mcc


class Get_Normalization:

    def __init__(self):
        self.sum_dim = np.zeros(1024)
        self.n_prot = 0
        self.sqrt_diff = np.zeros(1024)

    def get_vec_from_data(self, dir_path, metadata=None):
        if metadata is not None:
            metadata_df = pd.read_csv(metadata, sep="\t")
            list_files = metadata_df["File_Embedding"].tolist()
        else:
            list_files = os.listdir(dir_path)
        count = 0
        for file_name in tqdm(list_files):
            file_path = os.path.join(dir_path, file_name)
            if file_path.endswith(".h5") and os.path.isfile(file_path):
                self.get_sum(file_path)
        count = 0
        mean_vec = self.get_mean()
        for file_name in tqdm(list_files):
            file_path = os.path.join(dir_path, file_name)
            if file_path.endswith(".h5") and os.path.isfile(file_path):
                self.get_sqdiff(file_path, mean_vec)

    def save_data(self, name_file="mean_std_REPRtrain"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mean_vec = self.get_mean()
        std_vec = self.get_std()
        np.savez("{}/data/{}.npz".format(current_dir, name_file), mean=mean_vec, std=std_vec)

    def get_sqdiff(self, file_path, mean_vec):
        embeddings, len_proteome, _ = ProteomeDataset.open_embedfile(file_path)
        diff_vec = np.square(embeddings - mean_vec)
        self.sqrt_diff = np.add(self.sqrt_diff, np.sum(diff_vec, axis=0))

    def get_std(self):
        return np.sqrt(self.sqrt_diff/self.n_prot)

    def get_mean(self):
        return self.sum_dim/self.n_prot

    def get_sum(self, file_path):
        embeddings, len_proteome, _ = ProteomeDataset.open_embedfile(file_path)
        self.sum_dim = np.add(self.sum_dim, np.sum(embeddings, axis=0))
        self.n_prot += len_proteome

class EarlyStopping:

    def __init__(self, patience, delta, measure="max"):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.measure = measure
        self.count_stop = 0


    def __call__(self, val_measure):
        if self.best_score is None:
            self.best_score = val_measure
        else:
            if self.measure == "max":
                condition = val_measure < self.best_score + self.delta
            elif self.measure == "min":
                condition = val_measure > self.best_score - self.delta
            else:
                raise ValueError("The measure {} does not exists".format(self.measure))
            if condition:
                self.count_stop += 1
            else:
                self.count_stop = 0
                self.best_score = val_measure
        if self.count_stop > self.patience:
            return True
        else:
            return False
  
        

