import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
import pytorch_warmup as warmup
from tqdm import tqdm
import os
import numpy as np
import sys
from datetime import datetime
from torchvision import transforms
from torch.optim import swa_utils

sys.dont_write_bytecode = True

from data_utils import ProteomeDataset, ToTensor, Normalize_Data, BucketSampler, FractionEmbeddings, PhenotypeInteger
from results_record import Json_Results


class Prediction_NeuralNet:

    def __init__(self, network, configuration, weights, mixed_precision=None, results_dir=None):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, garbage_collection_threshold:0.6'
        torch.cuda.empty_cache()

        self.results_dir = self.set_results_files(results_dir=results_dir)
        if memory_report:
            self.profiler = self.start_memory_reports()
        else:
            self.profiler = None


        self.device = self.get_device()
        print("Training on {}".format(self.device))
        self.mixed_precision = mixed_precision
        self.results_dir = results_dir

    def predict(self, data):
        pass
    
