import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR,SequentialLR, OneCycleLR
from tqdm import tqdm
import h5py
import pickle
import gc
import os
import pandas as pd
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        dict_tensor = {}
        for k, v in sample.items():
            try:
                t_v = torch.from_numpy(v)
            except TypeError:
                t_v = v
            dict_tensor[k] = t_v
        return dict_tensor

class Normalize_Data:

    def __init__(self, saved_data=None):
        if saved_data is None:
            self.mean_vec = None
            self.std_vec = None
        else:
            path_saved = os.path.abspath(saved_data)
            self.mean_vec, self.std_vec = self.load_vector(path=path_saved)

    def load_vector(self, path):
        array_vec = np.load(path)
        return array_vec["mean"], array_vec["std"]

    def __call__(self, sample):
        embeddings = sample["Embeddings"]
        norm_embeddings = (embeddings-self.mean_vec)/self.std_vec
        sample["Embeddings"] = norm_embeddings.astype(np.float32)
        return sample
            

class ProteomeDataset(Dataset):

    dimension_embedding = 1024
    patho_integer = {"No Pathogenic": np.array([0], dtype=int),
                    "Pathogenic": np.array([1], dtype=int)}
    patho_vec = {"No Pathogenic": np.array([0,1], dtype=int),
                "Pathogenic": np.array([1,0], dtype=int)}

    def __init__(self, csv_file, root_dir, cluster_sampling=False,
                 dual_pred=False, cluster_tsv=None, transform=None, weighted=False):
        """
        Arguments
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the embeddings.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        landmarks_frame = pd.read_csv(csv_file, sep="\t")
        self.root_dir = root_dir
        self.transform = transform
        if not cluster_sampling:
            self.landmarks_frame = landmarks_frame
            self.cluster_sampling = cluster_sampling
            self.clusters = None
        elif cluster_sampling == "sample" or cluster_sampling == "bootstrap":
            cluster_tsv = pd.read_csv(cluster_tsv, sep="\t",skiprows=[0])
            cluster_tsv["Entry"] = cluster_tsv["#Sample"].str.split(" ").str[0]
            self.cluster_sampling = cluster_sampling
            self.landmarks_frame = pd.merge(landmarks_frame, cluster_tsv, left_on="Entry", right_on="Entry")
            self.clusters = self.landmarks_frame["Cluster"].unique()
        else:
            raise ValueError("{} is not an option for cluter sampling.".format(cluster_sampling))

        if not dual_pred:
            self.dict_patho = ProteomeDataset.patho_integer
        else:
            self.dict_patho = ProteomeDataset.patho_vec
        if not weighted:
            self.weights = torch.Tensor([1.])
        else:
            weights = ProteomeDataset.get_weights_classes(df=self.landmarks_frame,
                                patho_pred=self.dict_patho)
            self.weights = torch.Tensor(weights)

    def get_weights(self):
        return self.weights

    @staticmethod
    def get_weights_classes(df, patho_pred, homology_sample=False):
        counts_patho = df["PathoPhenotype"].value_counts().to_frame()
        weights = [float(counts_patho.loc["No Pathogenic","count"])/float(counts_patho.loc["Pathogenic","count"])]
        return weights

    def __len__(self):
        if not self.cluster_sampling or self.cluster_sampling == "bootstrap":
            return len(self.landmarks_frame)
        else:
            return len(self.clusters)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.cluster_sampling == "sample":
            cluster_subset = self.landmarks_frame[self.landmarks_frame["Cluster"]==self.clusters[idx]]
            idx_repr = np.random.choice(cluster_subset.index, 1, replace=False)[0]
        elif self.cluster_sampling == "bootstrap":
            cluster_num = self.landmarks_frame.iloc[idx]["Cluster"]
            cluster_subset = self.landmarks_frame[self.landmarks_frame["Cluster"]==cluster_num]
            idx_repr = np.random.choice(cluster_subset.index, 1, replace=False)[0]
        else:
            idx_repr = idx
        embedding_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx_repr]["File_Embedding"])
        embedings, length_proteome, protein_names = ProteomeDataset.open_embedfile(embedding_name)
        file_name = self.landmarks_frame.iloc[idx_repr]["File_Embedding"]
        pathophenotype = self.landmarks_frame.iloc[idx_repr]["PathoPhenotype"]

        try:
            patho_int = np.array([self.dict_patho[pathophenotype]], dtype=np.float32)
        except KeyError:
            raise KeyError("The word {} is not part of the scheme".format(pathophenotype))
        sample = {'Embeddings': embedings, 'Pathogen_Label': patho_int, "Length_Proteome": length_proteome,
                  "Protein_Names": protein_names, "File_Name": file_name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def collate_fn_mask(batch):
        '''
        Padds batch of variable length

        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        ## get sequence lengths
        length_batch = len(batch)
        lengths_batch = np.zeros((length_batch, 1), dtype=np.int32)
        patho_batch = np.zeros((length_batch, 1), dtype=np.float32)
        protein_lst = []
        file_lst = []
        embedding_lst = []
        for i, b in enumerate(batch):
            lengths_batch[i,:] = b["Length_Proteome"]
            patho_batch[i,:] = b["Pathogen_Label"]
            embedding_lst.append(b["Embeddings"])
            file_lst.append(b["File_Name"])
            protein_lst.append(b["Protein_Names"])
        ## padd
        embeddings = torch.nn.utils.rnn.pad_sequence(embedding_lst, batch_first=True)
        ## compute mask
        masks = torch.ones((length_batch, embeddings.size(1)), dtype=torch.bool)
        for i, seq in enumerate(embedding_lst):
            masks[i, :len(seq)] = 0
        return {"Embeddings": embeddings, "Masks": masks,
                "Length_Proteome": torch.from_numpy(lengths_batch),
                "Pathogen_Label": torch.from_numpy(patho_batch),
                "Protein_Names": protein_lst, "File_Name": file_lst
                }

    @staticmethod
    def open_embedfile(file):
        numpy_type = np.float32
        with h5py.File(file, "r") as f:
            if "Embeddings" in f:
                len_proteome = np.zeros(1, dtype=np.int32)
                len_proteome[:] = f.attrs["Amount Embeddings"]
            else:
                len_proteome = np.array([len(f.keys())])
            shape=(len_proteome[0], ProteomeDataset.dimension_embedding)
            embeddings = np.zeros(shape, dtype=numpy_type)
            if "Embeddings" in f:
                protein_names = np.empty(len_proteome[0], dtype=object)
                f["Embeddings"].read_direct(embeddings)
                f["Names"].read_direct(protein_names)
            else:
                protein_names = []
                count = 0
                for prot in f.keys():
                    protein_names.append(prot)
                    n1 = np.zeros(1024, dtype=numpy_type)
                    f[prot].read_direct(n1)
                    embeddings[count, :] = n1
                    count += 1        
        return embeddings, len_proteome, protein_names
