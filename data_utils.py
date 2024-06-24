import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from tqdm import tqdm
import h5py
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

class PhenotypeInteger(object):
    Patho_Predict = {"Single":
                        {"No Pathogenic": np.array([0], dtype=int),
                         "Pathogenic": np.array([1], dtype=int)},
                     "Dual":
                         {"No Pathogenic": np.array([0,1], dtype=int),
                          "Pathogenic": np.array([1,0], dtype=int)}
                    }
    def __init__(self, prediction):
        if prediction == "Single":
            self.pathointeger = PhenotypeInteger.Patho_Predict["Single"]
        elif prediction == "Dual":
            self.pathointeger = PhenotypeInteger.Patho_Predict["Dual"]
        else:
            raise ValueError("Only 'Single' and 'Dual' modes are available")

    def __call__(self, sample):
        pathophenotype = sample["Label"]
        patho_int = np.array([self.pathointeger[pathophenotype]], dtype=np.float32)
        sample["Label"] = patho_int
        return sample

class FractionEmbeddings(object):

    def __init__(self, fraction_str, split_str="-"):
        split_fr = fraction_str.split(split_str)
        assert len(split_fr) == 2
        start_fr, end_fr = split_fr
        self.start_fr = int(start_fr)
        self.end_fr = int(end_fr)
    def __call__(self, sample):
        embeddings = sample["Embeddings"]
        embeddings = embeddings[:,self.start_fr:self.end_fr]
        sample["Embeddings"] = embeddings
        return sample


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

    Dim_Embedding = 1024
    Patho_Predict = {"Single":
			{"No Pathogenic": np.array([0], dtype=int),
	                 "Pathogenic": np.array([1], dtype=int)},
		     "Dual":
			 {"No Pathogenic": np.array([0,1], dtype=int),
	                  "Pathogenic": np.array([1,0], dtype=int)}
		    }

    def __init__(self, csv_file, root_dir, sampling=False,
                 cluster_tsv=None, transform=None, weighted=False):
        self.landmarks_frame = pd.read_csv(csv_file, sep="\t")
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
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
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.landmarks_frame.iloc[idx]["File_Embedding"]
        file_path = os.path.join(self.root_dir, file_name)
        embedings, length_proteome, protein_names = ProteomeDataset.open_embedfile(file_path)
        pathophenotype = self.landmarks_frame.iloc[idx]["PathoPhenotype"]
        sample = {'Embeddings': embedings, 'Label': pathophenotype, "Proteome_Length": length_proteome,
                  "Protein_IDs": protein_names, "File_Name": file_name}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    @staticmethod
    def open_embedfile(file):
        numpy_type = np.float32
        with h5py.File(file, "r") as f:
            if "Embeddings" in f:
                len_proteome = np.zeros(1, dtype=np.int32)
                len_proteome[:] = f.attrs["Amount Embeddings"]
            else:
                len_proteome = np.array([len(f.keys())], dtype=np.int32)

            shape=(len_proteome[0], ProteomeDataset.Dim_Embedding)
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

    @staticmethod
    def collate_fn_mask(batch):
        '''
        Padds batch of variable length

        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        ## get sequence lengths
        length_batch = len(batch)
        b_proteome_len = np.zeros((length_batch, 1), dtype=np.int32)
        b_pathopheno = np.zeros((length_batch, 1), dtype=np.float32)
        protein_names = []
        file_names = []

        embedding_lst = []

        for i, b in enumerate(batch):
            b_proteome_len[i,:] = b["Proteome_Length"]
            b_pathopheno[i,:] = b["Label"]
            file_names.append(b["File_Name"])
            protein_names.append(b["Protein_IDs"])

            embedding_lst.append(torch.from_numpy(b["Embeddings"]))
        ## padd
        embeddings = torch.nn.utils.rnn.pad_sequence(embedding_lst, batch_first=True)
        ## compute mask
        masks = torch.ones((length_batch, embeddings.size(1)), dtype=torch.bool)
        for i, seq in enumerate(embedding_lst):
            masks[i, :len(seq)] = 0

        return {"Embeddings": embeddings, "Masks": masks,
                "Proteome_Length": torch.from_numpy(b_proteome_len),
                "PathoPhenotype": torch.from_numpy(b_pathopheno),
                "Protein_IDs": protein_names, "File_Names": file_names
                }

class BucketSampler(Sampler):
    
    def __init__(self, landmarks_frame, batch_size, num_buckets=10, random_buckets=True):
        assert batch_size > num_buckets
        assert len(landmarks_frame) > batch_size*num_buckets
        sort_df = landmarks_frame.sort_values(by=['Proteome_length'])
        lens_ind = np.array([sort_df.index.values, sort_df["Proteome_length"].values], dtype=np.int32).T
        len_df = len(sort_df)
        batch_buck = math.ceil((len_df/batch_size)/num_buckets)
        bucket_ind_l = []
        bucket_batch_l = []
        range_buckets = list(range(num_buckets))
        for n in range(batch_buck):
            range_buckets = list(range(num_buckets))
            bucket_ind_l.append(np.repeat(range_buckets, batch_size))
            bucket_batch_l.extend(range_buckets)
        bucket_ind = np.sort(np.concatenate(bucket_ind_l, axis=None)[:len_df])
        bucket_ind = np.expand_dims(bucket_ind, axis=1)
        self.lens_ind = np.append(lens_ind, bucket_ind, axis=1)
        self.batch_size = batch_size
        self.amount_batches = math.ceil(len_df/batch_size)
        self.bucket_batch = np.sort(np.array(bucket_batch_l)[:self.amount_batches])
        if random_buckets:
            random.shuffle(self.bucket_batch)

    def __iter__(self):
        selected_ind = []
        for n in range(len(self.bucket_batch)):
            bucket = self.bucket_batch[n]
            bucket_ind = self.lens_ind[(self.lens_ind[:,2]==bucket) & (np.in1d(self.lens_ind[:,0], selected_ind, invert=True)),0]
            if len(bucket_ind) >= self.batch_size:
                batch_ind = np.random.choice(bucket_ind, size=self.batch_size, replace=False)
            else:
                batch_ind = bucket_ind
            selected_ind.extend(batch_ind.tolist())
            yield batch_ind           

    def __len__(self):
        return self.amount_batches

class SimilaritySampler(Sampler):

    def __init__(self):
        pass
    def __iter__(self):
        pass
    def __len__(self):
        pass

class SimilarityBootstrap(Sampler):

    def __init__(self):
        pass
    def __iter__(self):
        pass
    def __len__(self):
        pass

class ProteomeDataset_Old(Dataset):

    dimension_embedding = 1024
    patho_integer = {"No Pathogenic": np.array([0], dtype=int),
                    "Pathogenic": np.array([1], dtype=int)}
    patho_vec = {"No Pathogenic": np.array([0,1], dtype=int),
                "Pathogenic": np.array([1,0], dtype=int)}

    def __init__(self, csv_file, root_dir, cluster_sampling=False,
                 dual_pred=False, cluster_tsv=None, transform=None, weighted=False,
                 fraction_embeddings=False, bucketing=False):
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
        self.bucketing = bucketing
        print(bucketing)
        if not cluster_sampling:
            self.landmarks_frame = landmarks_frame
            self.cluster_sampling = cluster_sampling
            self.clusters = None
            if bucketing:
                landmarks_frame.sort_values(by="Proteome_length", inplace=True)
                bucket_num = []
                len_buckets = int(len(landmarks_frame)/bucketing)
                init_b = 0
                landmarks_frame["Bucketing"] = 0
                for b in range(bucketing):
                    try:
                        landmarks_frame.iloc[int(init_b):int(init_b)+len_buckets, landmarks_frame.columns.get_loc("Bucketing")] = b
                    except IndexError:
                        landmarks_frame.iloc[int(init_b):len(landmarks_frame), landmarks_frame.columns.get_loc("Bucketing")] = b
                    print(landmarks_frame.iloc[int(init_b):int(init_b)+len_buckets, landmarks_frame.columns.get_loc("Bucketing")])
                    init_b += int(len_buckets)
                print(landmarks_frame)
                self.landmarks_frame = landmarks_frame
                self.buckets = range(bucketing)
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
        
        self.fraction_embeddings = fraction_embeddings

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
        elif self.bucketing:
            bucket_subset = self.landmarks_frame[self.landmarks_frame]
        else:
            idx_repr = idx
        embedding_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx_repr]["File_Embedding"])
        embedings, length_proteome, protein_names = ProteomeDataset.open_embedfile(embedding_name)
        
        if not self.fraction_embeddings:
            embedings = embedings
        elif self.fraction_embeddings == 1:
            embedings = embedings[:,0:347]
        elif self.fraction_embeddings == 2:
            embedings = embedings[:,337:687]
        elif self.fraction_embeddings == 3:
            embedings = embedings[:,677:]
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
            embedding_lst.append(torch.from_numpy(b["Embeddings"]))
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

if __name__ == '__main__':
    dataset = ProteomeDataset(csv_file="/work3/alff/embeddings/metadata/repr_metadataTRAIN90_len.tsv",
	                root_dir="/work3/alff/embeddings/data/",
			transform=transforms.Compose([PhenotypeInteger(prediction="Single")]))
    bucketing_sampler = BucketSampler(dataset.landmarks_frame, batch_size=64, num_buckets=6)

    load_data = DataLoader(dataset,num_workers=1,
				collate_fn=ProteomeDataset.collate_fn_mask,
                                batch_sampler=bucketing_sampler, persistent_workers=False, pin_memory=False)
    count = 0
    for d in tqdm(load_data):
        print(d["Embeddings"].shape)
    
