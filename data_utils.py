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
                 cluster_tsv=None, transform=None, weighted=False,
                 load_data=True, device="cpu"):
        self.landmarks_frame = pd.read_csv(csv_file, sep="\t")
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        if not weighted:
            self.weights = torch.Tensor([1.])
        else:
            weights = ProteomeDataset.get_weights_classes(df=self.landmarks_frame)
            self.weights = torch.Tensor(weights)
        self.load_data = load_data
        self.device = device

    def get_weights(self):
        return self.weights

    @staticmethod
    def get_weights_classes(df, homology_sample=False):
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
        if self.load_data:
            embeddings, length_proteome, protein_names = ProteomeDataset.open_embedfile(file_path)
        else:
            embeddings, length_proteome, protein_names = np.empty((2,2)), 0, None
        pathophenotype = self.landmarks_frame.iloc[idx]["PathoPhenotype"]
        sample = {'Embeddings': embeddings, 'Label': pathophenotype, "Protein Count": length_proteome,
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
            b_proteome_len[i,:] = b["Protein Count"]
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
                "Protein Count": torch.from_numpy(b_proteome_len),
                "PathoPhenotype": torch.from_numpy(b_pathopheno),
                "Protein_IDs": protein_names, "File_Names": file_names
                }

class BucketSampler(Sampler):
    
    def __init__(self, landmarks_frame, batch_size, num_buckets=10, random_buckets=True):
        assert batch_size > num_buckets
        assert len(landmarks_frame) > batch_size*num_buckets
        sort_df = landmarks_frame.sort_values(by=['Protein Count'])
        lens_ind = np.array([sort_df.index.values, sort_df["Protein Count"].values], dtype=np.int32).T
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


if __name__ == '__main__':
    dataset = ProteomeDataset(csv_file="../metadata/METADATA_completeDF_protLim.tsv",
	                root_dir="/work3/alff/embeddings/data/",
			transform=transforms.Compose([PhenotypeInteger(prediction="Single")]), load_data=False)
    bucketing_sampler = BucketSampler(dataset.landmarks_frame, batch_size=64, num_buckets=6)

    load_data = DataLoader(dataset,num_workers=1,
				collate_fn=ProteomeDataset.collate_fn_mask,
                                batch_sampler=bucketing_sampler, persistent_workers=False, pin_memory=False)
    count = 0
    for d in tqdm(load_data):
        print(d["PathoPhenotype"])
    
