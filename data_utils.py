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
from torchmetrics.classification import BinaryMatthewsCorrCoef

import time

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
        embeddings = sample["Input"]
        embeddings = embeddings[:,self.start_fr:self.end_fr]
        sample["Input"] = embeddings
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
        embeddings = sample["Input"]
        norm_embeddings = (embeddings-self.mean_vec)/self.std_vec
        sample["Input"] = norm_embeddings.astype(np.float32)
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

    def __init__(self, csv_file, root_dir, input_type="protein_embeddings",
                 sampling=False, cluster_tsv=None, transform=None, weighted=False,
                 load_data=True):
        self.landmarks_frame = pd.read_csv(csv_file, sep="\t")
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        if not weighted:
            self.weights = torch.Tensor([1.])
        else:
            weights = ProteomeDataset.get_weights_classes(df=self.landmarks_frame)
            self.weights = torch.Tensor(weights)
        self.load_data = load_data
        if input_type in ["protein_embeddings", "proteome_embedding", "protein_count"]:
            self.input_type = input_type
        else:
            raise ValueError("The input type {} is not available".format(input_type))

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
            if self.input_type == "protein_embeddings":
                input_nn, length_proteome, protein_names = ProteomeDataset.open_embedfile(file_path)
            elif self.input_type == "proteome_embedding":
                input_nn, length_proteome, protein_names = ProteomeDataset.open_embedfile(file_path)
                input_nn = np.sum(input_nn, axis=1)
            elif self.input_type == "protein_count":
                try:
                    input_nn = [self.landmarks_frame.iloc[idx]["Protein Count"]]
                    length_proteome = input_nn
                    protein_names = [None]
                except KeyError:
                    _, length_proteome, protein_names = ProteomeDataset.open_embedfile(file_path)
                    input_nn = length_proteome
                input_nn = np.array(input_nn, dtype=np.float32)
        else:
            input_nn, length_proteome, protein_names = np.empty((2,2)), self.landmarks_frame.iloc[idx]["Protein Count"], None
        pathophenotype = self.landmarks_frame.iloc[idx]["PathoPhenotype"]
        sample = {'Input': input_nn, 'Label': pathophenotype, "Protein Count": length_proteome,
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
    def collate_individual(batch):
        '''
        Padds batch of variable length

        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        ## get sequence lengths
        data = {"Input": None, "Masks": None,
                "Protein Count": None,
                "PathoPhenotype": None,
                "Protein_IDs": [], "File_Names": []
                }

        for i, b in enumerate(batch):
            data["Protein Count"] = torch.from_numpy(np.array([b["Protein Count"]]))
            data["PathoPhenotype"] = torch.from_numpy(np.array(b["Label"]))
            data["File_Names"].append(b["File_Name"])
            data["Protein_IDs"].append(b["Protein_IDs"])
            input_arr = np.array([b["Input"]])
#            input_arr = np.concatenate([np.array([b["Input"]]), np.zeros((input_arr.shape[0],1100, input_arr.shape[2]), dtype="float32")], axis=1)
            data["Input"] = torch.from_numpy(input_arr)
            data["Mask"] = torch.zeros(1,len(b["Input"]))

        return data

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
 #       b_pathopheno = np.zeros((length_batch, 1), dtype=np.float32)
        protein_names = []
        file_names = []

        embedding_lst = []
        patho_lst = []

        for i, b in enumerate(batch):
            b_proteome_len[i,:] = b["Protein Count"]
#            b_pathopheno[i,:] = b["Label"]
            patho_lst.append(torch.from_numpy(b["Label"]))
            file_names.append(b["File_Name"])
            protein_names.append(b["Protein_IDs"])

            embedding_lst.append(torch.from_numpy(b["Input"]))
        ## padd
        embeddings = torch.nn.utils.rnn.pad_sequence(embedding_lst, batch_first=True)
        b_pathopheno = torch.cat(patho_lst)
        ## compute mask
        masks = torch.ones((length_batch, embeddings.size(1)), dtype=torch.bool)
        for i, seq in enumerate(embedding_lst):
            masks[i, :len(seq)] = 0

        return {"Input": embeddings, "Masks": masks,
                "Protein Count": torch.from_numpy(b_proteome_len),
#                "PathoPhenotype": torch.from_numpy(b_pathopheno),
                "PathoPhenotype": b_pathopheno,
                "Protein_IDs": protein_names, "File_Names": file_names
                }

class BucketSampler(Sampler):
    
    def __init__(self, landmarks_frame, batch_size, num_buckets=10, random_buckets=True,
                    stratified=True, drop_last=True):

        assert len(landmarks_frame) > batch_size*num_buckets
        self.landmarks_frame = landmarks_frame.sort_values(by=['Protein Count'])
        self.num_buckets = num_buckets
        self.batch_size = int(batch_size)
        if drop_last:
            self.amount_batches = math.floor(len(landmarks_frame)/batch_size)
        else:
            self.amount_batches = math.ceil(len(landmarks_frame)/batch_size)
        if not drop_last:
            raise ValueError("The option for not dropping the last batch has not been implemented yet.")
            self.batchable_data = len(self.landmarks_frame)
        else:
            self.batchable_data = len(self.landmarks_frame) - len(self.landmarks_frame)%self.batch_size
            self.drop_last = drop_last
        self.stratified = stratified

    def get_bucket_vector(self, batch_size, len_vector, drop_last=True):
        range_buckets = list(range(self.num_buckets))
        random.shuffle(range_buckets)
        bucket_nums = np.repeat(range_buckets, math.ceil(self.amount_batches/self.num_buckets))
        random.shuffle(bucket_nums)
        return np.sort(bucket_nums[:self.amount_batches])


    def sort_packets(self, b, dict_pheno_count_old, dict_pheno_count, size_packet, index_info, column_packet, bucket_mask=None):
        pheno_keys = list(dict_pheno_count.keys())
        random.shuffle(pheno_keys)
        total_count = sum(dict_pheno_count.values())
        ceil_first = 0
        for pheno_k in pheno_keys:
            if ceil_first == 0:
                num_v = math.ceil((self.batch_size*size_packet*dict_pheno_count[pheno_k])/total_count)
                ceil_first = 1
            else:
                num_v = math.floor(((self.batch_size*size_packet*dict_pheno_count[pheno_k])/total_count))
            prev_num_v = dict_pheno_count_old[pheno_k]
            if bucket_mask is None:
                mask = np.where((index_info[:,-1]==pheno_k))[0][prev_num_v:prev_num_v+num_v]
            else:
                mask = np.where((index_info[:,1]==bucket_mask)&(index_info[:,-1]==pheno_k))[0][prev_num_v:prev_num_v+num_v]
            index_info[mask, column_packet] = b
            dict_pheno_count_old[pheno_k] += num_v
            dict_pheno_count[pheno_k] -= num_v
        return dict_pheno_count_old, dict_pheno_count, index_info

    def create_index_info(self):
        # Delete few examples so dropping last batch
        sel_ind = np.sort(np.random.choice(range(len(self.landmarks_frame)),
                                            size=self.batchable_data,
                                            replace=False))
        subset_landmarks_frame = self.landmarks_frame.loc[sel_ind].sort_values(by=['Protein Count'])
        batches = np.zeros(len(sel_ind), dtype=int)
        if self.stratified:
            vect_pheno = subset_landmarks_frame['PathoPhenotype'].rank(method='dense', ascending=False).astype(int).to_numpy()
        else:
            vect_pheno = np.ones(len(sel_ind), dtype=int)
        ind_buck = np.vstack((subset_landmarks_frame.index.values, np.zeros(len(sel_ind), dtype=int), batches, vect_pheno)).T
        return ind_buck
            
    
    def get_batch_pheno(self, bucket_info):
        a, b = np.unique(bucket_info[:,-1], return_counts=True)
        pheno_count = {A: B for A, B in zip(a, b)}
        return pheno_count

    def get_batch_pheno_init(self, bucket_info):
        dict_phenobatch = {}
        for n in np.unique(bucket_info[:,-1]):
            dict_phenobatch[n] = 0
        return dict_phenobatch

    def __iter__(self):
        index_info = self.create_index_info()
        batch_count = 0
        vector_buckets = self.get_bucket_vector(batch_size=self.batch_size, len_vector=len(index_info), drop_last=self.drop_last) 
        phenobucket_count_init = self.get_batch_pheno_init(bucket_info=index_info)
        phenobucket_count = self.get_batch_pheno(bucket_info=index_info)
        for b in range(self.num_buckets):
            phenobucket_count_init, phenobucket_count, index_info = self.sort_packets(b=b,
                                dict_pheno_count_old=phenobucket_count_init, dict_pheno_count=phenobucket_count,
                                size_packet=sum(vector_buckets==b), index_info=index_info, column_packet=1)
        for b in range(self.num_buckets):
            index_bucket = np.where([index_info[:,1]==b])[1]
            bucket_info = index_info[index_bucket]

            np.random.shuffle(index_bucket)
            phenobatch_count_init = self.get_batch_pheno_init(index_info[index_bucket])
            phenobatch_count = self.get_batch_pheno(index_info[index_bucket])
            for n in range(len(index_bucket)//self.batch_size):
                phenobatch_count_init, phenobatch_count, index_info = self.sort_packets(b=batch_count,
                                dict_pheno_count_old=phenobatch_count_init, dict_pheno_count=phenobatch_count,
                                size_packet=1, index_info=index_info, column_packet=2,
                                bucket_mask=b)
                batch_count += 1
        list_batches = list(np.unique(index_info[:,2]))
        random.shuffle(list_batches)
        for b in list_batches:
            batch_out = index_info[index_info[:,2]==b,0].T
            np.random.shuffle(batch_out)
            yield batch_out

    def __len__(self):
        return self.amount_batches
    
    def get_buckets(self):
        index_info = self.create_index_info()
        vector_buckets = self.get_bucket_vector(batch_size=self.batch_size, len_vector=len(index_info), drop_last=self.drop_last) 
        phenobucket_count_init = self.get_batch_pheno_init(bucket_info=index_info)
        phenobucket_count = self.get_batch_pheno(bucket_info=index_info)
        for b in range(self.num_buckets):
            phenobucket_count_init, phenobucket_count, index_info = self.sort_packets(b=b,
                                dict_pheno_count_old=phenobucket_count_init, dict_pheno_count=phenobucket_count,
                                size_packet=sum(vector_buckets==b), index_info=index_info, column_packet=1)
        for b in range(self.num_buckets):
            ind_buck = index_info[index_info[:,1]==b,0]
            yield self.landmarks_frame.loc[ind_buck]

class NN_Data:
    
    @staticmethod
    def create_dataset(input_type, data_df, data_loc, data_type="train", dual_pred=False, cluster_sample=False,
                        cluster_tsv=None, weighted=False, normalize=False, fraction_embeddings=False):
        transform_data = []

        if normalize:
            transform_data.append(Normalize_Data(normalize))
        else:
            pass

        if fraction_embeddings:
            transform_data.append(FractionEmbeddings(fraction_embeddings))
        else:
            pass

        if dual_pred:
            transform_data.append(PhenotypeInteger(prediction="Dual"))
        else:
            transform_data.append(PhenotypeInteger(prediction="Single"))

        transform_compose = transforms.Compose(transform_data)

        if data_type == "prediction":
            dataset = ProteomeDataset(input_type=input_type, csv_file=data_df, root_dir=data_loc, transform=transform_compose)
        elif data_type == "train":
            if not cluster_sample:
                dataset = ProteomeDataset(input_type=input_type, csv_file=data_df, root_dir=data_loc,
                                        transform=transform_compose, weighted=weighted)
            else:
                dataset = ProteomeDataset(input_type=input_type, csv_file=data_df, root_dir=data_loc,
                            transform=transform_compose, cluster_sampling=cluster_sample,
                            cluster_tsv=cluster_tsv, weighted=weighted)
        else:
            raise ValueError("The data_type {} is not an option (choose between train and val)".format(
                                data_type))
        return dataset

    @staticmethod
    def load_data(data_set, batch_size, num_workers=4, shuffle=True, pin_memory=False,
                  bucketing=None, stratified=False, drop_last=True):
        if batch_size > 1:
            collate_fn = ProteomeDataset.collate_fn_mask
        else:
            collate_fn = ProteomeDataset.collate_individual
        if bucketing:
            bucketing_sampler = BucketSampler(data_set.landmarks_frame, batch_size=batch_size,
                                              num_buckets=bucketing, stratified=stratified, drop_last=True)
            data_loader = DataLoader(data_set, num_workers=num_workers,
                              collate_fn=collate_fn, batch_sampler=bucketing_sampler,
                              persistent_workers=False, pin_memory=pin_memory)
        else:
            data_loader = DataLoader(data_set, batch_size=batch_size, num_workers=num_workers,
                              collate_fn=collate_fn, drop_last=drop_last,
                              shuffle=shuffle, persistent_workers=False, pin_memory=pin_memory)

        return data_loader

        

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
    from collections import Counter
    import matplotlib.pyplot as plt
 #   import seaborn as sns

    def calculate_intradistance(batch):
        distances = []
        range_prot = range(len(batch["Protein Count"]))
        for n1 in range_prot:
            for n2 in range_prot:
                if n1 == n2:
                    continue
                distances.append(abs(int(batch["Protein Count"][n2].item())-int(batch["Protein Count"][n1].item())))
        return np.mean(distances), np.std(distances), np.max(distances), torch.max(batch["Protein Count"]).item()
        

    def batch_graphs(weight_counts, title, filename):
        species = list(range(len(weight_counts["Patho"])))
        fig, ax = plt.subplots()
        bottom = np.zeros(len(species))

        for boolean, weight_count in weight_counts.items():
            p = ax.bar(species, weight_count, label=boolean, bottom=bottom)
            bottom += weight_count

        ax.set_title("{}".format(title))
        ax.legend(loc="lower right")
        plt.ylabel("%")
        plt.savefig("./{}.png".format(filename))
        plt.close()

    def boxplot_distances(distances_batch):
        data = pd.DataFrame(distances_batch, columns=["Method", "Measure","Value"])
        sns.boxplot(data=data, x="Method", y="Value", hue="Measure")
        plt.savefig("./batchinner_distance.png")
        plt.close()
        #print(pd.DataFrame.from_dict(distances_batch[], orient='tight'))
        #print(pd.DataFrame.from_dict(distances_batch, orient='columns'))


 #   ProteomeDataset.open_embedfile("/ceph/hpc/data/d2023d12-072-users/dataset20000_orig/embedding_files/all_files/GCF_003/GCF_003709045.1_ASM370904v1_genomic.h5")
  #  exit()
    batch_size = 64
    distances_batch_old = {"Bucketing":{"Max":[], "Mean":[], "Std":[]},
                        "Bucketing & Stratified":{"Max":[], "Mean":[], "Std":[]},
                        "Standard":{"Max":[], "Mean":[], "Std":[]}}
    distances_batch = []
    dataset = ProteomeDataset(csv_file="/ceph/hpc/data/d2023d12-072-users/dataset20000_orig/metadata/METADATA_valDF_protLim_phageclean.tsv",
	                root_dir="/ceph/hpc/data/d2023d12-072-users/dataset20000_orig/embedding_files/all_files/",
			transform=transforms.Compose([PhenotypeInteger(prediction="Single")]), load_data=False)
    bucketing_sampler = BucketSampler(dataset.landmarks_frame, batch_size=batch_size, num_buckets=12, random_buckets=True,
                                     stratified=True, drop_last=True)
#    weight_counts = {"Patho":[], "Non-patho":[]}
 #   count = 0
#    for d in tqdm(bucketing_sampler.get_buckets()):
 #       counts = dict(Counter(d["PathoPhenotype"].squeeze().tolist()))
  #      weight_counts["Patho"].append(counts["Pathogenic"])
   #     try:
    #        weight_counts["Non-patho"].append(counts["No Pathogenic"])
     #   except KeyError:
      #      weight_counts["Non-patho"].append(0.)
#    batch_graphs(weight_counts, title="Percentatge phenotype per bucket (stratified)", filename="bucket_stratified")
    load_data = DataLoader(dataset,num_workers=1,
				collate_fn=ProteomeDataset.collate_fn_mask, stratified=True,
                                batch_sampler=bucketing_sampler, persistent_workers=False, pin_memory=False)

    weight_counts = {"Patho":[], "Non-patho":[]}
    count = 0
    for d in tqdm(load_data):
        counts = dict(Counter(d["PathoPhenotype"].squeeze().tolist()))
        weight_counts["Patho"].append(100*(counts[1.0]/batch_size))
        try:
            weight_counts["Non-patho"].append(100*(counts[0.0]/batch_size))
        except KeyError:
            weight_counts["Non-patho"].append(0.)
        mean_, std_, max_, b_size = calculate_intradistance(batch=d)
        distances_batch.append(["Bucketing & Stratified","Mean",mean_])
        distances_batch.append(["Bucketing & Stratified","Std",std_])
        distances_batch.append(["Bucketing & Stratified","Max",max_])
        distances_batch.append(["Bucketing & Stratified","Batch Size",b_size])


    batch_graphs(weight_counts, title="Percentatge phenotype per batch (stratified)", filename="batch_stratified")
    batch_size = 64
    dataset = ProteomeDataset(csv_file="../metadata/METADATA_completeDF_protLim.tsv",
	                root_dir="/work3/alff/embeddings/data/",
			transform=transforms.Compose([PhenotypeInteger(prediction="Single")]), load_data=False)
    bucketing_sampler = BucketSampler(dataset.landmarks_frame, batch_size=batch_size, num_buckets=12, random_buckets=False,
                                     stratified=False)
    weight_counts = {"Patho":[], "Non-patho":[]}
    count = 0
    for d in tqdm(bucketing_sampler.get_buckets()):
        counts = dict(Counter(d["PathoPhenotype"].squeeze().tolist()))
        try:
            weight_counts["Patho"].append(counts["Pathogenic"])
        except KeyError:
            weight_counts["Patho"].append(1.)        
        try:
            weight_counts["Non-patho"].append(counts["No Pathogenic"])
        except KeyError:
            weight_counts["Non-patho"].append(0.)
    batch_graphs(weight_counts, title="Percentatge phenotype per bucket (no stratified)", filename="bucket_nostratified")

    load_data = DataLoader(dataset,num_workers=1,
				collate_fn=ProteomeDataset.collate_fn_mask,
                                batch_sampler=bucketing_sampler, persistent_workers=False, pin_memory=False)

    weight_counts = {"Patho":[], "Non-patho":[]}
    count = 0
    for d in tqdm(load_data):
        counts = dict(Counter(d["PathoPhenotype"].squeeze().tolist()))
        try:
            weight_counts["Patho"].append(100*(counts[1.0]/batch_size))
        except KeyError:
            weight_counts["Non-patho"].append(1.)
        try:
            weight_counts["Non-patho"].append(100*(counts[0.0]/batch_size))
        except KeyError:
            weight_counts["Non-patho"].append(0.)
        mean_, std_, max_, b_size = calculate_intradistance(batch=d)
        distances_batch.append(["Bucketing","Mean",mean_])
        distances_batch.append(["Bucketing","Std",std_])
        distances_batch.append(["Bucketing","Max",max_])
        distances_batch.append(["Bucketing","Batch Size",b_size])

    batch_graphs(weight_counts, title="Percentatge phenotype per batch (no stratified)", filename="batch_nostratified")

    load_data = DataLoader(dataset,num_workers=1, batch_size=batch_size,
				collate_fn=ProteomeDataset.collate_fn_mask, persistent_workers=False, pin_memory=False, drop_last=True)
    weight_counts = {"Patho":[], "Non-patho":[]}
    count = 0
    for d in tqdm(load_data):
        counts = dict(Counter(d["PathoPhenotype"].squeeze().tolist()))
        try:
            weight_counts["Patho"].append(100*(counts[1.0]/batch_size))
        except KeyError:
            weight_counts["Non-patho"].append(1.)
        try:
            weight_counts["Non-patho"].append(100*(counts[0.0]/batch_size))
        except KeyError:
            weight_counts["Non-patho"].append(0.)
        mean_, std_, max_, b_size = calculate_intradistance(batch=d)
        distances_batch.append(["Standard","Mean",mean_])
        distances_batch.append(["Standard","Std",std_])
        distances_batch.append(["Standard","Max",max_])
        distances_batch.append(["Standard","Batch Size",b_size])

    boxplot_distances(distances_batch)
    
