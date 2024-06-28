import h5py
import os
import sys
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import numpy as np

sys.path.append("{}/../../..".format(os.path.dirname(os.path.realpath(__file__))))

from dimensionality_reduction.dimensionality_reduct import Dimensionality_Study

class Reshape_Dataset:

    def __init__(self, batch_size, components):

        self.IPCA = IncrementalPCA(n_components=components)
        self.batch_size = batch_size

    def train_IPCA(self, folder_in):

        list_files = os.listdir(folder_in)
        bucket_files = [list_files[i:i+self.batch_size] for i in range(0,len(list_files), self.batch_size)]
        for bucket in tqdm(bucket_files):
            embeddings_lst = []
            for file_name in bucket:
                file_path = "{}/{}".format(folder_in, file_name)
                embeddings = Dimensionality_Study.open_embedfile(file_path)
                embeddings_lst.append(embeddings)
            embeddings_matrix = np.concatenate(embeddings_lst)
            self.IPCA.partial_fit(embeddings_matrix)




if __name__ == '__main__':

    reshape = Reshape_Dataset(batch_size=3, components=450)
    reshape.train_IPCA(folder_in="../../new_demo_data/")



