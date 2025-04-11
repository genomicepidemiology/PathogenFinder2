import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
from sklearn.neighbors import NearestNeighbors

import argparse

class MapEmbeddings:

    def __init__(self, out_folder:str, data_embed:str):
        self.out_folder = os.path.abspath(out_folder)
        self.data_embed = np.load(data_embed)    
        self.train_data, self.fit_model = self.fitdata()

    def fitdata(self) -> (np.array, umap.umap_.UMAP):
        train_embed = self.data_embed["embedding"]
        fit_model = umap.UMAP(random_state=42, n_neighbors=300, min_dist=0.5, n_jobs=1).fit(train_embed)
        train_umap = fit_model.transform(train_embed)
        return train_umap, fit_model
    
    def fittestdata(self, testdata:str) -> np.array:
        test_npz = np.load(testdata)
        embeddingstest = np.expand_dims(test_npz["embeddings_1"].flatten(), axis=0)
        test_trans = self.fit_model.transform(embeddingstest)
        return test_trans

    def knn(self, test_data:np.array, k:int=10, metric:str="minkowski") -> (pd.DataFrame, np.array):
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(self.train_data)
        distances, indices = nbrs.kneighbors(test_data)
        names = self.data_embed['name_refseq'][indices].flatten()
        species = self.data_embed['species_name'][indices].flatten()
        strain = self.data_embed['strain_name'][indices].flatten()
        refseq = self.data_embed['refseq_id'][indices].flatten()
        tax = self.data_embed['taxonomy_id'][indices].flatten()
        closer_df = pd.DataFrame({"Names": names, "Species": species, "Strain": strain, "RefSeq": refseq,
                                  "Taxonomy": tax, "Distances": distances.flatten()})
        closer_df = closer_df.astype({'Names': str, "Species": str, "Strain": str, "RefSeq": str, "Taxonomy": "int32", "Distances":"float32"})

        closer_df.to_csv("{}/closeneighbors_bpl.tsv".format(self.out_folder), sep="\t", index=False)
        closer_arr = np.squeeze(self.train_data[indices])
        return closer_df, closer_arr

    def make_graph(self, test_data:np.array, closer_data):
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        fig, ax = plt.subplots()
        sns.scatterplot(x=self.train_data[:, 0],
                        y=self.train_data[:, 1],
                        c="darkred",
                        s=2,
                        alpha=1, label="Pathogens")
        sns.scatterplot(x=closer_data[:,0],
                        y=closer_data[:,1], label="Closer Pathogens",
                        c="black", marker="x", s=10)
        sns.scatterplot(x=test_data[:,0],
                        y=test_data[:,1], label="Your Sequence",
                        c="gold", marker="D", s=20)
        ax.legend()
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("UMAP2")
        plt.xlabel("UMAP1")
        plt.savefig("{}/mapped_bpl.png".format(self.out_folder), dpi=800)
        plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='Mapping your sequence to the Bacterial Pathogenic Landscape')
    parser.add_argument('--embedding_train', help='Training data')
    parser.add_argument('--embedding_test', help='Test data', required=True)
    parser.add_argument("--out_folder", help='Folder where to output results', required=True)
    return parser.parse_args()

def main():
    args = get_args()
    mapemb = MapEmbeddings(out_folder=args.out_folder, data_embed=args.embedding_train)
    test_transf = mapemb.fittestdata(testdata=args.embedding_test)
    closer_df, closer_arr = mapemb.knn(test_transf)
    mapemb.make_graph(test_data=test_transf, closer_data=closer_arr)

if __name__ == "__main__":
    main()
