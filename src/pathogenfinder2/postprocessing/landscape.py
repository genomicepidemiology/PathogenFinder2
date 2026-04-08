"""
UMAP-based Bacterial Pathogenic Landscape (BPL) mapping for PathogenFinder2.

:class:`MapEmbeddings` projects a query proteome embedding into the
pre-computed BPL UMAP space and finds its *k* nearest neighbours from the
reference pathogen set, producing a labelled scatter-plot and a TSV of close
neighbours.
"""
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
    """Project proteome embeddings onto the Bacterial Pathogenic Landscape (BPL).

    On construction the reference embedding NPZ is loaded and a UMAP model is
    fitted to it.  Query embeddings are then transformed with
    :meth:`fit_test_data` and mapped with :meth:`knn`.

    Parameters
    ----------
    out_folder:
        Directory where output files (neighbour TSV, scatter PNG) are written.
    data_embed:
        Path to the reference BPL ``.npz`` file (``embeddings``,
        ``name_refseq``, ``species_name``, ``strain_name``, ``refseq_id``,
        ``taxonomy_id`` arrays expected).
    fitted_model:
        Optional pre-fitted UMAP model.  If provided, ``fitdata()`` is
        skipped and the given model is used directly for transformations.
    train_data:
        Optional pre-computed 2-D reference coordinates.  If provided
        together with *fitted_model*, both fitting and transforming the
        reference set are skipped entirely.
    """

    def __init__(self, out_folder: str, data_embed: str, fitted_model=None, train_data=None):
        self.out_folder = os.path.abspath(out_folder)
        self.data_embed = np.load(data_embed)
        if fitted_model is not None and train_data is not None:
            self.fit_model = fitted_model
            self.train_data = train_data
        elif fitted_model is not None:
            self.fit_model = fitted_model
            self.train_data = self.fit_model.transform(self.data_embed["embedding"])
        else:
            self.train_data, self.fit_model = self.fitdata()

    def fitdata(self) -> tuple[np.ndarray, umap.umap_.UMAP]:
        """Fit a UMAP model to the reference BPL embeddings.

        Returns
        -------
        tuple[np.ndarray, umap.umap_.UMAP]
            The UMAP-transformed reference coordinates and the fitted model.
        """
        train_embed = self.data_embed["embedding"]
        fit_model = umap.UMAP(random_state=42, n_neighbors=300, min_dist=0.5, n_jobs=1).fit(train_embed)
        train_umap = fit_model.transform(train_embed)
        return train_umap, fit_model

    def fit_test_data(self, testdata: str) -> np.ndarray:
        """Transform a query embedding NPZ into the BPL UMAP space.

        Parameters
        ----------
        testdata:
            Path to a ``.npz`` file produced by the neural-network ensemble
            (must contain an ``embeddings_1`` array).

        Returns
        -------
        np.ndarray
            Shape ``(1, 2)`` UMAP coordinates for the query proteome.
        """
        test_npz = np.load(testdata)
        embeddingstest = np.expand_dims(test_npz["embeddings_1"].flatten(), axis=0)
        test_trans = self.fit_model.transform(embeddingstest)
        return test_trans

    def knn(self, test_data: np.ndarray, k: int = 10,
            metric: str = "minkowski") -> tuple[pd.DataFrame, np.ndarray]:
        """Find the *k* nearest reference neighbours of *test_data*.

        Parameters
        ----------
        test_data:
            UMAP coordinates of the query proteome (output of :meth:`fit_test_data`).
        k:
            Number of neighbours to retrieve.
        metric:
            Distance metric passed to :class:`sklearn.neighbors.NearestNeighbors`.

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            A DataFrame of neighbour metadata and their UMAP coordinates.
        """
        nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(self.train_data)
        distances, indices = nbrs.kneighbors(test_data)
        names = self.data_embed['name_refseq'][indices].flatten()
        species = self.data_embed['species_name'][indices].flatten()
        strain = self.data_embed['strain_name'][indices].flatten()
        refseq = self.data_embed['refseq_id'][indices].flatten()
        tax = self.data_embed['taxonomy_id'][indices].flatten()
        closer_df = pd.DataFrame({"Names": names, "Species": species, "Strain": strain, "RefSeq": refseq,
                                  "Taxonomy": tax, "Distances": distances.flatten()})
        closer_df = closer_df.astype({'Names': str, "Species": str, "Strain": str, "RefSeq": str,
                                      "Taxonomy": "int32", "Distances": "float32"})
        closer_df.to_csv("{}/closeneighbors_bpl.tsv".format(self.out_folder), sep="\t", index=False)
        closer_arr = np.squeeze(self.train_data[indices])
        return closer_df, closer_arr

    def make_graph(self, test_data: np.ndarray, closer_data: np.ndarray,
                   add_sp: bool = False) -> None:
        """Save a UMAP scatter plot of the BPL with the query proteome highlighted.

        Parameters
        ----------
        test_data:
            UMAP coordinates of the query proteome.
        closer_data:
            UMAP coordinates of the *k* nearest neighbours.
        add_sp:
            If ``True``, annotate major pathogen clades on the plot.
        """
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)
        fig, ax = plt.subplots()
        sns.scatterplot(x=self.train_data[:, 0],
                        y=self.train_data[:, 1],
                        color="darkred",
                        s=2,
                        alpha=1, label="Pathogens")
        sns.scatterplot(x=closer_data[:,0],
                        y=closer_data[:,1], label="Closer Pathogens",
                        color="black", marker="x", s=10)
        sns.scatterplot(x=test_data[:,0],
                        y=test_data[:,1], label="Your Sequence",
                        color="gold", marker="D", s=20)
        if add_sp:
            ax.annotate("Helicobacter pylori",
                xy=(-5.5, 9), xycoords='data',
                xytext=(-7, 14), textcoords='data',fontstyle="italic", color="#3e3e3e",
                arrowprops=dict(arrowstyle="-[,widthB=3,lengthB=0.1,angleB=60", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Escherichia coli, Klebsiella spp,\nEnterobacter spp., Citrobacter spp.\nSerratia spp., Salmonella spp.",
                xy=(5.5, 14.5), xycoords='data',
                xytext=(-3, 16), textcoords='data',fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-[,widthB=4.5,lengthB=0.1,angleB=-5", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Providencia spp.",
                xy=(12, 8), xycoords='data',
                xytext=(13, 13), textcoords='data',fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Campylobacter spp.",
                xy=(1.5, -6.25), xycoords='data',
                xytext=(-7, -8), textcoords='data',fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Burkholderia spp.",
                xy=(4.5, -3.5), xycoords='data',
                xytext=(-6, -5.5), textcoords='data',fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Streptococcus pneumoniae",
                xy=(2, -1.75), xycoords='data',
                xytext=(-7, -1.5), textcoords='data',fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Mycobacterium spp.,\nNocardia farcinica,\nPrescottella equi",
                xy=(5, -1), xycoords='data',
                xytext=(-6, 1), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Haemophilus spp.",
                xy=(4, -8), xycoords='data',
                xytext=(-3, -10), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Acinetobacter spp.",
                xy=(4, 2.5), xycoords='data',
                xytext=(-1.5, 9), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-[,widthB=2.5,lengthB=0.2,angleB=5", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Pseudomonas spp.",
                xy=(10.5, 4.75), xycoords='data',
                xytext=(7, 7.5), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Aeromonas spp.",
                xy=(9, 3), xycoords='data',
                xytext=(5, 5.5), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Vibrio spp.",
                xy=(7, 0.5), xycoords='data',
                xytext=(2.5, 7.5), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Neisseria meningitidis",
                xy=(7.5, -1.1), xycoords='data',
                xytext=(-3, -3.5), textcoords='data', fontstyle="italic",va="center",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Streptococcus spp.",
                xy=(7.5, -10), xycoords='data',
                xytext=(3, -12), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-[,widthB=4,lengthB=0.2,angleB=50", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Staphylococcus spp.",
                xy=(10.5, -6), xycoords='data',
                xytext=(14.5, -9), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-[,widthB=2.5,lengthB=0.2,angleB=-60", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Enterococcus spp.",
                xy=(12, -8), xycoords='data',
                xytext=(12, -11), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-[,widthB=3,lengthB=0.2,angleB=-50", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Listeria monocytogenes",
                xy=(13, -6.25), xycoords='data',
                xytext=(17, -7.5), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Campylobacter concisus",
                xy=(10.75, -3.25), xycoords='data',
                xytext=(13, 6), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5, relpos=(0, 0)), fontsize=6)
            ax.annotate("Clostridium spp.",
                xy=(15, -2.75), xycoords='data',
                xytext=(16, -4.5), textcoords='data', fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-[,widthB=2,lengthB=0.2,angleB=25", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Porphyromonas gingivalis,\nPrevotella intermedia",color="#3e3e3e",
                xy=(13, 0.5), xycoords='data',
                xytext=(17, -3), textcoords='data', fontstyle="italic",
                arrowprops=dict(arrowstyle="-", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5), fontsize=6)
            ax.annotate("Bacteroides spp.",
                xy=(14.75, 2.75), xycoords="data",
                xytext=(16, 4), textcoords="data", fontstyle="italic",color="#3e3e3e",
                arrowprops=dict(arrowstyle="-[,widthB=2.5,lengthB=0.2,angleB=50", connectionstyle="arc3", color="#3e3e3e", linewidth=0.5,
                                ), fontsize=6)

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
    test_transf = mapemb.fit_test_data(testdata=args.embedding_test)
    closer_df, closer_arr = mapemb.knn(test_transf)
    mapemb.make_graph(test_data=test_transf, closer_data=closer_arr)


if __name__ == "__main__":
    main()
