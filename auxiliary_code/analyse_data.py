import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
import numpy as np
import random
from scipy.stats import shapiro

sys.path.append("{}/../../..".format(os.path.dirname(os.path.realpath(__file__))))
from dimensionality_reduction.dimensionality_reduct import Dimensionality_Study


def analyse_length_phenotype(df, hist_file):
    sns.histplot(data=df, x="Protein Count", hue="PathoPhenotype")
    plt.title("Pathogenicity related to length")
    plt.xlabel("Proteome Length")
    plt.ylabel("Count")
    plt.savefig(hist_file)
    plt.close()


def analyse_embeddings_input(folder, number_genomes=10):
    list_files = os.listdir(folder)
    random.shuffle(list_files)
    list_embeddings = []
    for file_name in list_files[:number_genomes]:
        file_path = "{}/{}".format(folder, file_name)
        embeddings = Dimensionality_Study.open_embedfile(file_path)
        list_embeddings.append(embeddings)
    embeddings_matrix = np.concatenate(list_embeddings)
    shape_m = embeddings_matrix.shape
    shapiro_res = []
    mean_res = []
    std_res = []
    for n in range(shape_m[1]):
        shapiro_res.append(shapiro(embeddings_matrix[:, n]).statistic)
        mean_res.append(np.mean(embeddings_matrix[:, n]))
        std_res.append(np.std(embeddings_matrix[:, n]))
    print(shapiro_res)
    plt.scatter(mean_res, std_res, s=1)
    plt.title("Each feature of embedding distribution")
    plt.xlabel("Mean")
    plt.ylabel("Standard Deviation")
    plt.savefig("./distribution_embedding.png")
    plt.close()
    plt.hist(shapiro_res, bins=50)
    plt.title("Shapiro test")
    plt.xlabel("Shapiro statistic")
    plt.ylabel("Count")
    plt.xlim(0,1)
    plt.savefig("./shapiro_embedding.png")
    plt.close()



def hist_embeddings(embeddings):
    shape_m = embeddings_matrix.shape
    dimension_list = []
    values_list = []
    for n in range(shape_m[1]):
        dimension_list.extend([n]*shape_m[0])
        values_list.extend(embeddings_matrix[:, n])
    df = pd.DataFrame({"Dimensions": dimension_list, "Values":values_list})
    sns.histplot(df, x="Values", hue="Dimensions", log=True)
    plt.savefig("./hist_embeddings.png")
    plt.close()


        



df = pd.read_csv("../../metadata/METADATA_completeDF_protLim.tsv", sep="\t")
analyse_length_phenotype(df, "./seq_pheno.png")
analyse_embeddings_input("../../../new_demo_data/", number_genomes=1)