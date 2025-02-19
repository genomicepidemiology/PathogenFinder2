import pandas as pd
import os
#import matplotlib.pyplot as plt
#import seaborn as sns
import re
import argparse
import numpy as np
from nltk.corpus import stopwords
import datetime



class PathogenNCBI_data:

    def __init__(self, data="./isolates.tsv", sep="\t"):
        self.data = pd.read_csv(data, sep=sep)

    def resume_species(self, file):
        df = pd.read_csv(file, sep=",")
        file_map = "../silva/tax_slv_ssu_138.1_bacteria.txt"
        dict_map = {}
        counts = df["#Organism group"].value_counts().rename_axis('species').reset_index(name='counts')
        with open(file_map, "r") as file_op:
            for line in file_op:
                line_split = line.strip().split("\t")
                print(line_split)
                if line_split[-3] == "genus":
                    name_genus = line_split[0].split(";")[-2]
                    dict_map[name_genus] = line_split[0].replace(";", " ")[:-1]
        list_map = []
        for index, row in counts.iterrows():
            genus = row["species"].split(" ")[0]
            print(row["species"])
            if genus == "E.coli":
                list_map.append(dict_map["Escherichia-Shigella"])
            elif genus == "Burkholderia":
                list_map.append(dict_map["Burkholderia-Caballeronia-Paraburkholderia"])
            elif genus == "Kluyvera_intermedia":
                list_map.append(dict_map["Kluyvera"])
            elif genus == "Phytobacter":
                list_map.append("Bacteria Proteobacteria Gammaproteobacteria Enterobacterales Enterobacteriaceae")
            elif genus == "Candida":
                list_map.append(np.NaN)
            else:
                list_map.append(dict_map[genus])
        counts["Lineage"] = list_map

        counts.to_csv("./species_hp.csv", index=False)

    @staticmethod
    def barplot_specieshost(data):
        data_df = pd.read_csv(data, sep="\t")

        print(data_df["Host"].value_counts())

    @staticmethod
    def read_names(file_name, file_node):
        df_name = pd.read_csv(file_name, sep="|", names=["tax_id", "name_txt",
                        "unique name", "name class", "SUP"])
        df_name.replace({"\t": ""}, inplace=True, regex=True)
        print(df_name)
        df_nodes = pd.read_csv(file_node, sep="|",
                        names=["tax_id", "parent tax_id", "rank", "embl code",
                            "division id", "inherited div", "genetic code id",
                            "inherited GC", "mitochondrial genetic code id",
                            "inherited MGC", "GenBank hidden",
                            "hidden subtree root","comments", "SUP"])
        df_nodes.replace({"\t": ""}, inplace=True, regex=True)
        return df_name, df_nodes

    def select_fromdate(self, start_date):
        self.data["year_col"]=None
        self.data["month_col"]=None
        self.data["day_col"] = None
        self.data["year_col"] = self.data["Create date"].str.split("T").str[0].str.split("-").str[0]
        self.data["month_col"] = self.data["Create date"].str.split("T").str[0].str.split("-").str[1]
        self.data["day_col"] = self.data["Create date"].str.split("T").str[0].str.split("-").str[2]
        self.data["year_col"] = self.data["year_col"].astype(int)
        self.data["month_col"] = self.data["month_col"].astype(int)
        self.data["day_col"] = self.data["day_col"].astype(int)
        self.data = self.data[self.data["year_col"]>=int(start_date.year)]
        if start_date.month != 1:
            self.data = self.data[self.data["month_col"]!=None]
            self.data = self.data[self.data["month_col"]>=int(start_date.month)]
        if start_date.day != 1:
            self.data = self.data[self.data["day_col"]!=None]
            self.data = self.data[self.data["day_col"]>=int(start_date.day)]
        del self.data["year_col"]
        del self.data["month_col"]
        del self.data["day_col"]


    def save_df(self, out_file):
        col = "Host"
        indices_1 = self.data[col].dropna()[self.data[col].dropna().str.contains("Human", case=False, na=False)].index
        indices_2 = self.data[col].dropna()[(self.data[col].dropna().str.contains("Homo", case=False, na=False)) & (self.data[col].dropna().str.contains("Sapiens", case=False, na=False))].index
        index_final = indices_1.union(indices_2)
        data = self.data.loc[index_final]
        data["Label"] = "HP"
        data["Origin_DB"] = "PathogenNCBI"
        data.to_csv(out_file, sep=",", index=False)

    def graphs(self):
        print(list(self.data))
        columns_interest = ["Host disease", "Location", "Isolation source",
                "Isolation type", "#Organism group"]
        for col in columns_interest:
            data = self.data[col]
            data_clean = data[~data.str.contains("not ", case=False, na=False)]

            data_ = data_clean.str.lower().value_counts().head(40)
            plt.figure(num=None, figsize=(20,18), dpi=80, facecolor='w',
                    edgecolor='r')
            ax = sns.barplot(y=data_.index, x=data_)
            ax.yaxis.set_tick_params(labelsize = 15)
            ax.set_title(col)
            ax.set_xlabel("Count")
            plt.tight_layout()
            plt.savefig("./graphs/ncbi{}.png".format(col))
            plt.close()

def get_arguments():
    parser = argparse.ArgumentParser(description='Clean NCBIPathogen metadata')
    parser.add_argument('-i','--in_metadataFile', help='Input metadata', required=True)
    parser.add_argument('-o','--out_metadataFile', help='Output metadata', required=True)
    parser.add_argument('--from_date', help="From Date", default=False, type=datetime.date.fromisoformat)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    a = PathogenNCBI_data(data=args.in_metadataFile)
    if args.from_date:
        a.select_fromdate(start_date=args.from_date)
    a.save_df(out_file=args.out_metadataFile)
