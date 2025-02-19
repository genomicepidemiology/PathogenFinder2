import os
import pandas as pd
import numpy as np
from tqdm import tqdm

class Extremophiles_DB:

    range_temp = (36,38)

    def __init__(self, folder="./", range=True):
        thermobase_db = pd.read_csv(
                            "{}/ThermoBase_ver_1.0_2022.csv".format(folder),
                            sep=",", usecols=['Name', 'Taxonomic ID', 'Domain',
                            'Phylum', 'Class', 'Order', 'Family', 'Ecosystem',
                            'Environment', 'Energy Source', 'Metabolism',
                            'Extended Metabolism', 'Ion for Chemiosmosis',
                            'Oxygen Requirement', 'Min. pH', 'Max. pH',
                            'Avg. Opt. pH', 'Min. Temp. (°C)',
                            'Max. Temp. (°C)', 'Avg. Optimum Temp (°C)',
                            'Pressure for Opt. Temp. (kpa)',
                            'Optimum Pressure (Mpa) ',
                            'Avg. Opt. salinity (%)', 'Note', 'Source',
                            'Additional_source'])
        tempura_db = pd.read_csv(
                            "{}/200617_TEMPURA.csv".format(folder),
                            sep=",")
        nasa_db = pd.read_excel(
                            "{}/S1_spreadsheet_REVISION.xlsx".format(folder))
        self.thermobase_db = self.select_bacteria(thermobase_db)
        thermobase_db = self.thermobase_db[(((~self.thermobase_db["Min. Temp. (°C)"].isna())&(~self.thermobase_db["Max. Temp. (°C)"].isna()))|(~self.thermobase_db["Avg. Optimum Temp (°C)"].isna()))]
        self.thermobase_db = self.select_temp(thermobase_db, range=range,
                                minmax=("Min. Temp. (°C)", "Max. Temp. (°C)"), average="Avg. Optimum Temp (°C)")
        tempura_db = self.select_bacteria(tempura_db, col="Superkingdom")
        self.tempura_db = self.select_temp(db=tempura_db, range=range)
        print(self.thermobase_db)
        print(self.tempura_db)


    def select_temp(self, db, range=False, minmax=('Tmin(ºC)', 'Tmax(ºC)'), average="Topt_average(ºC)"):
        if range:
            db_out = db[(db[minmax[1]]< Extremophiles_DB.range_temp[0])|(db[minmax[0]]>Extremophiles_DB.range_temp[1])]

        else:
            m = db[average].between(Extremophiles_DB.range_temp[0],
                                            Extremophiles_DB.range_temp[1])
            db_out = db[~m]
        return db_out


    def select_bacteria(self, db, col="Domain"):
        return db[db[col]=="Bacteria"]

    def add_accession(self, db, refseq_acc, genbank_acc):
        genbank_df = pd.read_csv(genbank_acc, sep="\t", skiprows=[0])
        #print(genbank_df)
        #print(db[db["Assembly_or_accession"].isnull()])
        accessions_lst = []
        for index, row in tqdm(db.iterrows()):
            if not pd.isna(row["Assembly_or_accession"]):
                accession = str(row["Assembly_or_accession"])
                accessions_lst.append(accession)
                continue
            species_subset = genbank_df[genbank_df["organism_name"]==str(row["Genus_and_species"])]
            strain_subset = species_subset[species_subset["infraspecific_name"].str.contains(str(row["Strain"]), na=False, regex=False)]
            if strain_subset.empty:
                accession = None
            elif len(strain_subset) == 1:
                accession = str(species_subset.loc[species_subset["infraspecific_name"].str.contains(str(row["Strain"]), na=False),"# assembly_accession"].values[0])
            else:
                accession = species_subset[species_subset["infraspecific_name"].str.contains(str(row["Strain"]), na=False)]["# assembly_accession"].values.tolist()
            if accession is not None and not isinstance(accession, list) and not accession.startswith("GCA"):
                print("WTF", row)
            accessions_lst.append(accession)
        db["Accession Genbank"] = accessions_lst
        db = db[db["Accession Genbank"]!=None]
        return db

    def join_db(self, refseq_acc, genbank_acc, save_file):
        self.tempura_db["Name"] = self.tempura_db["Genus_and_species"] + " " + self.tempura_db["Strain"]
        self.tempura_db.rename(columns={
            "Taxonomy_ID": "Taxonomic ID", "Superkingdom": "Domain",
            "Tmin(ºC)":'Min. Temp. (°C)', "Tmax(ºC)":'Max. Temp. (°C)',
            "Topt_average(ºC)": 'Avg. Optimum Temp.'}, inplace=True)
        final_db = pd.concat([self.tempura_db, self.thermobase_db])
        final_db = self.add_accession(final_db, refseq_acc, genbank_acc)
        final_db.to_csv(save_file, sep=",", index=False)




if __name__ == '__main__':
    instance = Extremophiles_DB()
    instance.join_db(save_file="./extreme_temp.csv",
        refseq_acc="../../computerome_data/assembly_summary_refseq.txt",
        genbank_acc="../../computerome_data/assembly_summary_genbank.txt")
