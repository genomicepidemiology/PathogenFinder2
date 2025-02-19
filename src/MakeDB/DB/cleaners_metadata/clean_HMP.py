import pandas as pd
from tqdm import tqdm

class Clean_Microbiomes:

    def __init__(self, folder="./"):

        self.microbiome_db = None
        self.columns_end = ["Accession Number", "Name", "Taxonomy ID",
                                "Species", "Strain", "Origin DB"]

    def join_db(self, db, final_db):
        if final_db is None:
            final_db = db
        else:
            final_db = pd.concat([db, final_db])

    @staticmethod
    def find_name(row, genbank_df, count):
        name_species_ = str(row["Organism Name"]).split(" ")
        name_species = " ".join(name_species_[1:])
        entry = genbank_df[genbank_df["organism_name"].str.contains(name_species, na=False)]
        if len(entry) != 1:
            split_name = row["Organism Name"].split(" ")
            species1 = split_name[0]
            species2 = split_name[1]
            strain = " ".join(split_name[2:])
            entry_ = genbank_df[genbank_df["infraspecific_name"].str.contains(split_name[-1], na=False)]
            entry = entry_[entry_["organism_name"].str.contains(species2, na=False)]
            if len(entry)>1:
                entry_plus = entry[entry["organism_name"].str.contains(species1, na=False)]
                if len(entry_plus) > 1:
                    accession = entry_plus["# assembly_accession"].values.tolist()
                    tax_id = entry_plus["taxid"].values.tolist()
                elif len(entry_plus) < 1:
                    accession = entry["# assembly_accession"].values.tolist()
                    tax_id = entry["taxid"].values.tolist()
                else:
                    accession = entry_plus["# assembly_accession"].values[0]
                    tax_id = entry_plus["taxid"].values[0]
            elif len(entry) == 1:
                accession = entry["# assembly_accession"].values[0]
                tax_id = entry["taxid"].values[0]
            else:
                count += 1
                accession = False
                tax_id = False
        else:
            accession = entry["# assembly_accession"].values[0]
            tax_id = entry["taxid"].values[0]
        return accession, tax_id, count

    def clean_HMP(self, db, genbank_path):
        genbank_df = pd.read_csv(genbank_path, sep="\t", skiprows=[0])
        genbank_df["cut_id"] = genbank_df["wgs_master"].str.split(".").str[0]
        db = db[db["Domain"]=="BACTERIAL"]
        pd.set_option('display.max_columns', None)
        assemblies = []
        taxid_lst = []
        count =0
        #db["Genome Accession"] = None
        #db = db[db["Genbank ID"].isna()]
        for index, row in tqdm(db.iterrows()):
            if pd.isna(row["Genbank ID"]):
                accession, tax_id, count = Clean_Microbiomes.find_name(row, genbank_df, count)
            else:
                #acc_n = str(row["Genbank ID"]) + "." + "\d"
                entry = genbank_df[genbank_df["cut_id"]==str(row["Genbank ID"])]
                #entry = genbank_df[genbank_df["wgs_master"].str.contains(r"\b{}".format(acc_n), regex=True, na=False)]
                if len(entry) > 1:
                    accession, tax_id, count = Clean_Microbiomes.find_name(row, genbank_df, count)
                    #raise ValueError("Too many entries")
                elif len(entry) < 1:
                    accession, tax_id, count = Clean_Microbiomes.find_name(row, genbank_df, count)
                else:
                    accession = entry["# assembly_accession"].values[0]
                    tax_id = entry["taxid"].values[0]
            if accession:
                assemblies.append(accession)
                taxid_lst.append(tax_id)
            else:
                assemblies.append(None)
                taxid_lst.append(None)

        db["Accession"] = assemblies
        db["TaxID"] = taxid_lst
        db["Species name"] = db["Organism Name"].str.split(" ").str[:2].str.join(" ")
        db = db[db["Accession"] != None]
        db.to_csv("./HMP_clean.tsv", sep="\t", index=False)

    def clean_HumGut(self, db, genbank_path):
        pass

    def add_db(self, db_path, out_file, genbank_path=None):
        db = pd.read_csv(db_path)
        hmp_db = self.clean_HMP(db, genbank_path)
        self.microbiome_db = self.join_db(db, self.microbiome_db)

def arguments_hmp():
    parser = argparse.ArgumentParser(description='Clean HMP')
    parser.add_argument('-i','--in_metadataFile', help='Input File', required=True)
    parser.add_argument('-o','--out_metadataFile', help='Out File', required=True)
    parser.add_argument("-g", "--genbank_file", help="Genbank File", required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arguments_hmp()
    instance = Clean_Microbiomes()
    instance.add_db(db_path=args.in_metadataFile, out_file=args.out_metadataFile, genbank_path=args.genbank_file)
