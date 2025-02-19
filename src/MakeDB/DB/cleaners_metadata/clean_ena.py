import pandas as pd
#import gc
#import xml.etree.cElementTree as ET
import subprocess
import re
import numpy as np
import argparse
import datetime

month_map = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7,
            "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}

class Clean_ENA_animal:

    def __init__(self, file_tsv, file_name=None,
                    file_node=None):

        ena_file = pd.read_csv(file_tsv, sep="\t")
        self.ena_file = ena_file[(~ena_file["host"].isna())|(~ena_file["host_scientific_name"].isna())|(~ena_file["host_tax_id"].isna())]

        if file_name is None or file_node is None:
            self.names_df = None
            self.nodes_df = None
        else:
            self.names_df, self.nodes_df = Clean_ENA_animal.read_names(
                                                        file_name, file_node)

    @staticmethod
    def read_names(file_name, file_node):
        if file_name is None:
            df_name = None
        else:
            df_name = pd.read_csv(file_name, sep="|", names=["tax_id", "name_txt",
                            "unique name", "name class", "SUP"])
            df_name.replace({"\t": ""}, inplace=True, regex=True)
            df_name["name_txt_lwr"] = df_name["name_txt"].str.lower()
        if file_node is None:
            df_nodes = None
        else:
            df_nodes = pd.read_csv(file_node, sep="|",
                            names=["tax_id", "parent tax_id", "rank", "embl code",
                            "division id", "inherited div", "genetic code id",
                            "inherited GC", "mitochondrial genetic code id",
                            "inherited MGC", "GenBank hidden",
                            "hidden subtree root","comments", "SUP"])
            df_nodes.replace({"\t": ""}, inplace=True, regex=True)
        return df_name, df_nodes

    @staticmethod
    def get_uniques(data):
        data_counts = data.value_counts().rename_axis('names').reset_index(name='counts')
        return data_counts

    def select_date(self, start_date):
        assert "collection_date" in list(self.ena_file)
        self.ena_file["year_col"] = None
        self.ena_file["month_col"] = None
        self.ena_file["day_col"] = None
        self.ena_file["collection_date"] = self.ena_file["collection_date"].astype(str)
        self.ena_file = self.ena_file.loc[(~self.ena_file["collection_date"].str.contains('/'))&(~self.ena_file["collection_date"].str.contains(' '))&(~self.ena_file["collection_date"].str.contains("missing", case=False))&(~self.ena_file["collection_date"].str.contains("unknown", case=False))&(self.ena_file["collection_date"]!="nan")&(self.ena_file["collection_date"]!="")]
        self.ena_file["date_format"] = self.ena_file["collection_date"].str.split("-").str.len()
        self.ena_file.loc[self.ena_file["date_format"]==3, "day_col"] = self.ena_file.loc[self.ena_file["date_format"]==3, "collection_date"].str.split("-").str[2].str.split("T").str[0].astype('Int64')
        self.ena_file.loc[self.ena_file["date_format"]>=2, "month_col"] = self.ena_file.loc[self.ena_file["date_format"]>=2, "collection_date"].str.split("-").str[1]
        self.ena_file.loc[self.ena_file["date_format"]>=1, "year_col"] = self.ena_file.loc[self.ena_file["date_format"]>=1, "collection_date"].str.split("-").str[0]
        self.ena_file = self.ena_file[self.ena_file["year_col"]!=None]
        self.ena_file = self.ena_file[self.ena_file["year_col"]!=""]
        self.ena_file = self.ena_file[pd.to_numeric(self.ena_file['year_col'], errors='coerce').notnull()]

        self.ena_file.loc[self.ena_file["day_col"]=="", "day_col"] = 0
        self.ena_file["day_col"] = self.ena_file["day_col"].fillna(0)

        self.ena_file.loc[self.ena_file["month_col"]=="", "month_col"] = 0
        self.ena_file["month_col"] = self.ena_file["month_col"].fillna(0)
        self.ena_file["month_col"] = self.ena_file["month_col"].replace(month_map)

        self.ena_file["day_col"] =self.ena_file["day_col"].astype(int)
        self.ena_file["month_col"] =self.ena_file["month_col"].astype(int)
        self.ena_file["year_col"] =self.ena_file["year_col"].astype(int)
        self.ena_file = self.ena_file[self.ena_file["year_col"]>=int(start_date.year)]
        if start_date.month != 1:
            self.ena_file = self.ena_file[self.ena_file["month_col"]!=np.nan]
            self.ena_file = self.ena_file[self.ena_file["month_col"]>=int(start_date.month)]
        if start_date.day != 1:
            self.ena_file = self.ena_file[self.ena_file["day_col"]!=np.nan]
            self.ena_file = self.ena_file[self.ena_file["day_col"]>=int(start_date.day)]    
        del self.ena_file["year_col"]
        del self.ena_file["month_col"]
        del self.ena_file["day_col"]
        del self.ena_file["date_format"]


    def get_humanhosts(self, save_file):
        ind_human = []
        for col in ["host_scientific_name", "host"]:
            ind_human1 = self.ena_file[(
                    self.ena_file["host_scientific_name"].str.contains(r"\bhuman", case=False, na=False, regex=True)
                    )&(~self.ena_file["host_scientific_name"].str.contains(r"\bno", case=False, na=False, regex=True))].index
            ind_human2 = self.ena_file[(
                    self.ena_file["host_scientific_name"].str.contains(r"\bhomo", case=False, na=False, regex=True)
                    )&(self.ena_file["host_scientific_name"].str.contains(r"\bsapiens", case=False, na=False, regex=True)
                    )&(~self.ena_file["host_scientific_name"].str.contains(r"\bno", case=False, na=False, regex=True))].index
            ind_human.extend(ind_human1)
            ind_human.extend(ind_human2)
        ind_human = list(set(ind_human))
        ind_tax = self.ena_file[self.ena_file["host_tax_id"]==9606].index
        ind_human.extend(ind_tax)
        ind_human = list(set(ind_human))
        human_db = self.ena_file.loc[ind_human]
        human_db.to_csv(save_file, sep="\t", index=False)


    def assign_taxtree(self):
        self.ena_file.reset_index(inplace=True)
        self.ena_file["species_hosttxid"] = np.NaN
        self.ena_file["genus_hosttxid"] = np.NaN
        self.ena_file["family_hosttxid"] = np.NaN
        self.ena_file["order_hosttxid"] = np.NaN
        self.ena_file["class_hosttxid"] = np.NaN
        self.ena_file["clade_hosttxid"] = np.NaN
        self.ena_file["kingdom_hosttxid"] = np.NaN
        self.ena_file["superkingdom_hosttxid"] = np.NaN
        self.host_taxid["names"] = pd.to_numeric(self.host_taxid["names"])
        taxid_lst = []
        tree_lst = []
        skip_row = True
        for index, row in self.host_taxid.iterrows():
            print(index, len(taxid_lst), len(self.host_taxid))
            taxid = int(row["names"])
            if pd.isna(taxid):
                tree = {"species":np.NaN, "genus":np.NaN,
                            "family":np.NaN, "order":np.NaN, "class":np.NaN}
            else:
                try:
                    tree = self.get_parenttaxid(taxid=taxid, stop_lvl="superkingdom")
                except TypeError:
                    name_sp = self.ena_file.loc[self.ena_file["host_tax_id"] == taxid, "host_scientific_name"]
                    genus = name_sp.str.split(' ').str[0].unique()
                    if len(genus) == 1:
                        taxid_genus = self.names_df.loc[self.names_df["name_txt_lwr"]==str(genus[0]).lower(), "tax_id"]
                        if len(taxid_genus) < 1:
                            taxid_genus = np.NaN
                        else:
                            taxid_genus = int(taxid_genus)
                        tree = self.get_parenttaxid(taxid=taxid_genus, stop_lvl="superkingdom")
                        tree["species"] = taxid
                    else:
                        raise Error("Problems")
            tree_lst.append(tree)
            taxid_lst.append(taxid)
            self.ena_file.loc[(self.ena_file["host_tax_id"]==int(row["names"])).index, "hosttxid"] = taxid
            for k,v in tree.items():
                self.ena_file.loc[(self.ena_file["host_tax_id"]==row["names"]).index,"{}_hosttxid".format(k)] = v
        self.ena_file.to_csv("./animalenapathogen.csv", index=False, sep="\t")

    def get_parenttaxid(self, taxid, stop_lvl=False):
        if not stop_lvl:
            final_parenttaxid = np.NaN
        else:
            final_parenttaxid = {"species":np.NaN, "genus":np.NaN,
                    "family":np.NaN, "order":np.NaN, "class":np.NaN,
                    "clade":np.NaN, "kingdom":np.NaN, "superkingdom": np.NaN}
        if not pd.isna(taxid):
            while True:
                entry = self.nodes_df[self.nodes_df["tax_id"]==int(taxid)]
                if len(entry) < 1:
                    raise TypeError()
                if int(taxid) == 1:
                    break
                if not stop_lvl:
                    final_parenttaxid = int(entry["parent tax_id"].item())
                    break
                else:
                    if str(entry["rank"].item()) in final_parenttaxid:
                        final_parenttaxid[str(entry["rank"].item())] = int(entry["tax_id"].item())
                    taxid = int(entry["parent tax_id"].item())
                    if str(entry["rank"].item()) == stop_lvl:
                        break
        return final_parenttaxid

    def fill_host(self):
        non_taxid = self.ena_file[self.ena_file["host_tax_id"].isna()]
        host_species = Clean_ENA_animal.get_uniques(data=non_taxid["host"])
        taxids_lst = []
        for index, row in host_species.iterrows():
            specie = str(row["names"]).lower()
            specie = re.sub('\d+', '', specie)
            specie = specie.replace("-", " ")
            if specie in ["not available", "not collected", "wastewater",
                    "not provided", "natural / free-living", "p-trap","p trap",
                    "environmental", "not appicable", "food", "environment",
                    "non-host associated", "water", "meat", "weed", "missing",
                    "dairy farm, faecal", "dairy product", "dairy farm, faecal",
                    "livestock", "'not collected'", "food, pork", "hospital environment",
                    "dairy farm, bootsock", "non-human", "imported food", "none", "unidentified",
                    "not available: to be reported later"]:
                new_taxid = np.NaN
            else:
                if specie in ["homo sapiens sapiens", "homo?sapiens", "homosapiens",
                              "homo_sapiens", "patient", "h sapiens", "homo sapinens",
                                "hospitalized patient", "Homo Sapien"]:
                    specie = "homo sapiens"
                elif specie == "freshwater fish":
                    specie = "fish"
                elif specie == "poephagus grunniens":
                    specie = "bos grunniens"
                elif specie == "apodemus sp":
                    specie = "apodemus"
                elif specie in ["pacific white shrimp (litopenaeus vannamei)", "white shrimp Litopenaeus vannamei"]:
                    specie = "litopenaeus vannamei"
                elif specie == "cornitermes sp.":
                    specie = "cornitermes"
                elif specie == "non human primate":
                    specie = "primate"
                elif specie == "feline":
                    specie = "felidae"
                elif specie == "greenfinch":
                    specie = "European greenfinch"
                elif specie in ["calf", "beef cattle", "veal calf", "beef cattle",
                                "holstein friesians", "calf", "lactating dairy cow",
                                "holstein", "dairy cattle"]:
                    specie = "bos taurus"
                elif str(row["names"]) == "9913":
                    specie = "bos taurus"
                elif specie in ["porcine", "porc", "pork", "pigs", "diseased pig", "tibetan pig"]:
                    specie = "pig"
                elif specie in ["boiler", "broiler chicken", "broiler",
                                "chicken (broiler)", "meat chicken", "broilers",
                                "egg laying hen", "chick"]:
                    specie = "gallus domesticus"
                elif specie in ["poultry animal (variety unknown)", "poultry", "poultry animal (variety unknown);poultry animal"]:
                    specie = "phasianidae"
                elif specie == "san clemente island goat":
                    specie = "goat"
                elif specie == "marmot":
                    specie = "marmota"
                elif specie == "buffalo":
                    specie = "bos bubalis"
                elif specie == "larus sp.":
                    specie = "larus"
                elif specie in ["roe deer", "water deer"]:
                    specie = "capreolus capreolus"
                elif specie == "chlrocebus sabaeus":
                    specie = "Cercopithecus sabaeus"
                elif specie == "marine shellfish":
                    specie == "protostomia"
                elif specie == "oreochromis sp.":
                    specie = "oreochromis"
                elif specie == "mouse":
                    specie = "mus musculus"
                elif specie in ["equis caballus", "equus ferus caballus", "mare"]:
                    specie = "equus caballus"
                elif specie in ["migratory bird", "wild bird", "wild birds", "migratory birds"]:
                    specie = "aves"
                elif specie in ["shrimp"]:
                    specie = "decapoda"
                elif specie == "lizard":
                    specie = "squamata"
                elif specie == "crab":
                    specie = "brachyura"
                elif specie == "ovine":
                    specie = "ovis"
                elif specie == "chaetodipus sp.":
                    specie = "chaetodipus"
                elif specie == "caprine":
                    specie = "caprinae"
                elif specie == "crow":
                    specie = "crows"
                elif specie == "canine":
                    specie = "dog"
                elif specie == "nectomys melanius":
                    specie = "nectomys"
                elif specie in ["flys", "fly"]:
                    specie = "diptera"
                elif specie == "martes sp.":
                    specie = "martes"
                elif specie == "non-human primate":
                    specie = "mammal"
                unique_taxids = self.names_df.loc[self.names_df["name_txt_lwr"]==str(specie).index,"tax_id"].unique()
                if len(unique_taxids) == 1:
                    new_taxid = int(unique_taxids)
                elif len(unique_taxids) > 1:
                    all_lst = []
                    for n in range(len(unique_taxids)):
                        all_lst.append([])
                    while True:
                        new_taxids = []
                        for n in range(len(all_lst)):
                            if 1 in all_lst[n]:
                                new_taxids.append(np.NaN)
                                continue
                            current_taxid = unique_taxids[n]
                            all_lst[n].append(current_taxid)
                            parent_taxid = self.get_parenttaxid(taxid=current_taxid)
                            new_taxids.append(parent_taxid)
                        unique_taxids = new_taxids
                        common_taxid = set.intersection(*map(set,all_lst))
                        if len(common_taxid) == 1:
                            leave_loop = True
                            (common_taxid,) = common_taxid
                            break
                        elif len(common_taxid) > 1:
                            raise ValueError("WE HAVE PROBLEMS")
                        else:
                            leave_loop = False
                    new_taxid = common_taxid #[7215], [32281], [2081351]]
                else:
                    split_specie = specie.split(" ")
                    if len(split_specie) > 2:
                        short_specie = " ".join(split_specie[:1])
                        unique_taxids = self.names_df[self.names_df["name_txt_lwr"]==short_specie]["tax_id"].unique()
                        if len(unique_taxids) == 1:
                            new_taxid = int(unique_taxids)
                        else:
                            new_taxid = np.NaN
                    elif len(split_specie) == 1:
                        if specie[-1] == "s":
                            specie_ = specie[:-1]
                            unique_taxids = self.names_df[self.names_df["name_txt_lwr"]==specie_]["tax_id"].unique()
                        else:
                            species = specie+"s"
                            unique_taxids = self.names_df[self.names_df["name_txt_lwr"]==species]["tax_id"].unique()
                        if len(unique_taxids) == 1:
                            new_taxid = int(unique_taxids)
                        else:
                            new_taxid = np.NaN
                    else:
                        new_taxid = np.NaN

            self.ena_file.loc[self.ena_file["host"]==str(row["names"]).index, "host_tax_id"] = new_taxid
            gc.collect()
        self.ena_file = self.ena_file

def get_arguments():
    parser = argparse.ArgumentParser(description='Clean ENA downloaded metadata')
    parser.add_argument('-i','--in_metadataFile', help='In metadata file', required=True)
    parser.add_argument('--from_date', help="From Date", default=False, type=datetime.date.fromisoformat)
    parser.add_argument('-o','--out_metadataFile', help='Out metadata file', required=True)
    parser.add_argument('-t', '--tax_folder', help='Folder with the taxonomy files (not implemented)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    instance_animal = Clean_ENA_animal(file_tsv=args.in_metadataFile)
    if args.from_date:
        instance_animal.select_date(start_date=args.from_date)
    instance_animal.get_humanhosts(save_file=args.out_metadataFile)
