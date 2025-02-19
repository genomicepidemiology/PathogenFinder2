import pandas as pd
from tqdm import tqdm
import argparse
pd.options.mode.chained_assignment = None

class Format_Metadata:

    def __init__(self, genbank_file, refseq_file, animals=False):

        self.pathogen_db = None
        self.nonpathogen_db = None
        self.db = None
        self.animals = animals
        self.genbank_file = pd.read_csv(genbank_file, sep="\t", skiprows=[0])
        self.refseq_file = pd.read_csv(refseq_file, sep="\t", skiprows=[0])
        self.genbank_file.columns = self.genbank_file.columns.str.replace("#", "")
        self.refseq_file.columns = self.refseq_file.columns.str.replace("#", "")
        self.replace_sp = {"E.coli and Shigella": "Escherichia coli",
                            "Bacillus cereus group": "Bacillus cereus",
                            "Burkholderia cepacia complex": "Burkholderia cepacia"}
        try:
            self.genbank_file["clean_accession"] = self.genbank_file["assembly_accession"].str.split(".").str[0]
        except KeyError:
            self.genbank_file["clean_accession"] = self.genbank_file["assembly_accession"].str.split(".").str[0]
        if animals:
            self.columns_end = ["Accession Number", #"Name",
                                "Taxonomy ID",
                                "Species", "Strain", "Origin DB", "Reason", "Host",
                                "Host TaxID", "Host Species TaxID", "Host Host Genus TaxID",
                                "Host Family TaxID", "Host Order TaxID", "Host Class TaxID",
                                "Host Clade TaxID"]
        else:
            self.columns_end = ["Accession Number", #"Name",
                                "Taxonomy ID",
                                "Species", "Strain", "Origin DB", "Reason Phenotype"]

    def define_abb(self, db):
        index_repl = db.loc[(db["Species"].str.split().str.len()==2)&(
                            ~db["Species"].str.contains("sp.", na=False))&(
                            ~db["Species"].str.contains("uncultured", na=False))&(
                            ~db["Species"].str.contains("Candidatus", na=False))].index
        db["Species abb."] = db["Species"]
        sup = db.loc[index_repl, "Species"].str.split().str[0].str[:1] + "."
        sup2 = db.loc[index_repl, "Species"].str.split().str[1:]
        db.loc[index_repl, "Species abb."] = sup + " " + sup2.str.join(' ')
        return db


    def refine_cols(self, columns, db):
        for k, v in columns.items():
            if isinstance(v, str):
                db = db.rename(columns={v:k})
 #               db.rename(columns={v:k}, inplace=True)
            else:
                if k == "Name":
                    supperpose = db.apply(lambda x: str(x[v[1]]) in str(x[v[0]]), axis=1)
                    if not supperpose.empty:
                        index_newname = db.loc[db.apply(lambda x: str(x[v[1]]) in str(x[v[0]]), axis=1)].index
                    else:
                        index_newname = []
                    db[k] = db[v[0]]
                    db.loc[~db.index.isin(index_newname), k] = db.loc[~db.index.isin(index_newname), v[0]] + " " + db.loc[~db.index.isin(index_newname), v[1]]
                else:
                    db[k] = db[v[0]] + " " + db[v[1]]
        print(db)
        db = db[db.columns.intersection(self.columns_end)]
        print(db)
        new_accession = db["Accession Number"].str.split(".").str[0].tolist()
        db.loc[db.index,"Accession Number"] = new_accession
        for old, new in self.replace_sp.items():
            db["Species"] = db["Species"].str.replace(old, new)
        db["Species"]=db["Species"].str.replace("[", "")
        db["Species"]=db["Species"].str.replace("]", "")
        return db

    def clean_redundant_old(self, redundant_db):
        g = redundant_db.groupby("Accession Number")
        x = g.agg("first")
        x.update(g.agg({"Origin DB": ", ".join}))
        redundant_db = x.reset_index()
 #       redundant_db.dropna(subset=["Accession Number"], inplace=True)
        redundant_db = redundant_db.dropna(subset=["Accession Number"])
        return redundant_db

    def join_db(self, db, columns, pathogenicity_db):
        if pathogenicity_db is None:
            redundant_db = self.refine_cols(columns, db)
        else:
            db = self.refine_cols(columns, db)
            redundant_db = pd.concat([pathogenicity_db, db])
        return redundant_db

    def clean_redundant(self, redundant_db):
        redundant_db = redundant_db[~redundant_db["Accession Number"].isna()]
        redundant_db = redundant_db.fillna('')
        pathogenicity_db = (redundant_db.groupby("Accession Number").agg(
                {"Species": ", ".join,
                "Strain": ", ".join, "Origin DB": ", ".join, "Reason Phenotype": ", ".join,
                "Taxonomy ID": "first"}).reset_index())
        return pathogenicity_db

    def refine_db(self, redundant_db):
        cols_repl = ["Species", "Strain"]
        for col in cols_repl:
#            redundant_db[col].fillna("", inplace=True)
            redundant_db[col] = redundant_db[col].fillna("")
        pathogenicity_db = self.clean_redundant(redundant_db=redundant_db)
        new_sp = []
        new_sp_str = []
        for index, rows in tqdm(pathogenicity_db.iterrows()):
            lst_str = rows["Strain"].split(", ")
            set_str = set(lst_str)
            if len(lst_str) < 2:
                new_sp_str.append(list(set_str)[0])
            else:
                new_sp_str.append("//".join(list(set_str)))
            lst_sp = rows["Species"].split(", ")
            set_sp = set(lst_sp)
            set_clean_sp = set_sp - set_str
            if len(set_clean_sp) == 0:
                set_clean_sp = set_sp
            if len(list(set_clean_sp)) < 2:
                new_sp.append(list(set_clean_sp)[0])
            else:
                new_sp.append("//".join(list(set_clean_sp)))
        pathogenicity_db["Species"] = new_sp
        pathogenicity_db["Strain"] = new_sp_str
        db = self.define_abb(pathogenicity_db)
        return db

    def refine_db_old_new(self, redundant_db):
        cols_repl = ["Species", "Strain"]
        for col in cols_repl:
#            redundant_db[col].fillna("",inplace=True)
            redundant_db[col] = redundant_db[col].fillna("")
        pathogenicity_db = self.clean_redundant(redundant_db=redundant_db)
        new_sp = []
        new_sp_str = []
        for index, rows in tqdm(pathogenicity_db.iterrows()):
            lst_str = rows["Strain"].split(", ")
            set_str = set(lst_str)
            if len(lst_str) < 2:
                new_sp_str.append(list(set_str)[0])
            else:
                new_sp_str.append("//".join(list(set_str)))
            lst_sp = rows["Species"].split(", ")
            set_sp = set(lst_sp)
            set_clean_sp = set_sp - set_str
            if len(set_clean_sp) == 0:
                set_clean_sp = set_sp
            if len(list(set_clean_sp)) < 2:
                new_sp.append(list(set_clean_sp)[0])
            else:
                new_sp.append("//".join(list(set_clean_sp)))
        pathogenicity_db["Species"] = new_sp
        pathogenicity_db["Strain"] = new_sp_str
        db = self.define_abb(pathogenicity_db)
        return db

    def fillna_sp(self, db):
        index_fill = db[db["Species"].isna()].index
        species = db.loc[index_fill,"Name"].str.split().str[:2].str.join(" ")
        db.loc[index_fill,"Species"] = species
        return db

    def add_pathogen_metadata(self, db_path, patho, origin, sep):
        db = pd.read_csv(db_path, sep=sep)
        db["Origin DB"] = origin
        if origin == "ENA":
            cols_dict = {"Accession Number": "assembly_accession",
                #"Name":["scientific_name", "strain"], 
                "Taxonomy ID":"tax_id",
                "Species":"scientific_name", "Strain":"strain",
                "Origin DB":"Origin DB", "Reason Phenotype": "Reason"}
            if self.animals:
                cols_dict["Host"] = "host"
                cols_dict["Host TaxID"] = "hosttxid"
                cols_dict["Host Species TaxID"] = "species_hosttxid"
                cols_dict["Host Genus TaxID"] = "genus_hosttxid"
                cols_dict["Host Family TaxID"] = "family_hosttxid"
                cols_dict["Host Order TaxID"] = "order_hosttxid"
                cols_dict["Host Class TaxID"] = "class_hosttxid"
                cols_dict["Host Clade TaxID"] = "clade_hosttxid"
            db["Reason"] = "ENADB"
        elif origin == "NCBI":
            cols_dict = {"Accession Number": "Assembly",
                #"Name":["#Organism group", "Strain"],
                "Taxonomy ID":"TaxID",
                "Species":"Organism group", "Strain":"Strain", #Updated from #Organism group
                "Origin DB":"Origin DB", "Reason Phenotype": "Reason"}
            db["Reason"] = "NCBIPathogenDBB"
            if self.animals:
                cols_dict["Host"] = "host"
                cols_dict["Host TaxID"] = "hosttxid"
                cols_dict["Host Species TaxID"] = "species_hosttxid"
                cols_dict["Host Genus TaxID"] = "genus_hosttxid"
                cols_dict["Host Family TaxID"] = "family_hosttxid"
                cols_dict["Host Order TaxID"] = "order_hosttxid"
                cols_dict["Host Class TaxID"] = "class_hosttxid"
                cols_dict["Host Clade TaxID"] = "clade_hosttxid"
                cols_dict["Species"] = "Organism group"
                cols_dict["Name"] = ["Organism group", "Strain"]
        elif "Patric" in origin:
            cols_dict = {"Accession Number": "Assembly Accession",
                #"Name":["Species", "Strain"],
                "Taxonomy ID":"NCBI Taxon ID",
                "Species":"Species", "Strain":"Strain",
                "Origin DB":"Origin DB", "Reason Phenotype": "Reason"}
        elif origin == "Extremophile":
            db["Reason"] = "Extremophile"
            cols_dict = {"Accession Number": "Accession Genbank",
                #"Name":["Genus_and_species", "Strain"],
                "Taxonomy ID":"Taxonomic ID",
                "Species":"Genus_and_species", "Strain":"Strain",
                "Origin DB":"Origin DB", "Reason Phenotype": "Reason"}
        elif "Microbiome" in origin:
            db["Reason"] = "Microbiome"
            cols_dict = {"Accession Number": "Accession",
                #"Name":"Organism Name",
                "Taxonomic ID":"TaxID",
                "Species":"Species name", "Strain":"Strain",
                "Origin DB":"Origin DB", "Reason Phenotype": "Reason"}
        else:
            raise KeyError("DOESNT EXIST")
        if patho:
            self.pathogen_db = self.join_db(db=db, columns=cols_dict,
                                        pathogenicity_db=self.pathogen_db)
            self.pathogen_db = self.refine_db(self.pathogen_db)

        else:
            self.nonpathogen_db = self.join_db(db=db, columns=cols_dict,
                                        pathogenicity_db=self.nonpathogen_db)
            self.nonpathogen_db = self.refine_db(self.nonpathogen_db)

        #pd.options.display.max_columns = None


    def final_db(self, pathogen_db, nonpathogen_db):
        self.db = pd.concat([pathogen_db, nonpathogen_db])

    def get_accession_info(self):
        if self.pathogen_db is not None:
            self.pathogen_db = self.pathogen_db.merge(self.genbank_file, left_on='Accession Number', right_on='clean_accession')

        if self.nonpathogen_db is not None:
            self.nonpathogen_db = self.nonpathogen_db.merge(self.genbank_file, left_on='Accession Number', right_on='clean_accession')


    def save_db(self, out="./", patho_db=None, nonpatho_db=None):
        if self.animals:
            str_host = "allhosts"
        else:
            str_host = "humans"

        if patho_db is None:
            pathogen_db = self.pathogen_db
        else:
            pathogen_db = patho_db            
        if nonpatho_db is None:
            nonpathogen_db = self.nonpathogen_db
        else:
            nonpathogen_db = nonpatho_db

        if pathogen_db is not None:
            try:
                del pathogen_db["clean_accession"]
                del pathogen_db["Accession Number"]
            except KeyError:
                pass
            #pathogen_db.rename(columns={"Species":"Species_DBs",
             #                           "Strain": "Strain_DBs",
             #                           "assembly accession": "Assembly Accession"
            #                            }, inplace=True)
            pathogen_db = pathogen_db.rename(columns={"Species":"Species_DBs",
                                                      "Strain": "Strain_DBs",
                                                    "assembly accession": "Assembly Accession"})
            pathogen_db["PathoPhenotype"] = "Pathogenic"
            pathogen_db.columns = pathogen_db.columns.str.replace("#", "").str.rstrip().str.lstrip()
            pathogen_db.to_csv("{}{}pathogen_db2.tsv".format(out,str_host), sep="\t", index=False)
        if nonpathogen_db is not None:
            try:
                del nonpathogen_db["clean_accession"]
                del nonpathogen_db["Accession Number"]
            except KeyError:
                pass
#            nonpathogen_db.rename(columns={"Species":"Species_DBs",
 #                                       "Strain": "Strain_DBs",
  #                                      "assembly accession": "Assembly Accession"
   #                                     }, inplace=True)
            nonpathogen_db = nonpathogen_db.rename(columns={"Species":"Species_DBs",
                                                           "Strain": "Strain_DBs",
                                                        "assembly accession": "Assembly Accession"})
            nonpathogen_db["PathoPhenotype"] = "No Pathogenic"
            nonpathogen_db.columns = nonpathogen_db.columns.str.replace("#", "").str.rstrip().str.lstrip()
            nonpathogen_db.to_csv("{}humannonpathogen_db2.tsv".format(out), sep="\t", index=False)
        if pathogen_db is not None and nonpathogen_db is not None:
            self.final_db(pathogen_db, nonpathogen_db)
            self.db.to_csv("{}{}novelpathogenicity2.tsv".format(out,str_host), sep="\t", index=False)
    
    def remove_nonpatho_path(self):
 #       discard_patho = self.nonpathogen_db[self.nonpathogen_db["# assembly_accession"].isin(self.pathogen_db["# assembly_accession"])]
  #      self.nonpathogen_db = self.nonpathogen_db[~self.nonpathogen_db["# assembly_accession"].isin(self.pathogen_db["# assembly_accession"])]
        discard_patho = self.nonpathogen_db[self.nonpathogen_db["assembly_accession"].isin(self.pathogen_db["assembly_accession"])]
        self.nonpathogen_db = self.nonpathogen_db[~self.nonpathogen_db["assembly_accession"].isin(self.pathogen_db["assembly_accession"])]
        return discard_patho

    
    def assign_species_rseq(self, db):
        return db["organism_name_refseq"].str.split().str[:2].str.join(" ").tolist()

    def get_refseq(self):
        if self.pathogen_db is not None:
            refseq_pathogen = self.pathogen_db[self.pathogen_db["paired_asm_comp"]=="identical"]
            refseq_pathogen = refseq_pathogen.merge(self.refseq_file, left_on="gbrs_paired_asm",
                                        right_on="assembly_accession", suffixes=("_genbank", "_refseq"))
            del refseq_pathogen["bioproject_genbank"]
            del refseq_pathogen["biosample_genbank"]
            del refseq_pathogen["wgs_master_genbank"]
            del refseq_pathogen["refseq_category_genbank"]
            del refseq_pathogen["taxid_genbank"]
            del refseq_pathogen["species_taxid_genbank"]
            del refseq_pathogen["organism_name_genbank"]
            del refseq_pathogen["infraspecific_name_genbank"]
            del refseq_pathogen["isolate_genbank"]
            del refseq_pathogen["version_status_genbank"]
            del refseq_pathogen["assembly_level_genbank"]
            del refseq_pathogen["release_type_genbank"]
            del refseq_pathogen["genome_rep_genbank"]
            del refseq_pathogen["seq_rel_date_genbank"]
            del refseq_pathogen["asm_name_genbank"]
            del refseq_pathogen["asm_submitter_genbank"]
            del refseq_pathogen["gbrs_paired_asm_genbank"]
            del refseq_pathogen["paired_asm_comp_genbank"]
            del refseq_pathogen["excluded_from_refseq_genbank"]
            del refseq_pathogen["relation_to_type_material_genbank"]
            del refseq_pathogen["asm_not_live_date_genbank"]
            refseq_pathogen["Species_Refseq"] = self.assign_species_rseq(refseq_pathogen)
        else:
            refseq_pathogen = None

        if self.nonpathogen_db is not None:
            refseq_nonpathogen = self.nonpathogen_db[self.nonpathogen_db["paired_asm_comp"]=="identical"]
            refseq_nonpathogen = refseq_nonpathogen.merge(self.refseq_file, left_on="gbrs_paired_asm",
                                        right_on="assembly_accession", suffixes=("_genbank", "_refseq"))
            del refseq_nonpathogen["bioproject_genbank"]
            del refseq_nonpathogen["biosample_genbank"]
            del refseq_nonpathogen["wgs_master_genbank"]
            del refseq_nonpathogen["refseq_category_genbank"]
            del refseq_nonpathogen["taxid_genbank"]
            del refseq_nonpathogen["species_taxid_genbank"]
            del refseq_nonpathogen["organism_name_genbank"]
            del refseq_nonpathogen["infraspecific_name_genbank"]
            del refseq_nonpathogen["isolate_genbank"]
            del refseq_nonpathogen["version_status_genbank"]
            del refseq_nonpathogen["assembly_level_genbank"]
            del refseq_nonpathogen["release_type_genbank"]
            del refseq_nonpathogen["genome_rep_genbank"]
            del refseq_nonpathogen["seq_rel_date_genbank"]
            del refseq_nonpathogen["asm_name_genbank"]
            del refseq_nonpathogen["asm_submitter_genbank"]
            del refseq_nonpathogen["gbrs_paired_asm_genbank"]
            del refseq_nonpathogen["paired_asm_comp_genbank"]
            del refseq_nonpathogen["excluded_from_refseq_genbank"]
            del refseq_nonpathogen["relation_to_type_material_genbank"]
            del refseq_nonpathogen["asm_not_live_date_genbank"]
            refseq_nonpathogen["Species_Refseq"] = self.assign_species_rseq(refseq_nonpathogen)

        else:
            refseq_nonpathogen = None
        return refseq_pathogen, refseq_nonpathogen

def get_arguments():
    parser = argparse.ArgumentParser(description='Join the databases and add genome information')
    parser.add_argument('-o','--output_folder', help="Output folder", required=True)
    parser.add_argument('-r','--refseq', help='RefSeq File', required=True)
    parser.add_argument('-g','--genbank', help='Genbank File', required=True)
    parser.add_argument('--ena', help="ENA File", default=False)
    parser.add_argument('--ncbi', help="NCBIPathogen File", default=False)
    parser.add_argument('--patric_disease', help="Patric Disease", default=False)
    parser.add_argument('--patric_nonpatho', help="Patric Non Patho", default=False)
    parser.add_argument('--patric_microbiome', help="Patric Microbiome", default=False)
    parser.add_argument('--patric_probiotic', help="Patric Probiotic", default=False)
    parser.add_argument('--patric_extremophile', help="Patric Extremophile", default=False)
    parser.add_argument('--extremophile', help="Extremophile", default=False)
    parser.add_argument('--hmp', help="HMP", default=False)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_arguments()
    instance = Format_Metadata(animals=False, refseq_file=args.refseq,
                    genbank_file=args.genbank)
    if args.ena:
        instance.add_pathogen_metadata(db_path=args.ena,
                            patho=True, origin="ENA", sep="\t")
    if args.ncbi:
        instance.add_pathogen_metadata(db_path=args.ncbi,
                            patho=True, origin="NCBI", sep=",")
    if args.patric_disease:
        instance.add_pathogen_metadata(db_path=args.patric_disease,
                            patho=True, origin="Patric (diseases)", sep=",")
    if args.patric_nonpatho:
        instance.add_pathogen_metadata(
        #        db_path="../patric/BVBRC_subsets/BVBRC_bacteria_nonpathogens2-selNLP_nonpathogen-col_healthdescription.csv",
                db_path=args.patric_nonpatho, patho=False, origin="Patric (nonpathogen)", sep=",")
    if args.patric_microbiome:
        instance.add_pathogen_metadata(
    #        db_path="../patric/BVBRC_subsets/BVBRC_bacteria_nonpathogens2-selNLP_microbiome-col_healthdescription.csv",
            db_path=args.patric_microbiome, patho=False, origin="Patric (microbiome)", sep=",")
    if args.patric_probiotic:
        instance.add_pathogen_metadata(
    #        db_path="../patric/BVBRC_subsets/BVBRC_bacteria_nonpathogens2-selNLP_probiotic-col_healthdescription.csv",
            db_path=args.patric_probiotic, patho=False, origin="Patric (probiotic)", sep=",")
    if args.patric_extremophile:
        instance.add_pathogen_metadata(
    #        db_path="../patric/BVBRC_subsets/BVBRC_bacteria_nonpathogens2-sel_extremophile-col_healthdescription.csv",
            db_path=args.patric_extremophile, patho=False, origin="Patric (extremophile)", sep=",")
    if args.extremophile:
        instance.add_pathogen_metadata(
    #        db_path="../extremophile/extreme_temp.csv",
            db_path=args.extremophile, patho=False, origin="Extremophile", sep=",")
    if args.hmp:
        instance.add_pathogen_metadata(
    #        db_path="../humanmicrobiome/HMP_clean.tsv",
            db_path=args.hmp, patho=False, origin="Microbiome (HMP1)", sep="\t")
    instance.get_accession_info()
    instance.save_db(out=args.output_folder)
    discard_double = instance.remove_nonpatho_path()
    discard_double.to_csv("{}/discard_nonpatho2.csv".format(args.output_folder),sep="\t", index=False)
    refseq_pathogen, refseq_nonpathogen = instance.get_refseq()
    instance.save_db(out="{}/refseq".format(args.output_folder), patho_db=refseq_pathogen, nonpatho_db=refseq_nonpathogen)
    #repr_patho, repr_nonpatho = instance.get_representative()
    ###############
