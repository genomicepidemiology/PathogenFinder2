import os
import pandas as pd
import numpy as np
import git
import uuid
import datetime
import json
import hashlib
import sys



class Prediction_Report:

    def __init__(self, out_folder):

        self.out_folder = out_folder

    @staticmethod
    def get_predictions(ensemble_results):
        predictions = {}
        for name, val in ensemble_results.items():
            pred_lst = [name]
            pred_lst.extend(val["Output"]["Prediction"])
            predictionPF = pd.Series(pred_lst, index=["File Name","Prediction_0",
                                                "Prediction_1", "Prediction_2", "Prediction_3"])
            predictionPF["Prediction Mean"] = np.mean(val["Output"]["Prediction"])
            predictionPF["Prediction STD"] = np.std(val["Output"]["Prediction"])
            if predictionPF["Prediction Mean"] > 0.5:
                predictionPF["Binary Prediction Mean"] = 1
                predictionPF["Phenotype"] = "Human Pathogenic"
            else:
                predictionPF["Phenotype"] = "Human Non Pathogenic"
                predictionPF["Binary Prediction Mean"] = 0
            ensemble_results[os.path.basename(name)]["Ensemble Predictions"] = predictionPF
        return ensemble_results

    def save_report(self, results_ensemble, save_attentions=True, save_embeddings=True):
        predictions_paths = []
        embeddings_paths = []
        att_paths = []
        for name, val in results_ensemble.items():
            folder_out_sample = self.out_folder[name]
            pred_path = "{}/predictions.tsv".format(folder_out_sample)
            df_results = val["Ensemble Predictions"].to_frame().T
            with open(pred_path, "w") as pp:
                pp.write("# Results From PathogenFinder2\n")
                pp.write("## 'File Name' has the input file, 'Prediction_0-3' has the predictions of each neural network,")
                pp.write(" 'Prediction Mean' has the mean of the 4 neural networks, 'Prediction_STD' has the standard deviation ")
                pp.write("of the four predictions,\n## 'Phenotype' has if the prediction is pathogenic or non pathogenic, and ")
                pp.write("'Binary Prediction Mean' has the value of the prediction as 0 (non pathogenic) or 1 (pathogenic).\n")            
            df_results.to_csv(pred_path, sep="\t", index=False, mode="a")
            predictions_paths.append(pred_path)
            if save_attentions:
                att_path = "{}/attentions.npz".format(folder_out_sample)
                np.savez_compressed(att_path,
                                    protIDs=val["Features"]["ProtIDs"],
                                    attentions=val["Output"]["Attention"])
            else:
                att_path = None
            att_paths.append(att_path)
            if save_embeddings:
                embed_path = "{}/embeddings.npz".format(folder_out_sample)
                np.savez_compressed(embed_path,
                                    embeddings_1=val["Output"]["Embeddings1"].astype("float"),
                                    embeddings_2=val["Output"]["Embeddings2"].astype("float"))
            else:
                embed_path = None
            embeddings_paths.append(embed_path)
        return predictions_paths, embeddings_paths,att_paths
            


class CGEResults:

    def __init__(self, args_dict:dict):

        self.software_result = dict()
        self.add_software_result(args_dict=args_dict)

    @staticmethod
    def runid_from_command() -> str:
        command = " ".join(sys.argv)
        return hashlib.sha1(command.encode()).hexdigest()

  
    @staticmethod
    def generate_random_str(string_feature, values_dict):
        if isinstance(string_feature, pd.Series):
            return string_feature.apply(
                lambda x: CGEResults.generate_random_str(x, values_dict)
            )
        while True:
            key = f"{string_feature};;{uuid.uuid4()}"
            if key not in values_dict:
                return key

    def add_software_result(self, args_dict:dict):
        # TODO UPDATE AUTOMATIC
        from pathogenfinder2 import __version__

        repo = git.Repo(search_parent_directories=True)
        tz = datetime.timezone.utc
        ft = "%Y-%m-%dT%H:%M:%S%z"
        data_time = datetime.datetime.now(tz=tz).strftime(ft)
        self.software_result["type"] = "software_result"
        self.software_result["key"] = "PathogenFinder2-{}".format(__version__)
        self.software_result["software_name"] = "PathogenFinder2"
        self.software_result["software_version"] = "{}".format(__version__)
        self.software_result["software_branch"] = "{}".format(repo.active_branch.name)
        self.software_result["software_commit"] = "{}".format(repo.head.object.hexsha)
        self.software_result["software_log"] = ""
        self.software_result["run_id"] = hashlib.sha1(" ".join(sys.argv).encode()).hexdigest()
        self.software_result["run_date"] = data_time
        self.software_result["phenotypes_ml"] = {}
        self.software_result["seq_regions"] = {}
        self.software_result["phenotypes"] = {}
        self.software_result["neighbors"] = {}
        self.software_result["software_executions"] = {}
        self.software_result["databases"] = {}

    def add_log(self, result_summary, log):
        self.software_result["software_log"] = log
        self.software_result["result_summary"] = result_summary


    def add_software_exec(self, parameters:[str, dict]):
        # TODO
        software_exec = {}
        software_exec["type"] = "software_exec"
        software_exec["key"] = hashlib.sha1(" ".join(sys.argv).encode()).hexdigest()
        software_exec["software_name"] = self.software_result["software_name"]
        software_exec["command"] = " ".join(sys.argv)
        software_exec["parameters"] = parameters
        self.software_result["software_executions"][software_exec["key"]] = software_exec

    def add_database(self, name, version, commit=""):
        database = {}
        database["type"] = "database"
        database["database_name"] = name
        database["database_version"] = version
        database["key"] = "{}_{}".format(name, version)
        database["database_commit"] = commit
        self.software_result["databases"][database["key"]] = database
        return database

    def add_phenotype_result(self, results_ensemble):
        phenotype_result = {}
        phenotype_result["type"] = "phenotype_ml"
        #phenotype_result["key"] = "human-bacterial-pathogenicity_{}".format(
         #                               hashlib.md5(str(results_ensemble).encode()).hexdigest())
        phenotype_result["key"] = "human_bacterial_pathogenicity"
        phenotype_result["category"] = "Pathogenicity"
        phenotype_result["ensemble_pred"] = True
        phenotype_result["type_pred"] = "Categorical"
        phenotype_result["prediction"] = results_ensemble["Phenotype"]
        phenotype_result["output_model"] = {}
        for n in ["0", "1", "2", "3"]:
            phenotype_result["Prediction_{}".format(n)] = round(results_ensemble["Prediction_{}".format(n)],4)
        phenotype_result["output_mean"] = round(results_ensemble["Prediction Mean"], 4)
        phenotype_result["output_std"] = round(results_ensemble["Prediction STD"], 4)
        self.software_result["phenotypes_ml"][phenotype_result["key"]] = phenotype_result
    
    def add_bacterialneighbors(self, query_id:str, neighbors_df:pd.DataFrame):
        for n in range(len(neighbors_df)):
            entry = neighbors_df.iloc[n]
            name = "{};;{}".format(query_id, entry["Names"])
            neighbor = {}
            neighbor["type"] = "neighbor"
            neighbor["key"] = CGEResults.generate_random_str(name, "neighbors")
            neighbor["query_id"] = query_id
            neighbor["query_name"] = query_id
            neighbor["ref_id"] = "{}_{}".format(entry["Names"], entry["RefSeq"])
            neighbor["ref_name"] = entry["Names"]
            neighbor["ref_acc"] = entry["RefSeq"]
            neighbor["distance_measure"] = "minkowski"
            neighbor["distance_value"] = entry["Distances"].item()
            neighbor["ref_database"] = ["PathogenFinder2"]
            neighbor["type_sequence"] = "Proteome"
            neighbor["type_compared"] = "PF2_Embedding"
            neighbor["ref_taxID"] = entry["Taxonomy"].item()
            neighbor["ref_species"] = entry["Species"]
            neighbor["ref_strain"] = entry["Strain"]
            neighbor["software"] = "Scikit-learn"
            neighbor["rank_neighbors"] = str(n)
            self.software_result["neighbors"][neighbor["key"]] = neighbor

    def add_proteinsatt(self, proteins_df, ref_db):
        columns_selected = ["type", "gene", "key", "name", "identity", "alignment_length", "ref_seq_length",
                            "coverage", "ref_id", "ref_start_pos", "ref_end_pos", "query_id",
                            "query_start_pos", "query_end_pos", "ref_acc", "ref_database", "grade"]
        proteins_df["type"] = "seq_region"
        proteins_df["gene"] = "Protein"
        proteins_df = proteins_df.rename(columns={
                        "stitle": "name", "pident": "identity", "length": "alignment_length",
                        "slen": "ref_seq_length", "scovhsp": "coverage", "sseqid": "ref_id",
                        "sstart": "ref_start_pos", "send": "ref_end_pos", "qseqid": "query_id",
                        "qstart": "query_start_pos", "qend": "query_end_pos"})
        proteins_df["ref_acc"] = proteins_df["ref_id"]
        proteins_df["ref_database"] = ref_db["key"]
        proteins_df = proteins_df.replace("-", None)
        proteins_df = proteins_df.astype({
            "coverage": "float", "identity": "float", "ref_start_pos": "float", "ref_end_pos": "float",
            "alignment_length": "float", "ref_seq_length": "float", "query_start_pos": "float", "query_end_pos": "float",
            "ref_start_pos":"float", "ref_end_pos": "float"})
        proteins_df["grade"] = 0
        proteins_df.loc[proteins_df["name"]=="No Match Found", "grade"] = -1
        proteins_df.loc[(proteins_df["coverage"] == 100) & (proteins_df["identity"] == 100), "grade"] = 3
        proteins_df.loc[(proteins_df["coverage"] == 100) & (proteins_df["identity"] < 100), "grade"] = 2
        proteins_df.loc[(proteins_df["coverage"] < 100), "grade"] = 1
        proteins_df = proteins_df.fillna("-")
        name = proteins_df["query_id"] + ";;" + proteins_df["ref_id"]
        proteins_df["key"] = CGEResults.generate_random_str(name, "proteins")
        protein_records = proteins_df[columns_selected].to_dict(orient="records")
        for protein in protein_records:
            self.software_result["seq_regions"][protein["key"]] = protein

    def add_gsearesults(self, gsea_df, ref_db):
        columns_selected = ["type", "key", "category", "vir_function", "vir_virulent", "vir_class",
                            "degree", "evidence", "seq_regions", "ref_database", "grade"]
        gsea_df["type"] = "phenotype"
        gsea_df["category"] = "functional_enrichment"
        gsea_df["Lead_genes_description"] = gsea_df["Lead_genes_description"].str.split(";")

        gsea_df = gsea_df.rename(columns={"Lead_genes_description":"seq_regions", "Term_class": "vir_class",
                            "FDR q-val":"evidence", "NES": "degree", "Term": "vir_function"})
        gsea_df = gsea_df[gsea_df["degree"]>0]
        gsea_df["vir_virulent"] = "unknown"
        gsea_df["ref_database"] = ref_db["key"]
        gsea_df["grade"] = 0
        gsea_df.loc[gsea_df["evidence"]<0.4, "grade"] = 1
        gsea_df.loc[gsea_df["evidence"]<0.05, "grade"] = 2
        gsea_df.loc[gsea_df["evidence"]<=0.01, "grade"] = 3
        name = gsea_df["vir_class"]+";;"+gsea_df["vir_function"]
        gsea_df["key"] = CGEResults.generate_random_str(name, "enrichment")
        gsea_records = gsea_df[columns_selected].to_dict(orient="records")
        for gsea in gsea_records:
            self.software_result["phenotypes"][gsea["key"]] = gsea

    def save_results(self, output_path):
        with open("{}".format(output_path), 'w') as f:
            json.dump(self.software_result, f)

