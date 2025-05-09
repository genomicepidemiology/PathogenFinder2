import os
import pandas as pd
import numpy as np
import git
import hashlib
import uuid
import datetime
import json
from pathogenfinder2 import __version__



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
                pp.write("of the four predictions,\n'Phenotype' has if the prediction is pathogenic or non pathogenic, and ")
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
    def generate_random_str(string_feature, values_dict):
        while True:
            key = "{};;{}".format(string_feature, uuid.uuid4())
            if key not in values_dict:
                break
        return key

    def add_software_result(self, args_dict:dict):
        # TODO UPDATE AUTOMATIC
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
        self.software_result["run_id"] = hashlib.md5(str(args_dict).encode()).hexdigest()
        self.software_result["run_date"] = data_time
        self.software_result["phenotypes_ml"] = {}
        self.software_result["seq_regions"] = {}
        self.software_result["neighbors"] = {}
        self.software_result["software_executions"] = {}
        self.software_result["databases"] = {}

    def add_log(self, result_summary, log):
        self.software_result["software_log"] = log
        self.software_result["result_summary"] = result_summary


    def add_software_exec(self, software_name:str, command:str, stdout:str, 
                            stderr:str, parameters:[str, dict]):
        # TODO
        software_exec = {}
        software_exec["type"] = "software_exec"
        software_exec["key"] = "{}".format(software_name)
 #       software_exec["key"] = "{}_{}".format(software_name,
  #                                                  hashlib.md5(str(parameters).encode()).hexdigest())
        software_exec["software_name"] = software_name
        software_exec["command"] = command
        software_exec["parameters"] = parameters
        software_exec["stdout"] = stdout
        software_exec["stderr"] = stderr
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
        phenotype_result["key"] = "human-bacterial-pathogenicity_{}".format(
                                        hashlib.md5(str(results_ensemble).encode()).hexdigest())
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
            neighbor["ref_database"] = "PathogenFinder2"
            neighbor["type_sequence"] = "Proteome"
            neighbor["type_compared"] = "PF2_Embedding"
            neighbor["ref_taxID"] = entry["Taxonomy"].item()
            neighbor["ref_species"] = entry["Species"]
            neighbor["ref_strain"] = entry["Strain"]
            neighbor["software"] = "Scikit-learn"
            neighbor["rank_neighbors"] = str(n)
            self.software_result["neighbors"][neighbor["type"]] = neighbor
            

    def add_proteinsatt(self, proteins_df, ref_db):
        for n in range(len(proteins_df)):
            entry = proteins_df.iloc[n]
            protein = {}
            name = "{};;{}".format(entry["Query_ID"], entry["Ref_ID"])
            protein["type"] = "seq_region"
            protein["key"] =  CGEResults.generate_random_str(name, "neighbors")
            protein["gene"] = "Protein"
            protein["name"] = entry["Ref_name"]
            protein["identity"] = entry["Identity"]
            protein["alignment_length"] = entry["Alignment_Length"]
            protein["ref_seq_lenght"] = entry["Ref_Length"]
            protein["coverage"] = entry["Ref_coverage"]
            protein["ref_id"] = entry["Ref_ID"]
            protein["ref_acc"] = entry["Ref_ID"]
            protein["ref_start_pos"] = entry["Ref_start_pos"]
            protein["ref_end_pos"] = entry["Ref_end_pos"]
            protein["query_id"] = entry["Query_ID"]
            protein["query_start_pos"] = entry["Query_start_pos"]
            protein["query_end_pos"] = entry["Query_end_pos"]
            protein["ref_database"] = "UniRef50"
#            protein["note"] = "Attention Score {}".format(entry["Attention Value"])
            if protein["name"] == "No Match Found":
                protein["grade"] = -1
            else:
                protein["coverage"] = float(protein["coverage"])
                protein["identity"] = float(protein["identity"])
                protein["ref_start_pos"] = int(protein["ref_start_pos"])
                protein["alignment_length"] = int(protein["alignment_length"])
                protein["ref_seq_lenght"] = int(protein["ref_seq_lenght"])
                protein["ref_start_pos"] = int(protein["ref_start_pos"])
                protein["ref_end_pos"] = int(protein["ref_end_pos"])
                protein["query_start_pos"] = int(protein["query_start_pos"])
                protein["query_end_pos"] = int(protein["query_end_pos"])
                if float(protein["coverage"]) == 100. and float(protein["identity"]) == 100.:
                    protein["grade"] = 3
                elif float(protein["coverage"]) == 100. and float(protein["identity"]) < 100.:
                    protein["grade"] = 2
                elif float(protein["coverage"]) < 100.:
                    protein["grade"] = 1
                else:
                    protein["grade"] = 0
            self.software_result["seq_regions"][protein["key"]] = protein

    def save_results(self, output_path):
        with open("{}".format(output_path), 'w') as f:
            json.dump(self.software_result, f)

