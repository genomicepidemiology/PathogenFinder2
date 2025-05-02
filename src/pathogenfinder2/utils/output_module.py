import os
import pandas as pd
import numpy as np




class Prediction_Report:

    def __init__(self, out_folder):

        self.out_folder = out_folder

    @staticmethod
    def get_predictions(ensemble_results):
        list_predictions = []
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
            folder_out_sample = self.out_folder[name]["folders"]["results"]
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

    def __init__(self):

        self.software_result = dict()
        self.software_exec = dict()
        self.phenotype_result = dict()
        self.proteins_results = list()
        self.neighbors_results = list()

    def add_software_exec(self, config:dict):
        self.software_exec["type"] = "software_exec"
        self.software_exec["key"] = ""
        self.software_exec["software_name"] = "PathogenFinder2"
        self.software_exec["command"] = ""
        self.software_exec["parameters"] = {}
        self.software_exec["parameters"]["inputfile"] = config.inference_parameters["Input Data"]
        self.software_exec["parameters"]["Produce Attentions"] = config.inference_parameters["Attentions"]
        self.software_exec["parameters"]["Produce Embeddings"] = config.inference_parameters["Embeddings"]
        self.software_exec["parameters"]["Sequence Format"] = config.inference_parameters["Sequence Format"]
        self.software_exec["stdout"] = ""
        self.software_exec["stderr"] = ""



    def add_software_result(self):
        # TODO UPDATE AUTOMATIC
        # TODO software version, branch
        self.software_result["type"] = "software_result"
        self.software_result["key"] = "PathogenFinder2-0.0.4"
        self.software_result["software_name"] = "PathogenFinder2"
        self.software_result["software_version"] = "0.0.4"
        self.software_result["software_branch"] = ""
        self.software_result["software_commit"] = ""
        self.software_result["software_log"] = ""
        self.software_result["run_id"] = ""
        self.software_result["run_date"] = ""
        self.software_result["phenotypes"] = ""
        self.software_result["software_exections"] = {}
        
    def add_phenotype_result(self, results_ensemble):
        self.phenotype_result["type"] = "phenotype_ml"
        self.phenotype_result["key"] = "Human Bacterial Pathogenicity"
        self.phenotype_result["category"] = "Pathogenicity"
        self.phenotype_result["ensemble_pred"] = True
        self.phenotype_result["type_pred"] = "Categorical"
        self.phenotype_result["prediction"] = results_ensemble["Phenotype"]
        self.phenotype_result["output_model"] = {}
        for n in ["0", "1", "2", "3"]:
            self.phenotype_result["Prediction_{}".format(n)] = round(results_ensemble["Prediction_{}".format(n)],4)
        self.phenotype_result["output_mean"] = round(results_ensemble["Prediction Mean"], 4)
        self.phenotype_result["output_std"] = round(results_ensemble["Prediction STD"], 4)
    
    def add_bacterialneighbors(self, query_id:str, neighbors_df:pd.DataFrame):
        for n in range(len(neighbors_df)):
            entry = neighbors_df.iloc[n]
            neighbor = {}
            neighbor["key"] = "{}_{}".format(query_id, entry["Names"])
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
            self.neighbors_results.append(neighbor)
            

    def add_proteinsatt(self, proteins_df):
        for n in range(len(proteins_df)):
            entry = proteins_df.iloc[n]
            protein = {}
            protein["key"] = "{}_{}".format(entry["Query_ID"], entry["Ref_ID"])
            protein["gene"] = "Protein"
            protein["name"] = entry["Ref_name"]
            protein["identity"] = entry["Identity"].item()
            protein["alignment_length"] = entry["Alignment_Length"].item()
            protein["ref_seq_lenght"] = entry["Ref_Length"].item()
            protein["coverage"] = entry["Ref_coverage"].item()
            protein["ref_id"] = entry["Ref_ID"]
            protein["ref_acc"] = entry["Ref_ID"]
            protein["ref_start_pos"] = entry["Ref_start_pos"].item()
            protein["ref_end_pos"] = entry["Ref_end_pos"].item()
            protein["query_id"] = entry["Query_ID"]
            protein["query_start_pos"] = entry["Query_start_pos"].item()
            protein["query_end_pos"] = entry["Query_end_pos"].item()
            protein["ref_database"] = "UniRef50"
#            protein["note"] = "Attention Score {}".format(entry["Attention Value"])
            if protein["coverage"] == 100. and protein["identity"] == 100.:
                protein["grade"] = 3
            elif protein["coverage"] == 100. and protein["identity"] < 100.:
                protein["grade"] = 2
            elif protein["coverage"] < 100.:
                protein["grade"] = 1
            else:
                protein["grade"] = 0

            self.proteins_results.append(protein)

    def save_results(self, output_path):
        results = {"software_result": self.software_result,
                   "phenotype_ml": self.phenotype_result,
                   "software_executions": self.software_exec,
                   }
        if len(self.neighbors_results) > 0:
            results["pathogenic_neighbors"] = self.neighbors_results
        if len(self.proteins_results) > 0:
            results["protein_results"] = self.proteins_results
        with open("{}/cge_output.json".format(output_path), 'w') as f:
            json.dump(results, f)

