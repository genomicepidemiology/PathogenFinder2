import os
import json
import pandas as pd

from pathlib import Path
from collections import UserDict

from pathogenfinder2.utils.os_utils import create_outputfolder, read_multifiles
from pathogenfinder2.utils.configuration_utils import NNEncoder, ParamsModel



class ConfigurationPF2(UserDict):

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, mode:str, user_config:[str,dict,bool]=False):
        
        self.data = {}
        self.mode = mode
        if not user_config:
            config_data = ConfigurationPF2.load_baseconfig()
            config_data = ConfigurationPF2.load_baseweights(config_data=config_data)
        else:
            config_data = self.load_userconfig(user_config=user_config)
        config_data["Misc Parameters"]["Actions"] = mode
        self.update(config_data)

    @staticmethod
    def load_baseconfig():
        config_file = Path(ConfigurationPF2.CURRENT_DIR) / '../data/configs/config_base.json'
        with open(config_file) as f:
            config_dict = json.load(f)
        return config_dict
    
    @staticmethod
    def load_baseweights(config_data):
        for i in [1, 2, 3, 4]:
            weight_file = Path(ConfigurationPF2.CURRENT_DIR) / '../data/models_weights/weights_model{}.pickle'.format(i)
            config_data["Model Parameters"]["Network Weights"].append(weight_file)
        return config_data

    @staticmethod
    def get_bplfile():
        return Path(ConfigurationPF2.CURRENT_DIR) / '../data/bpl/embeddings.npz'


    def load_user_config(self, user_config):
        if isinstance(user_config, str):
            with open(user_config) as f:
                config_dict = json.load(f)
        else:
            config_dict = user_config
        return config_dict


class Files_Module(UserDict):

    def __init__(self, outputFolder:str, mode:str):
        outputFolder = os.path.abspath(outputFolder)
        if not os.path.isdir(outputFolder):
            os.mkdir(outputFolder)
        if mode == "Prediction":
            mode_out = "{}/predictionPF2".format(outputFolder)
        elif mode == "Train":
            mode_out = "{}/trainPF2".format(outputFolder)
        elif mode == "Test":
            mode_out = "{}/testPF2".format(outputFolder)
        elif mode == "Map_Embeddings":
            mode_out = "{}/mapembedPF2".format(outputFolder)
        elif mode == "Infere":
            mode_out = "{}/inferePF2".format(outputFolder)
        elif mode == "Align_Proteins":
            mode_out = "{}/alnprotPF2".format(outputFolder)
        else:
            raise ValueError("The mode {} is not available to produce the base output".format(mode))
        if os.path.isdir(mode_out):
            raise OSError("""The folder {} seems to have been used already as """
                        """the output of a previous PathogenFinder2 {} run. """
                        """Please output the results in a different folder.""".format(
                            outputFolder, mode))
        os.mkdir(mode_out)

        self.data = {"base_folder": mode_out, "data_files":{}, "folders":{}}


    def create_nestedoutput(self, format_seq:str, input_file:[bool,str],
                                multi_file:[bool, str]=False):
        
        input_files, input_names = self.add_input(input_file=input_file, multi_file=multi_file)

        self.data["folders"]["results"] = {}
        self.data["folders"]["log"] = {}

        if format_seq in ["genome", "proteome"]:
            self.data["folders"]["preprocess"] = {}
        
        
        
        for seq, base_seq in zip(input_files, input_names):
            #self.data["sequences"][base_seq] = {"data_files":{}, "folders":{}}
            self.data["data_files"]["input_sequence"][base_seq] = seq
            
            results_folder_seq = "{}/results_{}".format(self.data["base_folder"], base_seq)
            self.data["folders"]["results"][base_seq] = results_folder_seq
            os.mkdir(results_folder_seq)
            log_folder_seq = "{}/log_{}".format(self.data["base_folder"], base_seq)
            self.data["folders"]["log"][base_seq] = log_folder_seq
            os.mkdir(log_folder_seq)

            if format_seq in ["genome", "proteome"]:
                preproc_folder_seq = "{}/preprocess_{}".format(self.data["base_folder"], base_seq)
                self.data["folders"]["preprocess"][base_seq] = preproc_folder_seq
                os.mkdir(preproc_folder_seq)
            else:
                preproc_folder_seq = None

            
            if format_seq == "genome":
                self.data["data_files"]["genome_sequence"][base_seq] = self.data["data_files"]["input_sequence"][base_seq]
                self.data["data_files"]["proteome_sequence"][base_seq] = "{}/PredictedProteins.faa".format(preproc_folder_seq)
            else:
                self.data["data_files"]["genome_sequence"][base_seq] = None
                self.data["data_files"]["proteome_sequence"][base_seq] = self.data["data_files"]["input_sequence"][base_seq]
            if format_seq in ["genome", "proteome"]:
                self.data["data_files"]["embedding_file"][base_seq] = "{}/ProteinEmbeddings.h5".format(preproc_folder_seq)
            else:
                self.data["data_files"]["proteome_sequence"][base_seq] = None
                self.data["data_files"]["embedding_file"][base_seq] = self.data["data_files"]["input_sequence"][base_seq]

    def add_input(self, input_file:[bool,str], multi_file:[bool, str]=False):
        if not input_file and not multi_file:
            raise ValueError("Input_file or multifile is required")
        elif multi_file and input_file:
            raise ValueError("Input_file and multifile are excluding options")
        elif multi_file:
            input_files, input_names = read_multifiles(multi_file)
        else:
            input_files = [os.path.abspath(input_file)]
            input_names = ["PF2"]
        self.data["input_files"] = input_names
        self.data["input_paths"] = input_files
        self.data["data_files"]["input_sequence"] = {}
        self.data["data_files"]["genome_sequence"] = {}
        self.data["data_files"]["proteome_sequence"] = {}
        self.data["data_files"]["embedding_file"] = {}

        return input_files, input_names
    
    def postprocess_output(self):
        for base_seq in self.data["sequences"].keys():
            postproc_folder_seq = "{}/postprocess_{}".format(self.data["base_folder"], base_seq)
            self.data["folders"]["postprocess"] = postproc_folder_seq
            os.mkdir(postproc_folder_seq)

    def save_inputmetadata(self):
        input_files = []
        genome_files = []
        protein_files = []
        embedding_files = []
        for k, val in self.data["sequences"].items():
            input_files.append(val["data_files"]["input_sequence"])
            genome_files.append(val["data_files"]["genome_sequence"])
            protein_files.append(val["data_files"]["proteome_sequence"])
            embedding_files.append(val["data_files"]["embedding_file"])

        metadata = pd.DataFrame({"Input_Files":input_files, "File_Genome":genome_files,
                            "File_Proteins":protein_files, "File_Embedding":embedding_files})
        self.data["input_metadata"]="{}/data_input.tsv".format(self.data["base_folder"])
        metadata.to_csv(self.data["input_metadata"], sep="\t", index=False)

    
    

    

class ConfigurationPF2_OLD:

    def __init__(self, mode:str, user_config:[str,dict]):

        self.mode = mode
        self.inference_parameters = None
        self.train_parameters = None
        self.test_parameters = None
        self.hyperopt_parameters = None

        if isinstance(user_config, str):

            self.misc_parameters = self.init_params(param_group="Misc Parameters",
                                    list_params=["Notes", "Name", "Actions", "Report Results", "Project Name", "Results Folder"])
            self.model_parameters = self.init_params(param_group="Model Parameters",
                                    list_params=["Model Name", "Input Dimensions", "Network Structure",
                                            "Out Dimensions", "Norm Scale", "Norm Type", "Data Parameters", "Attention Norm",
                                            "Mixed Precision", "Stochastic Depth Prob", "Sequence Dropout", "Attention Dropout",
                                            "Model Weights", "Batch Size", "Seed", "Stochastic Depth Prob Att", "Memory Report",
                                            "Loss Function", "Network Weights"])
            self.set_mode_parameters(mode=self.mode)

        else:
            std_json_path = "{}/../../../data/configs/config_empty.json".format(Path(__file__).parent.absolute())
            with open(std_json_path, "r") as stdjson:
                std_json = json.load(stdjson)
            self.misc_parameters = std_json["Misc Parameters"]
            self.model_parameters = std_json["Model Parameters"]
            if mode == "Inference":
                self.inference_parameters = std_json["Inference Parameters"]
            elif mode == "Train":
                self.train_parameters = std_json["Train Parameters"]
            elif mode == "Test":
                self.test_parameters = std_json["Test Parameters"]
            elif mode == "Hyperparam_Opt":
                self.hyperopt_parameters = std_json["Hyperparam_Opt Parameters"]
            else:
                raise ValueError("The mode {} is not available".format(mode))


    def set_mode_parameters(self, mode):
        if mode == "Train":
            self.train_parameters = self.init_params(param_group="Train Parameters",
                                        list_params=["Optimizer Parameters", "Epochs", "Loss Function", "Memory Report",
                                            "Save Model", "Report Results", "Train DF", "Train Loc",
                                            "Validation DF", "Validation Loc"])
        elif mode == "Inference":
            self.inference_parameters = self.init_params(param_group="Inference Parameters",
                                    list_params=["Preprocessing Parameters", "Sequence Format", "Input Data",
                                                    "Input Location", "Multiple Files", "Input Metadata",
                                                    "Attentions", "Embeddings"])
        elif mode == "Test":
            self.test_parameters = self.init_params(param_group="Test Parameters",
                                    list_params=["Input Data", "Label File", "Sequence Format", "Produce Attentions", "Produce Embeddings"])
        elif mode == "Hyperparam_Opt":
            self.hyperopt_parameters = self.init_params(parm_group="Hyperparam_Opt Parameters",
                                    list_params=["Optimizing Parameters", "Name Study", "Group",
                                            "Num Trials", "Load Study", "Storage", "Min Epochs Count",
                                            "Train DF", "Train Loc", "Validation DF", "Validation Loc",
                                            "Timeout", "Timeout", "Try Parameters", "Add Runs"])
        else:
            raise ValueError("The mode {} is not available".format(mode))


    def init_params(self, param_group, list_params):
        parameters = ParamsModel(name_params=param_group)
        parameters.init_params(list_params=list_params)
        return parameters

    def load_json_params(self, json_file):
        with open(json_file, "r") as f:
            json_data = json.load(f)
        for params, params_v in json_data.items():
            for k, val in params_v.items():
                if params == "Misc Parameters":
                    model_set = self.misc_parameters
                elif params == "Model Parameters":
                    model_set = self.model_parameters
                elif params == "Train Parameters":
                    model_set = self.train_parameters
                elif params == "Inference Parameters":
                    model_set = self.inference_parameters
                elif params == "Test Parameters":
                    model_set = self.test_parameters
                elif params == "Hyperparam_Opt Parameters":
                    model_set = self.hyperopt_parameters
                else:
                    raise ValueError("The Category '{}' in the JSON faile is not available".format(params))
                if model_set is None:
                    raise KeyError("The Parameters '{}' are in the JSON although the mode is '{}'".format(params, self.mode))
                else:
                    model_set.set_param(param=k, value=val)

        self.check_incompatible_params()


    def __str__(self):
        final_dict = self.collect_params()
        out_ = "{"
        for k, v in final_dict.items():
            out_ += "\n\t{}: {}".format(k, v)
        out_ += "\n}"
        return out_

    def check_incompatible_params(self):
        if self.inference_parameters["Multiple Files"] and self.inference_parameters["CGE Output"]:
            raise TypeError("CGE output can not be used with multiple files as input")
        
    def load_dict_params(self, dict_args):
        if dict_args["outputFolder"]:
            self.misc_parameters["Results Folder"] = create_outputfolder(outpath=os.path.abspath(dict_args["outputFolder"]))
        else:
            raise ValueError("It is necessary to set an output folder with --outputFolder when not using a config file")
        if dict_args["inputData"]:
            self.inference_parameters["Input Data"] = dict_args["inputData"]
        else:
            raise ValueError("It is necessary to set the input file with --inputData when not using a config file")
        if dict_args["formatSeq"]:
            self.inference_parameters["Sequence Format"] = dict_args["formatSeq"]
        else:
            raise ValueError("It is necessary to set what type of sequence with --formatSeq when not using a config file")
        self.inference_parameters["Multiple Files"] = dict_args["multiFiles"]
        self.inference_parameters["Embeddings"] = dict_args["embeddings"]
        self.inference_parameters["Attentions"] = dict_args["attentions"]
        self.inference_parameters["CGE Output"] = dict_args["cge"]
        
        self.misc_parameters["Prodigal Path"] = dict_args["prodigalPath"]
        self.misc_parameters["ProtT5 Path"] = dict_args["protT5Path"]
        self.misc_parameters["Diamond Path"] = dict_args["diamondPath"]


        if dict_args["weightsModel"]:
            files_weights = []
            for filew in dict_args["formatSeq"].split(","):
                files_weights.append(filew.strip())
            self.model_parameters["Network Weights"] = files_weights
        else:
            weights_path = "%s/../../../data/models_weights/weights_model{}.pickle" % Path(__file__).parent.absolute()
            files_weights = [weights_path.format(1), weights_path.format(2), weights_path.format(3), weights_path.format(4)]
            self.model_parameters["Network Weights"] = files_weights
        self.check_incompatible_params()

    def collect_params(self):
        final_dict = {"Misc Parameters": self.misc_parameters,
                            "Model Parameters": self.model_parameters,
                            "Train Parameters": self.train_parameters,
                            "Inference Parameters": self.inference_parameters,
                            "Test Parameters": self.test_parameters,
                            "Hyperparam Opt": self.hyperopt_parameters}
        return final_dict

    def save_json_params(self):
        final_dict = self.collect_params()
        file_save = "{}/config_run.json".format(self.misc_parameters["Results Folder"])
        with open(file_save, 'w') as f:
            json.dump(final_dict, f, cls=NNEncoder)


