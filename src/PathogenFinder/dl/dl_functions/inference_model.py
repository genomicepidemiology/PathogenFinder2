import pandas as pd
import torch

from dl.utils.data_utils import NN_Data
from dl.utils.nn_utils import Network_Module

class Inference_NeuralNetwork:

    def __init__(self, network_module, model_weights, out_folder, model_report=False):

        self.network_module = network_module
        self.network_module.load_model(model_weights)
        self.model_report = model_report
        self.out_folder = out_folder

        self.inference_loader = None
        self.asynchronity = None

    def set_dataloader(self, inference_dataset, num_workers=2, asynchronity=False,
                        max_batch_size=45):

        if len(inference_dataset) > max_batch_size:
            self.batch_size = max_batch_size
        else:
            self.batch_size = len(inference_dataset)

        self.asynchronity = asynchronity
        self.inference_loader = NN_Data.load_data(inference_dataset, self.batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=asynchronity, drop_last=False)

    def create_predresults(self, predictions, file_names, protids, attentions, lengths):
        pathopred = []
        protein_features = {}
        embedding_map = {}
        print(predictions.shape)
        for b in range(len(file_names)):
            filename = file_names[b]
            pathodict = {"File Name":filename, "Prediction":None, "Binary Prediction": None, "Protein Count": None, "Phenotype":""}
            pathodict["Prediction"] = float(predictions[int(b)])
#                pathodict["Prediction"] = float(predictions[b][n])
            if pathodict["Prediction"] > 0.5:
                pathodict["Binary Prediction"] = 1
                pathodict["Phenotype"] = "Human Pathogenic"
            else:
                pathodict["Binary Prediction"] = 0
                pathodict["Phenotype"] = "Human Non Pathogenic"
            pathodict["Protein Count"] = int(lengths[b])
            pathopred.append(pathodict)
#            attention_ind = torch.squeeze(attentions[b][n])[:int(lengths[b][n])]
 #           protein_features[filename] = pd.DataFrame({"ProteinIDs": protids[b][n], "Attentions": attention_ind})
        pathopred = pd.DataFrame(pathopred)
        return pathopred, protein_features, embedding_map
        

    def protein_analysis(self):
        pass

    def embedding_analysis(self):
        pass


    def __call__(self):

        (predictions, file_names, protID_tensor,
                att_tensor, protein_count) = self.network_module.predictive_pass(val_loader=self.inference_loader,
                                                                    asynchronity=self.asynchronity, batch_size=self.batch_size)
        if self.network_module.memory_profiler:
            self.network_module.memory_profiler.step()
    
        pathopred, protein_features, embedding_map = self.create_predresults(predictions=predictions, file_names=file_names,
                                    protids=protID_tensor, attentions=att_tensor, lengths=protein_count)
        if self.network_module.memory_profiler:
            self.network_module.memory_profiler.stop_memory_reports()

        return pathopred, protein_features, embedding_map


