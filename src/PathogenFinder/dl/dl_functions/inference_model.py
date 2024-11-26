import pandas as pd

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
        self.inference_loader = NN_Data.load_data(inference_dataset, self.batch_size,
                                        num_workers=num_workers, pin_memory=asynchronity)

    def create_predresults(self, predictions, file_names, out_folder):
        
        results_file = "{}/predictions.txt".format(out_folder)
        df_results = pd.DataFrame(data={"File Names":file_names,
                                        "Predictions": predictions})
        with open(results_file, 'w') as f:
            f.write('## RESULTS PATHOGENFINDER2 ##\n')
            df_results.to_csv(results_file, mode='a', index=False, sep="\t")

    def protein_analysis(self):
        pass

    def embedding_analysis(self):
        pass


    def __call__(self):

        predictions, file_names, protID_tensor, att_tensor = self.network_module.predictive_pass(val_loader=self.inference_loader,
                                                                    asynchronity=self.asynchronity, batch_size=self.batch_size)
        if self.network_module.memory_profiler:
            self.network_module.memory_profiler.step()
    
        self.create_predresults(predictions=predictions, file_names=file_names,
                                    out_folder=self.out_folder)
        if self.network_module.memory_profiler:
            self.network_module.memory_profiler.stop_memory_reports()


