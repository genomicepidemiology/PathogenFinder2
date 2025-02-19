import pickle
import wandb
import torch
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np


class Batch_Results:

    def __init__(self, filenames, predictions, protIDs, proteome_lengths, attentions=None, embeddings1=None,
            embeddings2=None):

        self.filenames = filenames
        self.predictions = predictions
        self.protIDs = protIDs
        self.attentions = attentions
        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.proteome_lengths = proteome_lengths

        assert len(self.filenames) == len(self.predictions)
        assert len(self.predictions) == len(self.protIDs)
        assert len(self.protIDs) == len(self.proteome_lengths)

        if attentions is not None:
            assert len(self.attentions) == len(self.proteome_lengths)
        if embeddings1 is not None:
            assert len(self.embeddings1) == len(self.proteome_lengths)
        if embeddings2 is not None:
            assert len(self.embeddings2) == len(self.proteome_lengths)

    def __len__(self):
        return len(self.filenames)

    def get_samples(self):
        samples = {}
        for n in range(len(self)):
#            name = Path(self.filenames[n]).stem
            name = self.filenames[n]
            samples[name] = {}
            samples[name]["Features"] = {}
            samples[name]["Features"]["Filename"] = [self.filenames[n]]
            samples[name]["Features"]["ProtIDs"] = self.protIDs[n][:int(self.proteome_lengths[n][0])]
            samples[name]["Output"] = {}
            samples[name]["Output"]["Prediction"] = self.predictions[n].tolist()
            samples[name]["Features"]["Proteome Length"] = self.proteome_lengths[n][0]
            if self.attentions is None:
                samples[name]["Output"]["Attention"] = [None]
            else:
                samples[name]["Output"]["Attention"] = self.attentions[n][:,:int(self.proteome_lengths[n][0])].numpy()
            if self.embeddings1 is None:
                samples[name]["Output"]["Embeddings1"] = [None]
            else:
                samples[name]["Output"]["Embeddings1"] = self.embeddings1[n].numpy()
            if self.embeddings2 is None:
                samples[name]["Output"]["Embeddings2"] = [None]
            else:
                samples[name]["Output"]["Embeddings2"] = self.embeddings2[n].numpy()
        return samples


class Inference_Report:

    def __init__(self, out_folder):

        self.out_folder = out_folder

    @staticmethod
    def get_predictions(ensemble_results):
        list_predictions = []
        for name, val in ensemble_results.items():
            predictionPF = pd.Series(val["Output"]["Prediction"], index=["Prediction_0", "Prediction_1", "Prediction_2", "Prediction_3"])
            predictionPF["Name"] = name
            predictionPF["Prediction Mean"] = np.mean(val["Output"]["Prediction"])
            predictionPF["Prediction STD"] = np.std(val["Output"]["Prediction"])
            if predictionPF["Prediction Mean"] > 0.5:
                predictionPF["Binary Prediction Mean"] = 1
                predictionPF["Phenotype Mean"] = "Human Pathogenic"
            else:
                predictionPF["Phenotype Mean"] = "Human Non Pathogenic"
                predictionPF["Binary Prediction Mean"] = 0
            ensemble_results[name]["Ensemble Predictions"] = predictionPF
        return ensemble_results

    def save_report(self, results_ensemble, save_attentions=True, save_embeddings=True):
        for name, val in results_ensemble.items():
            folder_out_sample = "{}/{}".format(self.out_folder, val["Features"]["Filename"])
            os.mkdir(folder_out_sample)
            val["Ensemble Predictions"].to_frame().to_csv("{}/predictions.tsv".format(folder_out_sample), sep="\t", index=False)
            if save_attentions:
                np.savez_compressed("{}/attentions.npz".format(folder_out_sample),
                                    protIDs=val["Features"]["ProtIDs"],
                                    attentions=val["Output"]["Attention"])
            if save_embeddings:
                np.savez_compressed("{}/embeddings.npz".format(folder_out_sample),
                                    embeddings_1=val["Output"]["Embeddings1"],
                                    embeddings_2=val["Output"]["Embeddings2"])



    def embeddingmap_results(self, embedding_maps_ensemble):
        pass


class CGEResults:

    def __init__(self):

        self.software_result = dict()
        self.phenotype_result = dict()

    def add_software_result(self):
        # TODO UPDATE AUTOMATIC
        self.software_result["type"] = "software_result"
        self.software_result["software_name"] = "PathogenFinder-2.0.0"
        self.software_result["software_version"] = "2.0.0"
        self.software_result["software_branch"] = "dev"
        self.software_result["software_commit"] = ""
        self.software_result["run_date"] = ""
        
    def add_phenotype_result(self, results_ensemble):
        self.phenotype_result["type"] = "phenotype_ml"
        self.phenotype_result["key"] = "Human Bacterial Pathogenicity"
        self.phenotype_result["category"] = "Pathogenicity"
        self.phenotype_result["ensemble_pred"] = True
        self.phenotype_result["type_pred"] = "Categorical"
        self.phenotype_result["prediction"] = results_ensemble["Phenotype Mean"]
        self.phenotype_result["output_model"] = {}
        for n in ["0", "1", "2", "3"]:
            self.phenotype_result["Prediction_{}".format(n)] = results_ensemble["Prediction_{}".format(n)]
        self.phenotype_result["output_std"] = results_ensemble["Prediction STD"]

    def save_results(self, output_path):
        results = {"software_result": self.software_result,
                   "phenotype_ml": self.phenotype_result
                   }
        with open("{}/cge_output.json".format(output_path), 'w') as f:
            json.dump(results, f)


class Memory_Report:

    def __init__(self, results_dir, process):
        self.results_dir = results_dir
        self.process = process
        self.prof = None

    def start_memory_reports(self, max_num_events_per_snapshot=1):
        memory_report = "{}/{}_memory-report".format(self.results_dir, self.process)
        torch.cuda.memory._record_memory_history(
            max_entries=max_num_events_per_snapshot)
        self.prof = torch.profiler.profile(
                        schedule=torch.profiler.schedule(wait=0, warmup=0, active=2),
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                       # on_trace_ready=torch.profiler.tensorboard_trace_handler("{}".format(memory_report)),
                        record_shapes=True, with_stack=True, profile_memory=True)
        self.prof.start()
        
    def step(self):
        self.prof.step()

    def stop_memory_reports(self):
        if self.prof is not None:
            self.prof.stop()
            print(torch.cuda.memory_summary())
 #           self.prof.export_memory_timeline("{}/{}_memory-record.html".format(self.results_dir, self.process))
            self.prof.export_chrome_trace("{}/{}_memory-record.json".format(self.results_dir, self.process))
            print(self.prof.key_averages().table())
            torch.cuda.memory._dump_snapshot("{}/{}_memory-record.pkl".format(self.results_dir, self.process))
            torch.cuda.memory._record_memory_history(enabled=None)
        else:
            raise ValueError("Profiler has not been started")

class ReportNN:

    def __init__(self, report_wandb, report_dict, name, path_dir,
                    configuration, modes=None, project=None):
        if report_wandb:
            self.report_wandb = Wandb_Report(name=name, wandb_dir=path_dir,
                                    configuration=configuration, project=project)
        else:
            self.report_wandb = False

        if report_dict:
            self.report_dict = File_Report(name=name, configuration=configuration,
                                     dict_dir=path_dir, modes=modes)
        else:
            self.report_dict = False

    def start_train_report(self, model, criterion):
        if self.report_wandb:
            self.report_wandb.start_train_report(model=model, criterion=criterion, log="all")

    def add_epoch_info(self, loss_t, loss_v, epoch, mcc_t, mcc_v, epoch_duration):
        log_results = {"Training Loss": loss_t, "Validation Loss": loss_v, "Training MCC": mcc_t,
                       "Validation MCC": mcc_v, "Epoch": epoch, "Epoch Runtime (seconds)": epoch_duration}
        if self.report_wandb:
            self.report_wandb.add_epoch_info(log_results)
        if self.report_dict:
            self.report_dict.add_epoch_info(log_results)

    def add_step_info(self, loss_train, lr, batch_n, len_dataloader):
        if self.report_wandb:
            self.report_wandb.add_step_info(loss_train, lr, batch_n, len_dataloader)

    def finish_report(self):
        if self.report_wandb:
            self.report_wandb.finish_report()
        if self.report_dict:
            self.report_dict.finish_report()


class File_Report:

    def __init__(self, configuration, name, dict_dir, modes):

        self.dict_file = "{}/{}".format(dict_dir, name)
        self.data = {}
        self.data["Configuration"] = configuration
        if "train" in modes:
            self.data["Train"] = {}
        elif "prediction" in modes:
            self.data["Prediction"] = {}
        elif "test" in modes:
            self.data["Test"] = {}

    def add_epoch_info(self, log_results):
        self.data["Train"]["Epoch {}".format(log_results["Epoch"])] = log_results

    def finish_report(self):
        with open("{}_results.pickle".format(self.dict_file), "wb") as f:
            pickle.dump(self.data, f)



class Wandb_Report:

    batch_checkpoint = 30

    def __init__(self, configuration, project, name="", wandb_dir=None):

        self.wandb_run = wandb.init(project=project, name=name, 
                                config=configuration, dir=wandb_dir)


    def start_train_report(self, model, criterion, log="all"):
        self.wandb_run.watch(model, criterion, log="all", log_freq=1)
        model_params = self.count_params(model)
        self.wandb_run.summary["Trainable parameters"] = model_params
        self.step_wandb = 0
        self.epoch = 0

    def count_params(self, model):
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return pytorch_total_params

    def add_epoch_info(self, log_results):
        self.epoch = log_results["Epoch"]
        wandb.log(log_results, step=self.step_wandb)

    def add_step_info(self, loss_train, lr, batch_n, len_dataloader):
        if batch_n % Wandb_Report.batch_checkpoint == 1 and self.wandb_run:
            wandb.log({"Training Loss/Step": loss_train, "Learning Rate": lr, "Epoch": self.epoch + ((batch_n+1)/len_dataloader)}, step=self.step_wandb)
            self.step_wandb += 1

    def finish_report(self):
        self.wandb_run.finish()

    def log_plot(self, fig, name):
        self.wandb_run.log({name: fig})


