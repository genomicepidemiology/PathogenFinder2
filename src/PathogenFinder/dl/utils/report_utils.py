import pickle
import wandb
import torch
import os
import pandas as pd
from pathlib import Path

class Inference_Report:

    def __init__(self, out_folder):

        self.out_folder = out_folder

    def reports_sample(self, pathopred_ensemble, prot_feat_ensemble, embedding_maps_ensemble):
        pathodf = self.pathopred_results(pathopred_ensemble)
        proteinfeats = self.proteinfeat_results(prot_feat_ensemble)
        report = {}
        for n in range(len(pathodf)):
            filepath = pathodf.loc[n, "File Name"]
            filename = Path(filepath).stem
      #      protfeat = proteinfeats[filepath]
       #     report[filename] = {"Predictions": pathodf.loc[n], "ProteinFeats": protfeat}
            report[filename] = {"Predictions": pathodf.loc[n]}
        return report

    def save_report(self, sample_report):
        for k, val in sample_report.items():
            folder_out_sample = "{}/{}".format(self.out_folder, k)
            os.mkdir(folder_out_sample)
            pred_df = pd.DataFrame(val["Predictions"]).T
            pred_df.to_csv("{}/predictions.tsv".format(folder_out_sample), sep="\t", index=False)
     #       val["ProteinFeats"].to_csv("{}/attentions.tsv".format(folder_out_sample), sep="\t", index=False)

    def pathopred_results(self, pathopred_ensemble):
        df = pathopred_ensemble[0]
        df = df.rename(columns={c: c+'_0' for c in df.columns if c not in ["File Name"]})
        pred_cols = ["Prediction_0"]
        count = 1
        for d in pathopred_ensemble[1:]:
            d_suff = d.rename(columns={c: c+'_{}'.format(count) for c in d.columns if c not in ["File Name"]})
            df = df.merge(d_suff, on='File Name')
            pred_cols.append("Prediction_{}".format(count))
            count += 1
        df["Prediction_mean"] = df[pred_cols].mean(axis=1)
        df["Prediction_std"] = df[pred_cols].std(axis=1)
        df["Binary Prediction_mean"] = 0
        df.loc[df["Prediction_mean"]>0.5, "Binary Prediction_mean"] = 1
        df["Phenotype_mean"] = "Human Non Pathogenic"
        df.loc[df["Prediction_mean"]>0.5, "Phenotype_mean"] = "Human Pathogenic"
        return df
        
    def proteinfeat_results(self, prot_feat_ensemble):
        protfeat_dict = {}
        pred_cols = []
        for n in range(len(prot_feat_ensemble)):
            pred_cols.append("Attentions_{}".format(n))
            for k, val in prot_feat_ensemble[n].items():
                protfeat_init = val.rename(columns={c: c+'_{}'.format(n) for c in val.columns if c not in ["ProteinIDs"]})
                if k not in protfeat_dict:
                    protfeat_dict[k] = protfeat_init
                else:
                    protfeat_dict[k] = protfeat_dict[k].merge(protfeat_init, on="ProteinIDs")
 #       for k, val in protfeat_dict.items():
  #          val["Attentions_mean"] = val[pred_cols].mean(axis=1)
   #         val["Attentions_std"] = val[pred_cols].std(axis=1)
    #        val["Attentions_sum"] = val[pred_cols].sum(axis=1)
        return protfeat_dict


    def embeddingmap_results(self, embedding_maps_ensemble):
        pass







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


