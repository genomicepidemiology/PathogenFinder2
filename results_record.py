import wandb
import pickle
from torchmetrics.classification import BinaryMatthewsCorrCoef
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
import numpy as np

class Results:

    def __init__(self):
        pass


class Wandb_Results:

    batch_checkpoint = 30

    def __init__(self, configuration, name="", wandb_dir=None):

        self.wandb_run = wandb.init(project="PathogenFinder2", name=name,
                               config=configuration, dir=wandb_dir)

    def count_params(self, model):
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return pytorch_total_params

    def start_train_report(self, model, criterion, log="all"):
        self.wandb_run.watch(model, criterion, log="all", log_freq=1)
        model_params = self.count_params(model)
        self.wandb_run.summary["Trainable parameters"] = model_params
        self.step_wandb = 0
        self.epoch_n = 0

    def add_epoch_info(self, loss_t, loss_v, epoch, mcc_t, mcc_v, loss_t_eval=None, mcc_t_eval=None):

        if loss_t_eval is None:
            log_results = {"Training Loss": loss_t,
                       "Validation Loss": loss_v, "Training MCC": mcc_t,
                       "Validation MCC": mcc_v, "Epoch": epoch
                       }
        else:
            log_results = {"Training Loss": loss_t,
                       "Validation Loss": loss_v, "Training MCC": mcc_t,
                       "Validation MCC": mcc_v, "Epoch": epoch,
                       "Training Loss Eval": loss_t_eval, "Training MCC Eval": mcc_t_eval
                       }
 #       print(log_reults)
        wandb.log(log_results, step=self.step_wandb)
        self.epoch_n += 1

    def add_step_info(self, loss_train, lr, batch_n, len_dataloader):
        if batch_n % Wandb_Results.batch_checkpoint == Json_Results.batch_checkpoint-1 and self.wandb_run:
            wandb.log({"Training Loss/Step": loss_train, "Learning Rate": lr, "Epoch": self.epoch_n + ((batch_n+1)/len_dataloader)}, step=self.step_wandb)
            self.step_wandb += 1
#            self.step_wandb += Wandb_Results.batch_checkpoint


    def add_time_info(self, epoch_duration):
        wandb.log({"Epoch Runtime (seconds)": epoch_duration}, step=self.step_wandb)

    def test_results(self, k, mean, std):
        self.wandb_run.summary["test_{}_mean".format(k)] = mean
        self.wandb_run.summary["test_{}_std".format(k)] = std
    
    def finish_report(self):
        self.wandb_run.finish()
    
    def log_plot(self, fig, name):
        self.wandb_run.log({name: fig}) 

  

class Json_Results:
    
    batch_checkpoint = 2

    def __init__(self, wandb_results=False, configuration=None, name="",
			model=None, criterion=None):

        self.epochs_training = {}
        self.last_model_data = {"Protein Names": list(),
			       "Genome Names": list(),
				"Attentions": list()}
        self.error_data = {}
        if wandb_results:
            self.wandb_run = self.initiate_wandb(project="PathogenFinder2", name=name,
					configuration=configuration, wandb_results=wandb_results)
            self.wandb_run.watch(model, criterion, log="all", log_freq=1)
            self.step_wandb = 0
        else:
            self.wandb_run = False
        self.epoch_n = 0


    def initiate_wandb(self, project, name, configuration, wandb_results):
        run = wandb.init(project=project, name=name, config=configuration, dir=wandb_results)
        return run


    def log_wandb(self, loss_t, loss_v, epoch, learning_rate, mcc_t, mcc_v):

        log_results = {"Training Loss": loss_t,
                   "Validation Loss": loss_v, "Training MCC": mcc_t,
                   "Validation MCC": mcc_v,
		   }
        self.wandb_run.log(log_results, step=self.step_wandb)

    def step_log(self, loss_train, lr, batch_n, len_dataloader):
        if batch_n % Json_Results.batch_checkpoint == Json_Results.batch_checkpoint-1 and self.wandb_run:
            self.wandb_run.log({"Training Loss/Step": loss_train, "Learning Rate": lr, "Epoch": self.epoch_n + ((batch_n+1)/len_dataloader)}, step=self.step_wandb)
            self.step_wandb += 1

    def time_log(self, epoch_duration):
        if self.wandb_run:
            self.wandb_run.log({"Epoch Runtime (seconds)": epoch_duration}, step=self.step_wandb)

    @staticmethod
    def _add_data(add_to, data):
        if isinstance(data, list):
            add_to.extend(data)
        else:
            add_to.append(data)

    def add_epoch_report(self, epoch, loss_t, loss_v, lr, mcc_t, mcc_v):

        if self.wandb_run:
            self.log_wandb(loss_t=loss_t, loss_v=loss_v, epoch=epoch, learning_rate=lr, mcc_t=mcc_t, mcc_v=mcc_v)
        self.epoch_n += 1

    def add_lastmodel_data(self, prot_names, genome_names, attentions):
        Json_Results._add_data(add_to=self.last_model_data["Protein Names"], data=prot_names)
        Json_Results._add_data(add_to=self.last_model_data["Genome Names"], data=genome_names)
        Json_Results._add_data(add_to=self.last_model_data["Attentions"], data=attentions)

    def add_error_batch(self, epoch, subset, names):
        if not epoch in self.error_data:
            self.error_data[epoch] = {}
        if not subset in self.error_data[epoch]:
            self.error_data[epoch][subset] = list()
        self.error_data[epoch][subset].append(names)

    def save_results(self, results_dir):
        if self.wandb_run:
            self.wandb_run.finish()
        results_dict = {"Training Process": self.epochs_training,
                        "Last model": self.last_model_data,
                        "Error data": self.error_data}
        with open("{}/train_results.pkl".format(results_dir), 'wb') as f:
            pickle.dump(results_dict, f)
