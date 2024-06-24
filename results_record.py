import wandb
import pickle
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
import numpy as np



class Json_Results:

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
        else:
            self.wandb_run = False


    def initiate_wandb(self, project, name, configuration, wandb_results):
        run = wandb.init(project=project, name=name, config=configuration, dir=wandb_results)
        return run

    @staticmethod
    def calculate_metrics(predictions, labels):
        predictions_binary = np.where(np.array(predictions) > 0.5, 1, 0)
        acc = balanced_accuracy_score(labels, predictions_binary)
        mcc = matthews_corrcoef(labels, predictions_binary)
        return acc, mcc


    def log_wandb(self, loss_t, loss_v, labels_t, labels_v, predictions_t,
			predictions_v, epoch, learning_rate, mcc_t, mcc_v, 
			acc_t, acc_v):

        log_results = {"epoch": epoch, "Training_loss": np.nanmean(loss_t),
                   "Validation_loss": np.nanmean(loss_v), "Training_balanced_accuracy": acc_t,
                   "Validation_balanced_accuracy": acc_v, "Training_mcc": mcc_t, "Validation_mcc": mcc_v,
		   "Learning_rate": np.mean(learning_rate)}
        wandb.log(log_results, step=epoch)

    @staticmethod
    def _add_data(add_to, data):
        if isinstance(data, list):
            add_to.extend(data)
        else:
            add_to.append(data)

    def add_epoch_report(self, epoch, loss_t, loss_v, labels_t, labels_v, predictions_t,
				predictions_v, lr, mcc_t, mcc_v, acc_t, acc_v):

        training_results = {"Loss": loss_t, "Prediction": predictions_t, "Labels": labels_t}
        validation_results = {"Loss": loss_v, "Prediction": predictions_v, "Labels": labels_v}
        self.epochs_training[epoch] = {"Training": training_results,
				       "Validation": validation_results,
				       "Learning Rate": lr}

        if self.wandb_run:
            self.log_wandb(loss_t=loss_t, loss_v=loss_v, labels_t=labels_t, labels_v=labels_v,
				predictions_t=predictions_t, predictions_v=predictions_v, epoch=epoch,
			learning_rate=lr, mcc_t=mcc_t, mcc_v=mcc_v, acc_t=acc_t, acc_v=acc_v)

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
