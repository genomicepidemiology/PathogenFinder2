import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
import pytorch_warmup as warmup
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import sys
import time
from datetime import datetime
from torchvision import transforms
import matplotlib.pyplot as plt
import wandb
from sklearn.utils import resample
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay, roc_curve, PrecisionRecallDisplay, precision_recall_curve, det_curve, DetCurveDisplay

from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM, EigenCAM

sys.dont_write_bytecode = True

from data_utils import ProteomeDataset, ToTensor, Normalize_Data, BucketSampler, FractionEmbeddings, PhenotypeInteger
from data_utils import NN_Data
from results_record import Json_Results
from utils import NNUtils, Metrics
from torch.optim import swa_utils



class Test_NeuralNet:

    def __init__(self, network, configuration, mixed_precision=None, results_dir=None, results_module=None):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, garbage_collection_threshold:0.6'
        torch.cuda.empty_cache()
        self.device = NNUtils.get_device()
        print("Testing on {}".format(self.device))
        print(network)
        self.network = network.to(self.device)
        self.configuration = configuration
        self.results_dir = results_dir
        self.results_module = results_module
        #if method_interpret == "GradCAM++":
            #self.method_interpret = GradCAMPlusPlus(model=self.network,
         #                                           target_layers=[self.network.features[-1]],
          #                                          reshape_transform=Test_NeuralNet.reshape_transform)

    @staticmethod
    def reshape_transform(tensor, height=14, width=14):
       # print(tensor.shape)
 #       result = tensor[:, 1:, :].reshape(tensor.size(0),
  #                                    tensor.size(1), tensor.size(2))
        result = tensor
        # Bring the channels to the first dimension,
        # like in CNNs.
 #       result = result.transpose(2, 3).transpose(1, 2)
       # print(result.shape)
        return result


    @staticmethod
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook



    def __call__(self, test_dataset, asynchronity, num_workers, batch_size, report_att, bucketing, stratified, return_layer=False):

        start_time = time.time()
 #       batch_size = 1

        test_loader = NN_Data.load_data(test_dataset, batch_size, num_workers=num_workers, stratified=False,
                                             shuffle=True, pin_memory=asynchronity, bucketing=bucketing, drop_last=True)
        len_dataloader = len(test_loader)
        predictions_lst = []
        labels_lst = []
        labels_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device, dtype=int)
        pred_tensor = torch.empty((len_dataloader*batch_size, self.network.num_classes), device=self.device)
#        labels_tensor = torch.empty((64, self.network.num_classes), device=self.device, dtype=int)
 #       pred_tensor = torch.empty((64, self.network.num_classes), device=self.device)
        lengths_lst = []
        filenames_lst = []

        features_lst = []
        
        
        if report_att and not os.path.isdir("{}/attention_vals_test".format(self.results_dir)):
            os.mkdir("{}/attention_vals_test".format(self.results_dir))

        self.network.eval()
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        count = 0
   #     if return_layer:
  #          self.network.avgpool.register_forward_hook(get_activation("{}".format(return_layer)))
        difference_length = []
        with torch.inference_mode():
            for batch in tqdm(test_loader):
                pos_first, pos_last = count, count+batch_size
                embeddings = batch["Input"]
                labels = batch["PathoPhenotype"]
                lengths = batch["Protein Count"]
                filename = batch["File_Names"]
                prot_name = batch["Protein_IDs"]
                difference_length.append(max(lengths)-min(lengths))
                labels_tensor[pos_first:pos_last,:] = labels
#                labels_lst.extend(labels.reshape(len(labels),).tolist())
                lengths_lst.extend(lengths.reshape(len(lengths,)).tolist())
                filenames_lst.extend(filename)
                #  sending data to device
                embeddings = embeddings.to(self.device, non_blocking=asynchronity)
                labels = labels.to(self.device, non_blocking=asynchronity)
                lengths = lengths.to(self.device, non_blocking=asynchronity)
                #  making predictions
                predictions_logit, attentions = self.network(embeddings, lengths)
                predictions, loss = self.calculate_loss(predictions_logit, labels)
                pred_c = predictions.detach()
                pred_tensor[pos_first:pos_last,:] = pred_c
                # interpet
                count += batch_size
 #               break
                if report_att:
                    attentions = attentions.to("cpu")
                    for filename_, attentions_, prot_name_ in zip(filename, attentions, prot_name):
                        prot_name_ = np.array(prot_name_)
                        attentions_ = attentions_.squeeze().numpy()[:len(prot_name_)]
                        np.savez_compressed("{}/attention_vals_test/{}".format(self.results_dir, os.path.basename(filename_)),
                                            attentions=attentions_, protein_IDs=prot_name_)
        exec_time = time.time() - start_time
 #       features = np.concatenate(features_lst)
  #      np.savez_compressed("{}/features".format(self.results_dir), return_layer=features)
        results_df = pd.DataFrame({"Filename": filenames_lst, "Protein_Count": lengths_lst})
        if pred_tensor.size()[1] == 2:
            pred_tensor = pred_tensor[:,0] - pred_tensor[:,1]
            pred_tensor = (pred_tensor+1)/2
            labels_tensor = labels_tensor[:,0] - labels_tensor[:, 1]
            labels_tensor = (labels_tensor+1)/2
        else:
            pred_tensor = pred_tensor
            labels_tensor = labels_tensor
        results_df["Predictions (Label)"] = 0
        results_df["Correct Label"] = labels_tensor.tolist()
        results_df["Predictions"] =  pred_tensor.tolist()
        results_df.loc[results_df["Predictions"]>0.5, "Predictions (Label)"] = 1
        results_df.to_csv("{}/predictions_test.tsv".format(self.results_dir), sep="\t", index=False)

        metrics = self.calculate_metrics(results_df=results_df, bootstrap=1)

        with open("{}/results_test.txt".format(self.results_dir), "w") as res_file:
            res_file.write("Test Results File\n")
            res_file.write("-----------------------------------------------------------------------\n")
            res_file.write("\n")
            for k, val in metrics.items():
                res_file.write("{}: {}({})\n".format(k, val["Mean"], val["Std"]))
                self.results_module.test_results(k, val["Mean"], val["Std"])
            res_file.write("Exec time: {}\n".format(exec_time))
            self.results_module.test_results("Exec time", exec_time, None)
            self.results_module.test_results("Loss", loss,None)

        cm_display, roc_display, pr_display, det_display = self.get_graphs(data=results_df)

        self.mcc_length(results_df=results_df)

        self.make_graph(display=cm_display, name_file="confusion_matrix",
                       title="Confusion Matrix")
        self.make_graph(display=roc_display, name_file="ROC_curve",
                       title="ROC Curve")
        self.make_graph(display=pr_display, name_file="precision_recall",
                       title="Precision Recall")
#        self.make_graph(display=det_display, name_file="det",
 #                      title="DET")

    def mcc_length(self, results_df, ranges=1000):
        mcc_values = []
        count_samples = []
        length_ranges = []
        new_data = []
        new_data2 = []
        init_count = 0
        while init_count < 8000:
            last_count = init_count + ranges
            length_ranges.append("{}-{}".format(init_count, last_count))
            length_range = results_df.loc[(results_df["Protein_Count"]<last_count)&(results_df["Protein_Count"]>init_count)]
            mcc = Metrics.calculate_MCC(labels=torch.tensor(length_range["Correct Label"].values),
                                            predictions=torch.tensor(length_range["Predictions (Label)"].values), device="cpu")
            count_samples.extend(length_range["Protein_Count"].tolist())
            mcc_values.append(mcc.item())
            new_data.append(["{}-{}".format(init_count, last_count), mcc.item()])
            new_data2.append(["{}-{}".format(init_count, last_count), len(length_range)])
            init_count += ranges
        table = wandb.Table(data=new_data, columns = ["MCC", "Gene Count"])
        wandb.log({"mcc_genecount" : wandb.plot.bar(table, "MCC", "Gene Count",
                               title="MCC vs gene count")})
        table = wandb.Table(data=new_data2, columns = ["Gene Count", "Count"])
        wandb.log({"count_genecount" : wandb.plot.bar(table, "Gene Count", "Count",
                               title="Amount at gene count")})

    def calculate_loss(self, predictions_logit, labels):
        predictions = torch.sigmoid(predictions_logit)
        loss = torch.nn.modules.loss.BCEWithLogitsLoss()(predictions_logit, labels)
        return predictions, loss

    def make_graph(self, display, name_file, title):
        fig, ax = plt.subplots()
        display.plot(ax=ax)
        plt.title(title)
        plt.savefig("{}/{}.png".format(self.results_dir, name_file))
        self.results_module.log_plot(fig=fig, name=title)
        plt.close()

    def get_graphs(self, data):
        y_test = data["Correct Label"]
        y_pred = data["Predictions (Label)"]
        y_score = data["Predictions"]

        cm_display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)

        prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=1)
        pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)

        fpr, fnr, _ = det_curve(y_test, y_pred)

        det_display = DetCurveDisplay(fpr=fpr, fnr=fnr)
        return cm_display, roc_display, pr_display, det_display
       

    def get_metrics(self, data):
        b_acc = balanced_accuracy_score(data["Correct Label"], data["Predictions (Label)"])
        f1 = f1_score(data["Correct Label"], data["Predictions (Label)"])
        precision = precision_score(data["Correct Label"], data["Predictions (Label)"])
        recall = recall_score(data["Correct Label"], data["Predictions (Label)"])
        rocauc = roc_auc_score(data["Correct Label"], data["Predictions (Label)"])
        mcc =  Metrics.calculate_MCC(labels=torch.tensor(data["Correct Label"].values),
                                predictions=torch.tensor(data["Predictions (Label)"].values), device="cpu")
        return b_acc, f1, precision, recall, rocauc, mcc

    def calculate_metrics(self, results_df, bootstrap=False):
        if bootstrap:
            bacc_lst = []
            f1_lst = []
            precision_lst = []
            recall_lst = []
            rocauc_lst = []
            mcc_lst = []
            for i in range(bootstrap):
                ind_boot = resample(results_df.index, n_samples=len(results_df))
                boot_data = results_df.iloc[ind_boot]
                _b_acc, _f1, _precision, _recall, _rocauc, _mcc = self.get_metrics(data=results_df)
                bacc_lst.append(_b_acc)
                f1_lst.append(_f1)
                precision_lst.append(_precision)
                recall_lst.append(_recall)
                rocauc_lst.append(_rocauc)            
                mcc_lst.append(_mcc)
            b_acc, b_acc_std = np.mean(bacc_lst), np.std(bacc_lst)
            f1, f1_std = np.mean(f1_lst), np.std(f1_lst)
            precision, precision_std = np.mean(precision_lst), np.std(precision_lst)
            recall, recall_std = np.mean(recall_lst), np.std(recall_lst)
            rocauc, rocauc_std = np.mean(rocauc_lst), np.std(rocauc_lst)
            mcc, mcc_std = np.mean(mcc_lst), np.std(mcc_lst)
        else:
            b_acc, f1, precision, recall, rocauc, mcc = self.get_metrics(data=results_df)
            b_acc_std, f1_std, precision_std, recall_std, rocauc_std, mcc_std = None
        metrics = {"Balanced_Accuracy": {"Mean":b_acc, "Std":b_acc_std}, "F1": {"Mean":f1, "Std":f1_std},
                   "Precision": {"Mean":precision, "Std":precision_std}, "Recall": {"Mean":recall, "Std":recall_std},
                   "ROC_AUC": {"Mean":rocauc, "Std":rocauc_std}, "MCC": {"Mean": mcc, "Std":mcc_std}}
        return metrics



