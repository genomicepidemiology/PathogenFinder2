import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
from sklearn.calibration import calibration_curve


class Test_NeuralNetwork:

    def __init__(self, out_folder):
        self.out_folder = "{}/test_results/".format(out_folder)
        os.mkdir(self.out_folder)


    def create_results(self, label_df, predicted_data):
        label_df["Predicted Label"] = np.nan
        label_df["Predicted Cont"] = np.nan
        label_df["Predicted Std"] = np.nan
        label_df["Predicted Phenotype"] = None
        for k, val in predicted_data.items():
            label_df.loc[label_df["File Name"]==val["Features"]["Filename"],"Predicted Label"] = val["Ensemble Predictions"]["Binary Prediction Mean"]
            label_df.loc[label_df["File Name"]==val["Features"]["Filename"],"Predicted Cont"] = val["Ensemble Predictions"]["Prediction Mean"]
            label_df.loc[label_df["File Name"]==val["Features"]["Filename"],"Predicted Std"] = val["Ensemble Predictions"]["Prediction STD"]
            label_df.loc[label_df["File Name"]==val["Features"]["Filename"],"Predicted Phenotype"] = val["Ensemble Predictions"]["Phenotype Mean"]
        print(label_df[pd.isnull(label_df).any(axis=1)])
        print(label_df.isnull().values.any())
        assert not label_df.isnull().values.any()
        label_df.to_csv("{}/test_results.tsv".format(self.out_folder), sep="\t", index=False)
        return label_df

    def calculate_measures(self, results_df):
        mcc = matthews_corrcoef(results_df["PathoPhenotype"], results_df["Predicted Label"])
        bacc = balanced_accuracy_score(results_df["PathoPhenotype"], results_df["Predicted Label"])
        rocauc = roc_auc_score(results_df["PathoPhenotype"], results_df["Predicted Label"])
        sensit = recall_score(results_df["PathoPhenotype"], results_df["Predicted Label"], pos_label=1)
        specifi = recall_score(results_df["PathoPhenotype"], results_df["Predicted Label"], pos_label=0)
        return {"MCC": mcc, "BACC": bacc, "ROCAUC": rocauc, "Sensitivity": sensit, "Specificity": specifi}

    def graph_objects(self, results_df):
        cm = confusion_matrix(results_df["PathoPhenotype"], results_df["Predicted Label"])
        cm_display = ConfusionMatrixDisplay(cm)
        fpr, tpr, thresholds = roc_curve(results_df["PathoPhenotype"], results_df["Predicted Label"])
        roc_auc = auc(fpr, tpr)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

        fraction_of_positives, mean_predicted_value = calibration_curve(results_df["PathoPhenotype"],
                                                                            results_df["Predicted Cont"], n_bins=10)

        curve_cal = {"frac_pos": fraction_of_positives, "mean_pred": mean_predicted_value}
        return cm_display, roc_display, curve_cal


    def save_results(self, measures, cm_display=None, roc_display=None, curve_cal=None):
        measures_file = "{}/measures.txt".format(self.out_folder)
        with open(measures_file, "w") as measuresf:
            measuresf.write("======== Test Results ==========\n\n")
            for k, v in measures.items():
                measuresf.write("{}: {}\n".format(k, v))
        fig, ax = plt.subplots()
        cm_display.plot()
        plt.savefig("{}/cfm.png".format(self.out_folder))
        plt.close()
        fig, ax = plt.subplots()
        roc_display.plot()
        plt.savefig("{}/roc.png".format(self.out_folder))
        plt.close()
        fig, ax = plt.subplots()
        ax.plot(curve_cal["mean_pred"], curve_cal["frac_pos"], "s-")
        plt.savefig("{}/cal_curve.png".format(self.out_folder))
        plt.close()




