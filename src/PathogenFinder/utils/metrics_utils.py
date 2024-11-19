from torchmetrics.classification import BinaryMatthewsCorrCoef



class Metrics:

    @staticmethod
    def calculate_metrics(predictions, labels):
        predictions_binary = np.where(np.array(predictions) > 0.5, 1, 0)
        acc = balanced_accuracy_score(labels, predictions_binary)
        mcc = matthews_corrcoef(labels, predictions_binary)
        return acc, mcc

    @staticmethod
    def calculate_MCC(predictions, labels, device):
        metric = BinaryMatthewsCorrCoef().to(device)
        mcc = metric(predictions, labels)
        return mcc
