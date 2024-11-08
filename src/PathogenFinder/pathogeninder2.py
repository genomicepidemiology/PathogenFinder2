import argparse


def cl_arguments():
    baseline_parser = argparse.ArgumentParser(prog='Pathogenfinder2.0 Model',
                            description="Arguments for pathogenicity inference, training,"
                                        " testing and hyperparameter selection of the model.")
    baseline_parser.add_argument("-c", "--config", help="Json file with the configuration for PathogenFinder2 model.",
                                required=True)  # ToDO: Not require, add a standard json in case of default=None
    baseline_parser.add_argument("-f", "--formatSeq", help="The format of the input data.",
                                    choices=["genome", "proteome", "embeddings"], required=True)
    baseline_parser.add_argument("-o", "--outputFolder", help="Folder where to output the results",
                                    required=True)
    baseline_parser.add_argument("--prodigalPath", help="Path to Prodigal", default="prodigal")
    baseline_parser.add_argument("--protT5Path", help="Path to protT5", default="protT5")
    
    train_parser = subparsers.add_parser("train", help="Train the PathogenFinder2 model based on certain data and configuration.")

    test_parser = subparsers.add_parser("test", help="Test the PathogenFinder2 model.")

    inference_parser = subparsers.add_parser("inference", help="Predict using PathogenFinder2 model.")
    inference_parser.add_argument("-w", "--weightsModel", help="Weights used by the deep learning model to predict")


def main(args):

    #Check the arguments
