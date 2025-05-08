import argparse
import logging

from pathogenfinder2 import __version__

def check_arguments(argum):
    if argum.action == "Prediction":
        if argum.attProteins == "align" and not argum.dbProteins:
            raise ValueError(("""If wanted to align the highlighted proteins by PF2, """
                                """please use --dbProteins to point to the diamond indexed database"""))
        if not argum.inputFile and not argum.multipleFiles:
            raise ValueError(("""Argument -i/--inputFile or --multiFiles are required when predicting with pathogenfinder2"""))
       


def pf2_arguments():
    parser = argparse.ArgumentParser(prog='Pathogenfinder2',
                    description=("""Command line version of PathogenFinder2. Contains the options to """
                                """predict the pathogenic capacity using an already trained model, as well as """
                                """mapping the embedding of the genomic sequence to the Pathogenic Bacteria Landscape """
                                """and aligning the highlighted proteins by the model to a protein database.\n"""
                                """"It also include options for training your own model, as well as testing an already """
                                """trained model."""),
                    add_help=True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-v", "--version", action="version",
                                    version=__version__, help="Show program's version number and exit")
    parent_parser.add_argument('-d', '--debug', help="For debugging", action="store_const",
                                    dest="loglevel", const=logging.DEBUG,
                                    default=logging.WARNING)
    parent_parser.add_argument('--verbose', help="Be verbose", action="store_const", dest="loglevel", const=logging.INFO)

    in_out_parser = parent_parser.add_argument_group('General Input/Output options',
                                    description="General Input/Output options that can apply to any mode/functionality")
    in_out_parser.add_argument("-c", "--config", help=("""Config file (.json) with the configuration to run any of"""
                                                    """ the PathogenFinder2 functionalities (only recommended for experienced users)."""
                                                    """ It will overwrite any of the commandline arguments, """
                                                    """but provides full control of the model."""), default=False)  
    in_out_parser.add_argument("-o", "--outputFolder", help="Path to folder where to save the files produced by PathogenFinder2",
                                                required=True)
    exec_paths = parent_parser.add_argument_group('Executable paths',
                                    description=("""Paths to executables or tertiary software that might be required by PathogenFinder2."""
                                                """Only required if the software is not included on the executable PATH."""))
    exec_paths.add_argument("--prodigalPath", help="Path to Prodigal executable.", default="prodigal")
    exec_paths.add_argument("--protT5Path", help="Path to protT5", default="protT5")
    exec_paths.add_argument("--diamondPath", help="Path to Diamond executable", default="diamond")

    subparsers = parser.add_subparsers(title="PathogenFinder2 functionalities", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict bacterial pathogenic capacity with PathogenFinder2",
                                                parents=[parent_parser])
    predict_parser.set_defaults(action="Prediction")
    predict_parser.add_argument("-i", "--inputFile", help=("""Path to input for predicting the pathogenic capacity."""
                                                            """The format must be described in the -f/--formatSeq argument"""),
                                                        default=False)
    predict_parser.add_argument("--multipleFiles", help=("""Path to text file with paths to input files. This option allows for the"""
                                            """ prediction of bacterial pathogenic capacity of multiple bacterial genomes, by parallelizing """
                                            """ the neural network step."""), default=False)   
    predict_parser.add_argument("-f", "--formatSeq", help="The format of the bacterial sequence(s).",
                                                        choices=["genome", "proteome", "embeddings"], required=True)
    predict_parser.add_argument("-w", "--weightsModel", help=("""Path to the file(s) with weights used by"""
                                                """ the deep learning model to predict. If not selected, the model will use """
                                                """ the weights provided by the authors."""))
    predict_parser.add_argument("--embedProteome", help="If used, report or/and map the embeddings to the Bacterial Pathogenic Landscape",
                                  choices=[False, "report", "map"], default=False)
    predict_parser.add_argument("--attProteins", help="If used, report the attentions or/and align the 20 proteins with highest attention score to a protein database",
                                  choices=[False, "report", "align"], default=False)
    predict_parser.add_argument("--dbProteins", help="Path to protein database indexed with diamond to align the proteins highlighted by the attention layer",
                                default=False)
    predict_parser.add_argument("--cge", help="Save the predictions in the standard CGE output", action="store_true")

    train_parser = subparsers.add_parser("train",
                                        help="Train the PathogenFinder2 model for  bacterial pathogenic capacity using your own data",
                                        parents=[parent_parser])
    train_parser.set_defaults(action="Train")

    infer_parser = subparsers.add_parser("infere_proteomeLM",
                                        help="Predict the protein content and create its embeddings with Prodigal and protT5",
                                        parents=[parent_parser])
    infer_parser.set_defaults(action="Infere")
    infer_parser.add_argument("-i", "--inputFile", help=("""Path to genome file to predict its protein content and create embeddings."""),
                                                        default=False)
    infer_parser.add_argument("--multipleFiles", help=("""Path to text file with paths to input files."""), default=False)  

#    align_parser = subparsers.add_parser("align_proteins",
 #                                       help="Align the protein content to the protein database of interest",
  #                                      parents=[parent_parser])
   # align_parser.set_defaults(action="Align_Proteins")
#
 #   mapembed_parser = subparsers.add_parser("map_embedding",
  #                                      help="Map the genome embedding to the Pathogenic Bacterial Landscape",
   #                                     parents=[parent_parser])
    #mapembed_parser.set_defaults(action="Map_Embeddings")

    argums = parser.parse_args()
    check_arguments(argums)
    return argums




