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
    parser = argparse.ArgumentParser(
        prog='Pathogenfinder2',
        description=(
            "Command line version of PathogenFinder2. Contains the options to "
            "predict the pathogenic capacity using an already trained model, as well as "
            "mapping the embedding of the genomic sequence to the Pathogenic Bacteria Landscape "
            "and aligning the highlighted proteins by the model to a protein database.\n"
            "It also include options for training your own model, as well as testing an already "
            "trained model."
        ),
        add_help=True,
    )
    parser.add_argument(
        "--version", action="version", version=__version__,
        help="Show program's version number and exit",
    )

    # ---------- COMMON (shared by all subcommands) ----------

    flags_parent = argparse.ArgumentParser(add_help=False)


    # General flags
    flags_parent.add_argument(
        "-v", "--version", action="version", version=__version__,
        help="Show program's version number and exit"
    )
    flags_parent.add_argument(
        "-d", "--debug", help="For debugging",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING
    )
    flags_parent.add_argument(
        "--verbose", help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO
    )

    # General I/O options
    common_parent = argparse.ArgumentParser(add_help=False)
    in_out = common_parent.add_argument_group(
        'General Input/Output options',
        "General Input/Output options that can apply to any mode/functionality"
    )
    in_out.add_argument(
        "-c", "--config",
        help=("Config file (.json) with the configuration to run any of the "
              "PathogenFinder2 functionalities (only recommended for experienced users). "
              "It will overwrite any of the commandline arguments, but provides full control of the model."),
        default=None
    )
    in_out.add_argument(
        "-o", "--outputFolder",
        help="Path to folder where to save the files produced by PathogenFinder2",
        required=True
    )

    # Executable paths
    exec_paths = common_parent.add_argument_group(
        'Executable paths',
        "Paths to executables or tertiary software that might be required by PathogenFinder2. "
        "Only required if the software is not included on the executable PATH."
    )
    exec_paths.add_argument("--prodigalPath", help="Path to Prodigal executable.", default="prodigal")
    exec_paths.add_argument("--diamondPath", help="Path to Diamond executable", default="diamond")

    # ---------- SUBPARSERS ----------
    subparsers = parser.add_subparsers(
        title="PathogenFinder2 functionalities", required=True, dest="subcmd"
    )

    setup_parser = subparsers.add_parser(
        "setup_gsea",
        help="Set up the data for GSEA",
        parents=[flags_parent]
    )
    setup_parser.set_defaults(action="Setup_SwissProt")

    base_setup = setup_parser.add_argument_group(
        'SetUp SwissProt for GSEA Options',
        "Options for setting up SwissProt for GSEA Options"
    )
    base_setup.add_argument("--swissprot_tsv", help="Swiss-Prot TSV metadata file to be formated", default=None)
    base_setup.add_argument("--go_file", help="Go-basic file", default=None)
    base_setup.add_argument("--outputFolder", help="Out folder", required=True)
    base_setup.add_argument("--diamondPath", help="Path to Diamond executable", default="diamond")


    # ----- PREDICT -----
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict bacterial pathogenic capacity with PathogenFinder2",
        parents=[flags_parent, common_parent]
    )
    predict_parser.set_defaults(action="Prediction")

    base_pred = predict_parser.add_argument_group(
        'Pathogen Capacity prediction Options',
        "Options for running the basic functionalities to predict pathogenic capacity of PathogenFinder2"
    )
    base_pred.add_argument(
        "-i", "--inputFile",
        help=("Path to input for predicting the pathogenic capacity. "
              "The format must be described in the -f/--formatSeq argument"),
        default=None
    )
    base_pred.add_argument(
        "--multipleFiles",
        help=("Path to text file with paths to input files. This option allows for the "
              "prediction of bacterial pathogenic capacity of multiple bacterial genomes, "
              "by parallelizing the neural network step."),
        default=None
    )
    base_pred.add_argument(
        "-f", "--formatSeq",
        help="The format of the bacterial sequence(s).",
        choices=["genome", "proteome", "embeddings"],
        required=True
    )
    base_pred.add_argument("--protT5Path",
        help="Path to local protT5 model", default=None)

    base_pred.add_argument(
        "--weightsModel",
        help=("Path to the file(s) with weights used by the deep learning model to predict. "
              "If not selected, the model will use the weights provided by the authors.")
    )
    base_pred.add_argument(
        "--embedProteome",
        help="If used, report or/and map the embeddings to the Bacterial Pathogenic Landscape",
        choices=["report", "map"],  # default is None = disabled
        default=None
    )
    base_pred.add_argument(
        "--attProteins",
        help=("If used, report the attentions or/and align the 20 proteins with highest "
              "attention score to a protein database"),
        choices=["report", "align"],
        default=None
    )
    base_pred.add_argument("--cge", help="Save the predictions in the standard CGE output", action="store_true")

    protaln = predict_parser.add_argument_group(
        'Protein Alignment Options',
        "Options for aligning the proteins of interest highlighted by PathogenFinder2"
    )
    protaln.add_argument(
        "--dbProteins",
        help=("Path to protein database indexed with diamond to align the proteins "
              "highlighted by the attention layer"),
        default=None
    )
    protaln.add_argument(
        "--gsea",
        help="Run GSEA-like analysis on highlighted proteins",  # adjusted help to avoid confusion
        action="store_true",
        default=False
    )
    protaln.add_argument(
        "--dbMetadataProteins",
        help=("Path to protein database metadata previously formated with setup_gsea"),
        default=None
    )
    protaln.add_argument(
        "--minsize_gsea",
        help=("Minimum size for a gene set to be included in gsea"),
        default=15
    )

    # ----- INFER PROTEOME LM -----
    infer_parser = subparsers.add_parser(
        "infer_proteomeLM",
        help="Predict the protein content and create its embeddings with Prodigal and protT5",
        parents=[flags_parent, common_parent]
    )
    infer_parser.set_defaults(action="Infer")
    infer_parser.add_argument(
        "-i", "--inputFile",
        help="Path to genome file to predict its protein content and create embeddings.",
        default=None
    )
    infer_parser.add_argument(
        "--multipleFiles",
        help="Path to text file with paths to input files.",
        default=None
    )
    infer_parser.add_argument(
        "--protT5Path",
        help="Path to local protT5 model",
        default=None,
    )

    # ----- TRAIN (skeleton) -----
    train_parser = subparsers.add_parser(
        "train",
        help="Train the PathogenFinder2 model for bacterial pathogenic capacity using your own data",
        parents=[flags_parent, common_parent]
    )
    train_parser.set_defaults(action="Train")

    # Parse + custom checks
    args = parser.parse_args()
    check_arguments(args)
    return args




