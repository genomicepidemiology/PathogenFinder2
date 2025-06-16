# DATA for PathogenFinder2
This folder contains the data used on the paper "Ferrer Florensa, Alfred, et al." and for running PathogenFinder with all funtionalities.

## Configs
This folder contains the files "config_empty.json" and "config_train.json", that can be used as templates for running PathogenFinder2 with the --config option.

## PF2_data
This folder contains the metadata used for building the dataset of PathogenFinder2, which was used for training the model and evaluating it. As all the sequences are from public repositories, we provide with the address and phenotype (plus extra information).
### METADATA_PF2DB
TSV file with the entries used for training and testing the model. The path to the sequences is in the column "ftp_path", and the pathogenic capacity on the column "PathoPhenotype". Information about the origin of the metadata and reason for the phenotype are included. The column "cluster partition" indicate which partition each entry is part of (one of them used for testing while the rest for training).
### test2024strain
TSV file with the entries used for testing the model. It has the same format as "METADATA_PF2DB.tsv"