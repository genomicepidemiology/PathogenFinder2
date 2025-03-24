# PathogenFinder2
Prediction of bacterial pathogenic capacity on humans with protein Language Models.

This repository contains the code to run PathogenFinder2 through the commandline. If prefered, the program can also be runned (only for prediction) through the its [webpage](https://cge.food.dtu.dk/services/PathogenFinder2/). PathogenFinder2 consists of a main package (pathogenfinder2) that predicts the pathogenic capacity of a bacterial genome. The prediction is made through four steps:
1. Protein prediction: PathogenFinder2 uses Prodigal to predict the protein content of the bacterial genome
2. Protein Embedding production: PathogenFinder2 uses ProtT5 to embed each protein into a vector.
3. Concat embeddings: Each vector is stacked in the same order as the proteins appear in the Prodigal prediction.
4. Deep Neural Model: The embeddings of the proteins are used as input for a deep neural network with convolutional layers and an attention layer.

The deep neural model is an ensemble of 4 neural networks that have beent trained with different splits of our data. Due to this, the prediction of the model is the mean of those 4 predictions. The model can also output the attention scores to look into which proteins have mattered the most for each prediction. Moreover, the model also reports the last layer of the neural network before the classification layer, as it can work as a sequence embedder based on its pathogenic capacity.

Besides, PathogenFInder2 has an extra package (pathogenfinder2_mapping) that is able to map those embeddings to the Bacterial Pathogenic Landscape, as well as the top proteins highlighted by the attention layer to a protein database (UniRef50).
## Installation
PathogenFinder2 consist of a main package (pathogenFinder2), and a secondary package (pathogenfinder2_mapping). It can be installed with Pip (recommended) or with Docker (beta).

If you are installing through Pip, the dependencies of external software and optional databases must be installed as described below. If you are installing thhrough Docker, the dependencies and databases will be installed through it.

### Installing external dependencies (for Pip installation)
PathogenFinder2 requires Prodigal to be installed. If it is intended to map the highlighted proteins to a protein database (pathogenfinder2_mapping), Diamond is required, as well as the protein database (UniRef50 is the database used in the PathogenFinder2 article).

#### Prodigal
In order to install Prodigal, you can follow the instructions described in its [repository](https://github.com/hyattpd/Prodigal). Here is the example for Generic Unix, but we recommend you read their short installation instructions, to fit your software better to your system.
```unix
git clone https://github.com/hyattpd/Prodigal
cd Prodigal
make
```
This will produce an executable that you can save wherever you prefer, and indicate to pathogenfinder2 through the commandline where it is located.

#### Diamond (optional)
If you want to map the highlighted proteins to a protein database, diamond needs to be installed. Notice that this will install the version 2.1.11, which is the same version that was used in the Paper.
```unix
wget http://github.com/bbuchfink/diamond/releases/download/v2.1.11/diamond-linux64.tar.gz
tar xzf diamond-linux64.tar.gz
```

#### Protein Database (UniRef50) (Optional)
The recommended protein database for aligning the highlighted proteins to is UniRef50. To download the database and format it for Diamond, follow the next steps:
```unix
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
gunzip uniref50.fasta.gz
/path/to/diamond makedb --in uniref50.fasta -d uniref50
```
The database should be placed on the _/data/protein_db_ folder of this repository.
### Installing through Pip
**Important**: This will install PathogFinder2 in the environment where you run pip and potenitally update the python modules ResFinder depends on. It is recommended to run PathogenFinder2 in its own environment, in order to avoid breaking existing installations and prevent ResFinder from getting broken by future unrelated pip installations. This is described in the optional step below.

#### Optional: Create virtual environment

Go to the location where you want to store your environment.
```unix
# Create environment
python3 -m venv pathogenfinder2_env

# Activate environment
source pathogenfinder2_env/bin/activate

# When you are finished using ResFinder deactivate the environment
deactivate
```
#### Install PathogenFinder2
In order to install the basic functionalities of PathogenFinder2, use:
```unix
pip install .
```

If you want to use also the mapping functionalities, use:
```unix
pip install .[mapping]
```

### Installing through Docker (Beta)

The PathogenFinder2 application has been build into a single image on docker hub named "genomicepidemiology/pathogenfinder2". Below is an example run, where the current working directory is bound to the container "/app" path which is the container working directory.

## Test data

Test data can be found in the sub-directory test/data

## Usage
**Important**: Any of the modes of the main module will improve its speed notably if used on a computer with GPU available. In particular, the steps infering the embeddings of each protein (using protT5) and the neural network to predict pathogenic capacity. The step for predicting the protein content, as well as mapping the embeddings or the proteins to a database (mapping submodule) will always run on CPUs.


### Main module

The main module of PathogenFinder2 can be used in 4 modes:
* **Inference**: Predicts pathogenic capacity using pre-trained weights for the neural network. The model can predict several inputs at the same time, although notice that the steps of protein prediction and embedding are not parallelized for several sequences. The input can be a genome in fasta file, a collection of proteins in fasta file, or an embeddings file in HDF5 format. 
* **Test**: Predicts pathogenic capacity using pre-trained weights for the neural network, and tests the accuracy of the model against the labels provided for the data. As with *Inference*, several inputs can be used with different type of inputs.
* **Train**: Trains the neural network, with a training and validation dataset. Can only be done using embedding files (HDF5 format).
* **Hyperparameter Optimization** (not available at the moment): Uses Optuna to calculate the best hyperparameter set.

In order to control the behavior of the software in each of the 4 modes, a json file can be used as input (as the one in *data/configs/config_empty.json*), or different arguments of the command line (only for *inference*).
```unix
pathogenfinder2 -h

usage: Pathogenfinder2.0 Model [-h] {train,test,inference,hyperparam_opt} ...

Arguments for pathogenicity inference, training, testing and hyperparameter selection of the model.

options:
  -h, --help            show this help message and exit

PathogenFinder functionalities:
  {train,test,inference,hyperparam_opt}
    train               Train the PathogenFinder2 model based on certain data and configuration.
    test                Test the PathogenFinder2 model.
    inference           Predict using PathogenFinder2 model.
    hyperparam_opt      Hyperparam optimizaiton PathogenFinder2 model based on certain data and configuration.
```
All the modes create a new folder with the results of the action. The contents of the folder will vary depending on the action. The name of the folder will be the value of the argument *--outputFolder* + a string with the date and time of execution.

#### Inference
```unix
pathogenfinder2 inference -h

usage: Pathogenfinder2.0 Model inference [-h] [-c CONFIG] [-o OUTPUTFOLDER] [--cge] [-d] [-v] [--prodigalPath PRODIGALPATH] [--protT5Path PROTT5PATH] [--diamondPath DIAMONDPATH]
                                         [-i INPUTDATA] [-w WEIGHTSMODEL] [-f {genome,proteome,embeddings}] [--multiFiles] [--embeddings {False,report,map}]
                                         [--attentions {False,report,map}]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Json file with the configuration for PathogenFinder2 model.
  -o OUTPUTFOLDER, --outputFolder OUTPUTFOLDER
                        Folder where to output the results
  --cge                 Output the cge format output
  -d, --debug           For debugging
  -v, --verbose         Be verbose
  -i INPUTDATA, --inputData INPUTDATA
                        Input data for inference
  -w WEIGHTSMODEL, --weightsModel WEIGHTSMODEL
                        Weights used by the deep learning model to predict
  -f {genome,proteome,embeddings}, --formatSeq {genome,proteome,embeddings}
                        The format of the input data.
  --multiFiles          If the input data are multiple files
  --embeddings {False,report,map}
                        If used, report or/and map the embeddings to the Bacterial Pathogenic Landscape
  --attentions {False,report,map}
                        If used, report the attentions or/and map the 20 proteins with highest attention score to UniRef50

Paths to executables:
  Paths to executables. They might not be necessary.

  --prodigalPath PRODIGALPATH
                        Path to Prodigal
  --protT5Path PROTT5PATH
                        Path to protT5
  --diamondPath DIAMONDPATH
                        Path to Diamond
```
If the options **--embeddings** or **--attentions** are used with *map*, the submodule *mapping* from PathogenFinder2 will be used. Please see below more details of it. 

##### Results files:
The output folder will contain a folder with the configuration used for running PathogenFinder2, and a folder for each of the samples inferred. 
* *config* folder: Contains the configurations and inputs used by PathogenFinder2
* *Name of the sample* folder: Contains the results of from PathogenFinder2:
    * *preprocess*: will contain files produced during protein prediction and protein embedding.
    * *postprocess*: (Only when the embeddings and/or the attentions are being mapped) will contain files produced during the mapping of proteins of interests to the protein database.
    * *log*: Will contain the stdout and stderr of programs executed during the run of PathogenFinder2 (as for Prodigal or Diamond)
    * *out*: will contain the result files from PathogenFinder2:
        * *predictions.tsv*: the predictions of the PathogenFinder2 model on pathogenic capacity.
        * *embeddings.npz*: the embeddings defining the genomic sequence. Only produced if --embeddings is used.
        * *attentions.npz*: the attention score values per protein. Only produced if --attentions is used.
        * *closeneighbors_metadata.tsv*: the metadata of the 10 closer pathogens to the sequence analyzed. Only produced if --embeddings is used with "map".
        * *mapped_bpl.png*: the Bacterial Pathogenic Landscape with the location of the sequence analyzed. Only produced if --embeddings is used with "map".
        * *mapped_proteins.tsv*: hits of the top proteins highlighted by the attentions score on the UniRef50 database. Only produced if --attentions is used with "map".
        * *cge_out.json*: results in the cge format. Only produced if --cge is used.

##### Example
```unix
pathogenfinder2 inference -i test/data/GCF_000014385.1_ASM1438v1_genomic.fna.gz -f genome -o test/out --prodigalPath path/to/prodigal --embeddings report --attentions report
```

#### Test
```unix
pathogenfinder2 predict -h

usage: Pathogenfinder2.0 Model test [-h] [-c CONFIG] [-o OUTPUTFOLDER] [--cge] [-d] [-v] [--prodigalPath PRODIGALPATH] [--protT5Path PROTT5PATH] [--diamondPath DIAMONDPATH]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Json file with the configuration for PathogenFinder2 model.
  -o OUTPUTFOLDER, --outputFolder OUTPUTFOLDER
                        Folder where to output the results
  --cge                 Output the cge format output
  -d, --debug           For debugging
  -v, --verbose         Be verbose

Paths to executables:
  Paths to executables. They might not be necessary.

  --prodigalPath PRODIGALPATH
                        Path to Prodigal
  --protT5Path PROTT5PATH
                        Path to protT5
  --diamondPath DIAMONDPATH
                        Path to Diamond
```

#### Train 
```unix
pathogenfinder2 train -h

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Json file with the configuration for PathogenFinder2 model.
  -o OUTPUTFOLDER, --outputFolder OUTPUTFOLDER
                        Folder where to output the results
  --cge                 Output the cge format output
  -d, --debug           For debugging
  -v, --verbose         Be verbose

Paths to executables:
  Paths to executables. They might not be necessary.

  --prodigalPath PRODIGALPATH
                        Path to Prodigal
  --protT5Path PROTT5PATH
                        Path to protT5
  --diamondPath DIAMONDPATH
                        Path to Diamond

```

#### Hyperparameter optimization (not available at the moment)
```unix
pathogenfinder2 hyperparam_opt -h

usage: Pathogenfinder2.0 Model hyperparam_opt [-h] [-c CONFIG] [-o OUTPUTFOLDER] [--cge] [-d] [-v] [--prodigalPath PRODIGALPATH] [--protT5Path PROTT5PATH] [--diamondPath DIAMONDPATH]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Json file with the configuration for PathogenFinder2 model.
  -o OUTPUTFOLDER, --outputFolder OUTPUTFOLDER
                        Folder where to output the results
  --cge                 Output the cge format output
  -d, --debug           For debugging
  -v, --verbose         Be verbose

Paths to executables:
  Paths to executables. They might not be necessary.

  --prodigalPath PRODIGALPATH
                        Path to Prodigal
  --protT5Path PROTT5PATH
                        Path to protT5
  --diamondPath DIAMONDPATH
                        Path to Diamond
```

### Mapping submodule
The mapping submodule has functionalities that can complement the pathogenic capacity prediction. It includes:
1. Mapping the embeddings produced by the neural network against the dataset of pathogens from PathogenFinder2
2. Mapping the top proteins highlighted by the attention layer to a database of proteins (UniRef50)

Notice that it is necessary to have had installed the PathogenFinder2 module with *[mapping]*.
```unix
pathogenfinder2_mapping -h

usage: Pathogenfinder2 Mapping [-h] {proteins,embeddings} ...

Arguments for mapping the proteins highlighted by the attention layer, or for mapping the embeddings on the Patogenic Bacterial Landscape.

options:
  -h, --help            show this help message and exit

PathogenFinder2 Mapping functionalities:
  {proteins,map_embeddings}
    proteins            Map the proteins highlighted by the attention layer to a protien database
    embeddings      Map embeddings produced by PF2 model to the dataset of pathogenic genomic sequences landscape

```
**Important**: If mapping the embeddings, please unzip the file *data/embeddings_BPL/embeddings.npz.gz* containing the information of the Bacterial Pathogenic Landscape.

#### Mapping Embeddings

```unix
pathogenfinder2_mapping embeddings -h

usage: Pathogenfinder2 Mapping embeddings [-h] [-o OUT_FOLDER] [--embeddings_bpl EMBEDDINGS_BPL] --embeddings_pred EMBEDDINGS_PRED

options:
  -h, --help            show this help message and exit
  -o OUT_FOLDER, --out_folder OUT_FOLDER
                        Folder where to output the results
  --embeddings_bpl EMBEDDINGS_BPL
                        Path to the npz file with the embeddings from the PF2 paper
  --embeddings_pred EMBEDDINGS_PRED
                        Path to the npz file with the embeddings predicted

```

#### Mapping Proteins

```unix
pathogenfinder2_mapping proteins -h

usage: Pathogenfinder2 Mapping proteins [-h] [-o OUT_FOLDER] [--db_path DB_PATH] --prot_path PROT_PATH --att_path ATT_PATH [--log_folder LOG_FOLDER] [--amount_hits AMOUNT_HITS]
                                        [--amount_prots AMOUNT_PROTS] [--diamond_path DIAMOND_PATH] [--tmp_folder TMP_FOLDER]

options:
  -h, --help            show this help message and exit
  -o OUT_FOLDER, --out_folder OUT_FOLDER
                        Folder where to output the results
  --db_path DB_PATH     Diamond formatted db path
  --prot_path PROT_PATH
                        Path to protein fasta file
  --att_path ATT_PATH   Path to attention npz file
  --log_folder LOG_FOLDER
                        Folder for the logs of diamond
  --amount_hits AMOUNT_HITS
                        Amount of hits reported
  --amount_prots AMOUNT_PROTS
                        Amount of proteins reported
  --diamond_path DIAMOND_PATH
                        Path to Diamond aligner
  --tmp_folder TMP_FOLDER
                        Folder for products of the aligning of Diamond
```

## PathogenFinder2 Dataset
The PathogenFinder2 Dataset can be accessed in its [repository](https://github.com/genomicepidemiology/PathogenFinder2_DB), or by recursively cloning this repository:
```unix
git clone --recursive https://github.com/genomicepidemiology/PathogenFinder2
```

## Citation
When using the method please cite:

## References
1. Hyatt, Doug, et al. "Prodigal: prokaryotic gene recognition and translation initiation site identification." BMC bioinformatics 11 (2010): 1-11.
2. Buchfink, Benjamin, Klaus Reuter, and Hajk-Georg Drost. "Sensitive protein alignments at tree-of-life scale using DIAMOND." Nature methods 18.4 (2021): 366-368.
3. Suzek, Baris E., et al. "UniRef clusters: a comprehensive and scalable alternative for improving sequence similarity searches." Bioinformatics 31.6 (2015): 926-932.

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
