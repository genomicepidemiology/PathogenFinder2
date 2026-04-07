# Installation
PathogenFinder2 consist of a main package (PathogenFinder2) that can be installed with Pip (recommended) or with Docker (comming).

If you are installing through Pip, the dependencies of external software and optional databases must be installed as described below. If you are installing through Docker, the dependencies and databases will be installed through it.

## Installing through Pip
**Important**: This will install PathogFinder2 in the environment where you run pip and potentially update the python modules PathogenFinder2 depends on. It is recommended to run PathogenFinder2 in its own environment, in order to avoid breaking existing installations and prevent PathogenFinder2 from getting broken by future unrelated pip installations. This is described in the optional step below.

### Optional: Create virtual environment

Go to the location where you want to store your environment.
```unix
# Create environment. PathogenFinder2 requires a Python version equal or newer than 3.10
python3 -m venv pathogenfinder2_env

# Activate environment
source pathogenfinder2_env/bin/activate

# When you are finished using PathogenFinder2 deactivate the environment
deactivate
```
### Install PathogenFinder2
In order to install the basic functionalities of PathogenFinder2, use:
```unix
git clone https://github.com/genomicepidemiology/PathogenFinder2.git
cd PathogenFinder2
pip install .
```

If you want to use also the mapping functionalities, use:
```unix
pip install .[mapping]
```

### Installing external dependencies (for Pip installation)
PathogenFinder2 requires Prodigal to be installed. If it is intended to map the highlighted proteins to a protein database (pathogenfinder2_mapping), Diamond is required, as well as the protein database (UniRef50 is the database used in the PathogenFinder2 article).

#### Prodigal
In order to install Prodigal, you can follow the instructions described in its [repository](https://github.com/hyattpd/Prodigal). Here is the example for Generic Unix, but we recommend you read their short installation instructions, to fit your software better to your system.
```unix
git clone https://github.com/hyattpd/Prodigal
cd Prodigal/
make
cd ..
```
This will produce an executable that you can save wherever you prefer, and indicate to pathogenfinder2 through the commandline where it is located.

#### Diamond (optional)
If you want to map the highlighted proteins to a protein database, diamond needs to be installed. Notice that this will install the version 2.1.11, which is the same version that was used in the Paper.
```unix
wget http://github.com/bbuchfink/diamond/releases/download/v2.1.11/diamond-linux64.tar.gz
tar xzf diamond-linux64.tar.gz
```

#### Protein Database (Optional)
The protein databases available for default to align the highlighted proteins by the attention module are UniRef50 and Swiss-Prot. While the first one offers more coverage due to its size, the second one will provide better metadata and the possibility to perform GSEA, being also less intense computationally.
##### UniRef50
To download the database and format it for Diamond, follow the next steps. Please notice that "/path/to" means the path where that file is located, as PathogenFinder2 is flexible enough to allow you to place the file where is necessary:
```unix
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
gunzip uniref50.fasta.gz
/path/to/diamond makedb --in /path/to/uniref50.fasta -d /path/to/uniref50
```
#### Swiss-Prot
As downloading the database (and formatting it for performing GSEA) is slightly more difficult, we have created a functionality on pathogenfinder2 to perform those steps for you.
* **setup_gsea**: Can download the swiss-prot (bacteria proteins) dataset, index it with Diamond, and format it for GSEA. If the user only wants to format the already downloaded dataset, just point to the tsv file (protein dataset metadata) with the --swissprot_tsv option.

```unix
pathogenfinder2 setup_gsea -h
usage: Pathogenfinder2 setup_gsea [-h] [-v] [-d] [--verbose] [--swissprot_tsv SWISSPROT_TSV] [--go_file GO_FILE] --outputFolder OUTPUTFOLDER

options:
  -h, --help            show this help message and exit
  -v, --version         Show program's version number and exit
  -d, --debug           For debugging
  --verbose             Be verbose

SetUp SwissProt for GSEA Options:
  Options for setting up SwissProt for GSEA Options

  --swissprot_tsv SWISSPROT_TSV
                        Swiss-Prot TSV metadata file to be formated
  --go_file GO_FILE     Go-basic file
  --outputFolder OUTPUTFOLDER
                        Out folder
```
Example of usage:
```unix
pathogenfinder2 setup_gsea --outputFolder /path/to/folder
```
