# Installation
PathogenFinder2 consist of a main package (pathogenFinder2) that can be installed with Pip (recommended) or with Docker (comming).

If you are installing through Pip, the dependencies of external software and optional databases must be installed as described below. If you are installing thhrough Docker, the dependencies and databases will be installed through it.

## Installing through Pip
**Important**: This will install PathogFinder2 in the environment where you run pip and potenitally update the python modules ResFinder depends on. It is recommended to run PathogenFinder2 in its own environment, in order to avoid breaking existing installations and prevent ResFinder from getting broken by future unrelated pip installations. This is described in the optional step below.

### Optional: Create virtual environment

Go to the location where you want to store your environment.
```unix
# Create environment
python3 -m venv pathogenfinder2_env

# Activate environment
source pathogenfinder2_env/bin/activate

# When you are finished using ResFinder deactivate the environment
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

### Installing through Docker (Beta)

The PathogenFinder2 application has been build into a single image on docker hub named "genomicepidemiology/pathogenfinder2". Below is an example run, where the current working directory is bound to the container "/app" path which is the container working directory.