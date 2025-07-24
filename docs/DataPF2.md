# PathogenFinder2 Data

The data used for training and testing PathogenFinder2 is contained in this repository in [link](../data/).
This folder contains *configs* (templates for running PathogenFinder2) and *PathogenFinder2_dataset* (metadata used for trianing and testing PathogenFinder2)

## Configs
This folder contains the files "config_empty.json" and "config_train.json", that can be used as templates for running PathogenFinder2 with the --config option.

## PathogenFinder2_dataset
This folder contains the metadata used for building the dataset of PathogenFinder2, which was used for training the model and evaluating it. We provide with a tsv file where each entry is one of the bacterial genomes used for training/testing, and each column provides information about the genome, including the accession number to download the sequence. 

The metadata files were created as described in the article. Furthermore, each sample (row) contains information from the phenotypic database (NCBI, ENA, Patric...) and RefSeq database.
### METADATA_PF2DB
*Created: November 2023*

TSV [file](../data/PathogenFinder2_dataset/METADATA_PF2DB.tsv) with the entries used for training and testing the model. 

**Columns**:

* **Species_DBs**: Name of the bacterial species on the phenotypic database
* **Strain_DBs**: Name of the bacterial strain on the phenotypic database
* **Origin DB**: Name of the phenotypic database(s) the entry was found.
* **Taxonomy ID**: Taxonomy ID on the phenotypic database
* **Species abb.**: Species abbreviation with the phenotypic database.
* **Assembly Accession Genbank**: Accession in GenBank
* **FTP path genbank**: FTP link to the Genbank entry
* **Assembly Accession RefSeq**: Accession in RefSeq
* **FTP path refseq**: FTP link to the RefSeq entry
* **TaxID refseq**: Taxonomy ID in RefSeq
* **Organism Name Refseq**: Organism name (Species) in RefSeq
* **Infraspecific Name Refseq**: Strain name in RefSeq
* **version_status_refseq**: Version status of RefSeq entry
* **assembly_level_refseq**: Assembly level on RefSeq
* **seq_rel_date_refseq**: Release date of RefSeq
* **Partition**: Partition number for PathogenFinder2. 1-4 were used for training, 5 for testing. The partitions were produced with SpanSeq.
* **Protein Count**: Amount of proteins predicted with Prodigal. Numer necessary for training with Bucketing.
* **PathoPhenotype**: Annotated pathogenic phenotype.

### METADATA_2024strain
*Created: December 2024*

TSV [file](../data/PathogenFinder2_dataset/METADATA_2024strain.tsv) with the entries used for testing the model. It has the same format as "METADATA_PF2DB.tsv" Columns **Partition** and **Protein Count** are not included, as the first was only necessary to separate the test and training set (this dataset was only used for testing); while **Protein Count** was only necessary for training.