# Test PathogenFinder2
In this section, we described how to run inference with PathogenFinder2 with a test file, as well as desccribing the output provided by the software.
## Run Inference PathogenFinder2
We provide with a couple of test files to observe how PathogenFinder2 inference behaves. We encourage you to run it using GPUs, as the steps of producing the embeddings and going through the neural network can be quite expensive on a CPU.

To run PathogenFinder2 with the simple mode (only pathogenic prediction):
```unix
pathogenfinder2 predict -i test/data/GCF_000014385.1_ASM1438v1_genomic.fna -f genome -o /path/to/outputfolder --prodigalPath path/to/prodigal
```
*Expecting you are located in the main folder of the repository.*

If you want the complete mode (with Pathogenic Bacterial Landscape and Proteins of Interest):
```unix
pathogenfinder2 predict -i test/data/GCF_000014385.1_ASM1438v1_genomic.fna -f genome -o /path/to/outputfolder --prodigalPath /path/to/prodigal --embedProteome map --attProteins align --dbProteins /path/to/uniref50 --diamondPath /path/to/diamond
```
*Expecting you are located in the main folder of the repository.*


## Results files:
The output path indicated with *-o* will contain the folder **predictionPF2/** the prediction and other products produced during the inference. The folders/products of running PathogenFinder2 with simple mode:
* **data_input.tsv**: resume of the input files.
* **preprocess_PF2**: will contain files produced during protein prediction (Prodigal) and protein embedding (protT5).
* **log_PF2**: Will contain the stdout and stderr of programs executed during the run of PathogenFinder2 (as for Prodigal or Diamond)
* **results_PF2**: will contain the result files from PathogenFinder2:
    * *predictions.tsv*: the predictions of the PathogenFinder2 model on pathogenic capacity. There is a detailed description on the header of the tsv file.
    * *cge_out.json*: results in the cge format. Only produced if --cge is used.

If PathogenFinder2 is run with the complete mode (BPL and Proteins of interest) the next files will also be produced.
* **postprocess_PF2**: will contain files produced during alignment (Diamond).
*  **results_PF2**:
    * *embeddings.npz*: the embeddings defining the genomic sequence. Only produced if --embedProteome is used.
    * *attentions.npz*: the attention score values per protein. Only produced if --attProteins is used.
    * *closeneighbors_bpl.tsv*: the metadata of the 10 closer pathogens to the sequence analyzed. Only produced if --embedProteome is used with "map".
    * *mapped_bpl.png*: the Bacterial Pathogenic Landscape with the location of the sequence analyzed. Only produced if --embedProteome is used with "map".
    * *mapped_proteins.tsv*: hits of the top proteins highlighted by the attentions score on the UniRef50 database. Only produced if --attProteins is used with "map".
