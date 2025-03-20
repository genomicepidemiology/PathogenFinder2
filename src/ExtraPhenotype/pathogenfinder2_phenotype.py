import argparse
import os
from pathlib import Path
from embedding_PF2feature import MapEmbeddings
from proteins_PF2feature import MapProteins


def cl_arguments():
    parser = argparse.ArgumentParser(prog='Pathogenfinder2.0 Model',
                                    description="Arguments for pathogenicity inference, training,"
                                                " testing and hyperparameter selection of the model.",
                                    add_help=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-o", "--outputFolder", help="Folder where to output the results")
    subparsers = parser.add_subparsers(title="PathogenFinder Phenotyping", required=True)

    map_proteins = subparsers.add_parser("map_proteins", help="Map proteins selected by the attention layer", parents=[parent_parser])
    map_proteins.set_defaults(action="MapProteins")
    map_proteins.add_argument("--diamond_path", help='Diamond path', default="../../bin/diamond/")
    map_proteins.add_argument('--db_path', help='Diamond formatted db path', default="../../data/protein_db/uniref50")
    map_proteins.add_argument('--prot_path', help='Path to protein fasta file', required=True)
    map_proteins.add_argument('--att_path', help='Path to attention npz file', required=True)
    map_proteins.add_argument('--log_folder', help='Folder for the logs of diamond', required=True)
    map_proteins.add_argument('--amount_hits', help="Amount of hits reported", default=3)
    map_proteins.add_argument('--amount_prots', help="Amount of proteins reported", default=20)

    neighbors_bpl = subparsers.add_parser("map_embeddings", help="Map embeddings produced by PF2 model to the dataset of pathogenic genomic sequences landscape",
                                          parents=[parent_parser])
    neighbors_bpl.set_defaults(action="NeigborsBPL")
    neighbors_bpl.add_argument('--embeddings_bpl', help='Path to the npz file with the embeddings from the PF2 paper', default="../../data/embeddings_BPL/")
    neighbors_bpl.add_argument('--embeddings_pred', help='Path to the npz file with the embeddings predicted', required=True)

    return parser.parse_args()


class PathogenFinder2_Phenotype:

    def __init__(self, out_folder:str):
        self.out_folder = out_folder

    def map_proteins(self, diamond_path:str, db_path:str, prot_path:str, att_path:str, log_folder:str,
                     amount_hits:int, amount_prots:int):

        mapprot = MapProteins(folder=args.out_folder,
                          diamond_path=args.diamond_path,
                          db_path=args.db_path)
        tsv_file, fsa_file = mapprot.read_attentionfile(att_file=args.att_path,
                                    prot_file=args.prot_path,
                                    num_prot=args.amount_prots)
        diamond_file = mapprot.run_diamond(infile=fsa_file, num_report=args.amount_hits)
        mapprot.analyze_results(infile=diamond_file, df_att=pd.read_csv(tsv_file, sep="\t"), amount_hits=args.amount_hits)

    def map_embeddings(self, embeddings_bpl:str, embeddings_pred:str):
        mapemb = MapEmbeddings(out_folder=args.out_folder, data_embed=args.embedding_train)
        test_transf = mapemb.fittestdata(testdata=args.embedding_test)
        closer_df, closer_arr = mapemb.knn(test_transf)
        closer_df.to_csv("{}/closeneighbors_metadata.tsv".format(args.out_folder), sep="\t", index=False)
        mapemb.make_graph(test_data=test_transf, closer_data=closer_arr)

def main():
    args = cl_arguments()
    pathogenfinder_pheno = PathogenFinder2_Phenotype(out_folder=args.outputFolder)
    if args.action == "NeigborsBPL":
        pathogenfinder_pheno.map_embeddings(embeddings_bpl=args.embeddings_bpl, embeddigns_pred=args.embeddings_pred)
    elif args.action == "MapProteins":
        pathogenfinder_pheno.map_proteins(diamond_path=args.diamond_path, db_path=args.db_path, prot_path=args.prot_path, att_path:args.att_path,
                                          log_folder=args.log_folder, amount_hits=args.amount_hits, amount_prots=args.amount_prots)
    else:
        raise ValueError("The action {} is not available".format(args.action))


if __name__ == "__main__":
    main()
