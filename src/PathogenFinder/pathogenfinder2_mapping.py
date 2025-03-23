import os
import argparse

from .postprocessdata.protein_PF2feature import MapProteins
from .postprocessdata.embedding_PF2feature import MapEmbeddings

DATA_FOLDER = "{}/../../data/".format(Path(__file__).parent.resolve())

def cl_arguments():
    parser = argparse.ArgumentParser(prog='Pathogenfinder2.0 Mapping',
                            description="Arguments for mapping the proteins highlighted by the attention layer, or "
                                     "for mapping the embeddings on the Patogenic Bacterial Landscape.",
                            add_help=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(title="PathogenFinder functionalities", required=True)
    subparser.add_argument("-o", "--out_folder", help="Folder where to output the results")
    map_proteins = subparsers.add_parser("proteins", help="Map the proteins highlighted by the attention layer to a protien database",
                                             parents=[parent_parser])
    map_proteins.set_defaults(action="MapProteins")
    map_proteins.add_argument('--db_path', help='Diamond formatted db path', default="../../data/protein_db/uniref50")
    map_proteins.add_argument('--prot_path', help='Path to protein fasta file', required=True)
    map_proteins.add_argument('--att_path', help='Path to attention npz file', required=True)
    map_proteins.add_argument('--log_folder', help='Folder for the logs of diamond', default=None)
    map_proteins.add_argument('--amount_hits', help="Amount of hits reported", default=3)
    map_proteins.add_argument('--amount_prots', help="Amount of proteins reported", default=20)
    map_proteins.add_argument("--diamond_path", help="Path to Diamond aligner")
    map_proteins.add_argument("--tmp_folder", help="Folder for products of the aligning of Diamond", default=None)

    neighbors_bpl = subparsers.add_parser("map_embeddings", help="Map embeddings produced by PF2 model to the dataset of pathogenic genomic sequences landscape",
                                          parents=[parent_parser])
    neighbors_bpl.set_defaults(action="NeighborsBPL")
    neighbors_bpl.add_argument('--embeddings_bpl', help='Path to the npz file with the embeddings from the PF2 paper', default="../../data/embeddings_BPL/")
    neighbors_bpl.add_argument('--embeddings_pred', help='Path to the npz file with the embeddings predicted', required=True)

    return parser.parse_args()


class PathogenFinder2_Mapping:

    @staticmethod
    def map_proteins(diamond_path:str, folder_out:str, folder_tmp:str, db_path:str, prot_path:str,
                     att_path:str, log_folder:str, amount_hits:int=1, amount_prots:int=20):

        mapprot = MapProteins(folder_out=folder_out, folder_tmp=folder_tmp,
                                diamond_path=diamond_path, db_path=db_path)
        tsv_file, fsa_file = mapprot.read_attentionfile(att_file=att_path,
                                                        prot_file=prot_path)
        diamond_file = mapprot.run_diamond(infile=fsa_file, num_report=amount_hits, log_folder=log_folder)
        mapped_data = mapprot.analyze_results(infile=diamond_file, df_att=pd.read_csv(tsv_file, sep="\t"))
        return mapped_data

    @staticmethod
    def map_embeddings(self, folder_out:str, embeddings_bpl:str, embeddings_pred:str):
        mapemb = MapEmbeddings(out_folder=folder_out, data_embed=embeddings_bpl)
        test_transf = mapemb.fittestdata(testdata=embeddings_pred)
        closer_df, closer_arr = mapemb.knn(test_transf)
        mapemb.make_graph(test_data=test_transf, closer_data=closer_arr)

def main():
    args = cl_arguments()
    logging.basicConfig(level=args.loglevel)
    if args.action == "MapProteins":
        if args.tmp_folder is None:
            args.tmp_folder = args.out_folder
        PathogenFinder2_Mapping.map_proteins(diamond_path=args.diamond_path, folder_out=args.out_folder, folder_tmp=args.tmp_folder,
                                             db_path=args.db_path, prot_path=args.prot_path, att_path=args.att_path, log_folder=args.log_folder,
                                             amount_hits=args.amount_hits, amount_prots=args.amount_prots)
    elif args.action == "NeighborsBPL":
        PathogenFinder2_Mapping.map_embeddings(folder_out=args.out_folder, embeddings_bpl=args.embeddings_bpl, embeddings_pred=args.embeddings_pred)
    else:
        raise ValueError("The action {} is not available yet".format(args.action))

if __name__ == '__main__':
    main()

