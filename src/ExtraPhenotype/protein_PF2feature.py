import pandas as pd
import numpy as np
import subprocess
import os


class MapProteins:

    def __init__(self, folder:str, db_path:str, diamond_path:str):
        self.folder = folder
        self.db_path = db_path
        self.diamond_path = diamond_path

    @staticmethod
    def read_protfasta(file_path:str) -> dict:
        dict_prots = {}
        with open(file_path, "r") as file_read:
            for line in file_read:
                if line.startswith(">"):
                    id_prot = line.rstrip()[1:].split()[0]
                    dict_prots[id_prot] = ""
                else:
                    dict_prots[id_prot] += line.rstrip()
        return dict_prots

    @staticmethod
    def write_protfasta(file_path:str, dict_prot:dict) -> None:
        with open(file_path, "w") as file_write:
            for protid, protseq in dict_prot.items():
                file_write.write(">{}\n".format(protid))
                file_write.write("{}\n".format(protseq))

    def read_attentionfile(self, att_file:str, prot_file:str, num_prot:int) -> (str,str):
        att_data = np.load("{}".format(att_file), allow_pickle=True)
        att = att_data["attentions"]
        protids = att_data["protIDs"]
        max_ind = (-att).argsort(axis=-1)[:, :num_prot]
        prot_seq = MapProteins.read_protfasta(file_path=prot_file)
        prots_pd = list()
        nn_pd = list()
        att_pd = list()
        selected_prots = dict()
        for nn in [0, 1, 2, 3]:
            max_atts = att[nn, max_ind[nn,:]]
            max_protids = protids[max_ind[nn,:]]
            prots_pd.extend(max_protids.astype(str).tolist())
            att_pd.extend(max_atts.tolist())
            nn_pd.extend([nn]*num_prot)
            for protid in max_protids:
                protid_utf8 = protid.decode("utf-8").split()[0]
                selected_prots[protid_utf8]=prot_seq[protid_utf8]
        df = pd.DataFrame({"ProtNames": prots_pd, "NN": nn_pd, "Attention Value": att_pd})
        df.to_csv("{}/selected_prots.tsv".format(self.folder), sep="\t", index=False)

        MapProteins.write_protfasta(file_path="{}/selected_prots.fsa".format(self.folder),
                                    dict_prot=selected_prots)
        return "{}/selected_prots.tsv".format(self.folder), "{}/selected_prots.fsa".format(self.folder)

    def run_diamond(self, infile:str, num_report:int=1) -> str:
        command = "{diamond_path} blastp -q {query_path} -d {db_path} -o {out_path}/out_diamond.tsv".format(
                                            diamond_path=self.diamond_path, query_path=infile, db_path=self.db_path, out_path=self.folder)
        command_opt = " --faster --max-target-seqs {}".format(num_report)
        command_out = " --outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qtitle stitle scovhsp slen"
        command += command_opt
        command += command_out
        diamond_proc = subprocess.run(command.split(), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True)
        stderr_file = "{}/diamond.stderr".format(self.folder)
        with open(stderr_file, "w") as createdberr:
            createdberr.write(diamond_proc.stderr)
        return "{out_path}/out_diamond.tsv".format(out_path=self.folder)

    def analyze_results(self, infile:str, df_att:pd.DataFrame, amount_hits:int=1):
        data_diamond = pd.read_csv(infile, sep="\t", names=["qseqid","sseqid","pident","length","mismatch","gapopen","qstart","qend",
                                                                "sstart","send","evalue","bitscore","qtitle","stitle", "scovhsp", "slen"])
        print(data_diamond)
        print(df_att)
        df_att["ProtIDs"] = df_att["ProtNames"].str.split().str[0]
        list_prots = np.unique(data_diamond["qseqid"]).tolist()
        data_list = []
        for nn in [0, 1, 2, 3]:
            att_prots = df_att[df_att["NN"]==nn]
            print(att_prots)
            for num in range(len(att_prots)):
                prot_df = self.analyze_nn_results(att_prots.iloc[num]["ProtIDs"], data_diamond=data_diamond,
                                        amount_hits=amount_hits)
                prot_df["ProtNames"] = att_prots.iloc[num]["ProtNames"]
                prot_df["NN"] = att_prots.iloc[num]["NN"]
                prot_df["Attention Value"] = att_prots.iloc[num]["Attention Value"].astype("float")
                prot_df["ProtIDs"] = att_prots.iloc[num]["ProtIDs"]
                data_list.append(prot_df)
        data_df = pd.concat(data_list)
        data_df = data_df.set_index(["ProtIDs", "ProtNames", "NN", "Attention Value"])
        data_df.to_csv("{}/mapped_proteins.tsv".format(self.folder), sep="\t")


    def analyze_nn_results(self, protID:str, data_diamond:pd.DataFrame, amount_hits:int) -> pd.DataFrame:
        prot_df = data_diamond[data_diamond["qseqid"]==protID].sort_values(by=['pident'], ascending=False).head(amount_hits)
        if len(prot_df) == 0:
            (ref_id, ref_name, identity, alignment_length, ref_gene_length, coverage,
                ref_startpos, ref_endpos, query_id, query_start_pos, query_end_pos, taxname, taxid) = [["-"]]*13
        else:
            ref_id = prot_df["sseqid"]
            ref_name = prot_df["stitle"]
            identity = prot_df["pident"]
            alignment_length = prot_df["length"]
            ref_gene_length = prot_df["slen"]
            coverage = prot_df["scovhsp"]
            ref_startpos = prot_df["sstart"]
            ref_endpos = prot_df["send"]
            query_id = prot_df["qseqid"]
            query_startpos = prot_df["qstart"]
            query_endpos = prot_df["qend"]
            taxname = prot_df["stitle"].str.split("Tax").str[1].str.replace("=","")
            taxid = prot_df["stitle"].str.split(" ").str[-2].str.replace("TaxID=","")
        pheno_df = pd.DataFrame({"Ref_ID":ref_id, "Ref_name":ref_name, "Identity":identity, "Alignment_Length": alignment_length,
                                 "Ref_Length": ref_gene_length, "Ref_coverage": coverage, "Ref_start_pos": ref_startpos,
                                 "Ref_end_pos": ref_endpos, "Query_ID": query_id, "Query_start_pos": query_startpos,
                                 "Query_end_pos": query_endpos, "TaxName": taxname, "TaxID": taxid})
        return pheno_df

def get_args():
    parser = argparse.ArgumentParser(description='Mapping your sequence to the Bacterial Pathogenic Landscape')
    parser.add_argument('--diamond_path', help='Diamond path')
    parser.add_argument('--db_path', help='Diamond formatted db path', required=True)
    parser.add_argument('--prot_path', help='Path to protein fasta file')
    parser.add_argument('--att_path', help='Path to attention npz file')
    parser.add_argument('--log_folder', help='Folder for the logs of diamond')
    parser.add_argument('--amount_hits', help="Amount of hits reported", default=3)
    parser.add_argument('--amount_prots', help="Amount of proteins reported", default=20)
    parser.add_argument("--out_folder", help='Folder where to output results', required=True)
    return parser.parse_args()

def main():
    args = get_args()
    mapprot = MapProteins(folder=args.out_folder,
                          diamond_path=args.diamond_path,
                          db_path=args.db_path)
    tsv_file, fsa_file = mapprot.read_attentionfile(att_file=args.att_path,
                                    prot_file=args.prot_path,
                                    num_prot=args.amount_prots)
    diamond_file = mapprot.run_diamond(infile=fsa_file, num_report=args.amount_hits)
    mapprot.analyze_results(infile=diamond_file, df_att=pd.read_csv(tsv_file, sep="\t"), amount_hits=args.amount_hits)


if __name__ == "__main__":

    main()

