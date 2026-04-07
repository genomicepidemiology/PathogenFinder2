"""
Protein alignment and GSEA post-processing for PathogenFinder2.

:class:`MapProteins` takes the attention-weighted top-N proteins from a
prediction run, aligns them against a Diamond-indexed protein database
(UniRef50/SwissProt), and optionally runs a GSEA-like enrichment analysis
on the GO/functional annotations of the hit proteins.
"""
import argparse
import pandas as pd
import numpy as np
import subprocess
import warnings



class MapProteins:
    """Align top-attention proteins against a Diamond database and run optional GSEA.

    The pipeline executed by :meth:`map_proteins` is:

    1. Load attention weights and select the top-N proteins.
    2. Run Diamond BLASTp against *db_path*.
    3. (Optional) Run ``gseapy.prerank`` on GO/functional term annotations.
    4. Write TSV result files to *folder_out*.

    Parameters
    ----------
    folder_out:
        Directory where final result TSVs are written.
    db_path:
        Path to the Diamond-indexed protein database (``.dmnd`` file).
    diamond_path:
        Path or name on ``$PATH`` of the Diamond executable.
    log_folder:
        Directory for Diamond stderr logs.  Defaults to *folder_out*.
    folder_tmp:
        Directory for temporary files (selected protein FASTA, Diamond output).
        Defaults to *folder_out*.
    metadata:
        Optional path to a SwissProt metadata TSV (required for GSEA).
    """

    def __init__(self, folder_out: str, db_path: str, diamond_path: str,
                 log_folder: str | None = None, folder_tmp: str | None = None,
                 metadata: str | None = None):

        self.db_path = db_path
        self.diamond_path = diamond_path
        self.folder_out = folder_out
        if folder_tmp is None:
            self.folder_tmp = folder_out
        else:
            self.folder_tmp = folder_tmp
        
        if log_folder is None:
            self.log_folder = folder_out
        else:
            self.log_folder = log_folder
        if metadata is None:
            self.metadata = None
        else:
            self.metadata = pd.read_csv(metadata, sep="\t")
            MapProteins.check_metadata(self.metadata)

    @staticmethod
    def check_metadata(metadata: pd.DataFrame) -> None:
        """Raise :exc:`KeyError` if required columns are missing from *metadata*."""
        for col in ["Entry"]:
            if col not in metadata.columns:
                raise KeyError("""The column '{}' does not appear on the metadata file. Check if it comes from Swiss-Prot, """
                                """if the correct columns have been downloaded, and if it has been formated correctly using """
                                """pathogenfinder2 setup_gsea""".format(col))

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

    @staticmethod
    def load_attfile(attfile, combine_max=True, top_proteins=20):
        attnp = np.load(attfile,  allow_pickle=True)
        attdf = pd.DataFrame({"protein_id": attnp["protIDs"]})
        attdf["att0"] = attnp["attentions"][0,:]*len(attnp["attentions"][0,:])
        attdf["att1"] = attnp["attentions"][1,:]*len(attnp["attentions"][1,:])
        attdf["att2"] = attnp["attentions"][2,:]*len(attnp["attentions"][2,:])
        attdf["att3"] = attnp["attentions"][3,:]*len(attnp["attentions"][3,:])
        attdf["protIDs"] = attdf["protein_id"].str.decode("utf-8").str.split(" # ").str[0]
        if combine_max:
            attdf["attmax"] = attdf[["att0", "att1", "att2", "att3"]].max(axis=1)
            attdf["top"] = np.zeros(len(attdf["att0"]), dtype=bool)
            attdf.loc[attdf.index[:top_proteins], "top"] = True
        # Top-N per head (no attmax)
        else:
            cols = ["att0", "att1", "att2", "att3"]
            k = min(int(top_proteins), len(attdf))
            for c in cols:
                attdf[f"{c}_top"] = False
                attdf.loc[attdf[c].nlargest(k).index, f"{c}_top"] = True
            # Optional union flag
            attdf["top"] = attdf[[f"{c}_top" for c in cols]].any(axis=1)
        return attdf

    @staticmethod
    def load_alnfile(alnfile, metadata=None):
        data_diamond = pd.read_csv(alnfile, sep="\t", names=["qseqid","sseqid","pident","length","mismatch","gapopen","qstart","qend",
                                                                "sstart","send","evalue","bitscore","qtitle","stitle", "scovhsp", "slen"])
        result = data_diamond.loc[data_diamond.groupby('qseqid')['pident'].idxmax()]
        result["Entry"] = result['sseqid'].str.extract(r'\|(.*?)\|')
        if metadata is not None:
            result = result.merge(metadata, on="Entry")
        return result

    def run_diamond(self, infile:str, log_folder:str, num_report:int=1) -> str:
        command = "{diamond_path} blastp -q {query_path} -d {db_path} -o {out_path}/out_diamond.tsv".format(
                                            diamond_path=self.diamond_path, query_path=infile, db_path=self.db_path, out_path=self.folder_tmp)
        command_opt = " --faster --max-target-seqs {}".format(num_report)
        command_out = " --outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qtitle stitle scovhsp slen"
        command += command_opt
        command += command_out
        diamond_proc = subprocess.run(command.split(), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True)
        stderr_file = "{}/diamond.stderr".format(log_folder)
        with open(stderr_file, "w") as createdberr:
            createdberr.write(diamond_proc.stderr)
        return "{out_path}/out_diamond.tsv".format(out_path=self.folder_tmp)

    @staticmethod
    def rank_proteins(gsea_df):
        # One score per gene; sort descending (Series is the fastest input for prerank)
        sub = gsea_df.loc[gsea_df["attmax"].notna(), ["qseqid", "attmax"]].copy()
        ranked = (
            sub.groupby("qseqid", as_index=False)["attmax"].max()
                .rename(columns={"attmax": "score"})
        )
        ranked["score"] = ranked["score"] * 1_000_000  # keep your scaling
        rnk = ranked.set_index("qseqid")["score"].sort_values(ascending=False)
        return rnk

    @staticmethod
    def create_proteinsets(gsea_df, terms):
        df = gsea_df.dropna(subset=[terms]).copy()
        df["terms"] = (
            df[terms].astype(str)
            .str.split(";")
            .apply(lambda toks: [t.strip() for t in toks if t and t.strip() != ""])
        )
        long = df.explode("terms")[["terms", "qseqid"]].dropna()
        long = long[long["terms"] != ""]

        go_dict = {}
        for t, sub in long.groupby("terms"):
            # order-preserving dedup
            seen, genes = set(), []
            for g in sub["qseqid"]:
                if g not in seen:
                    seen.add(g); genes.append(g)
            go_dict[t] = genes
        return go_dict

    @staticmethod
    def run_gsea(map_df, att_df, threads=1, min_size=15):
        import gseapy as gp
        if min_size < 15:
            warnings.warn("""The min_size is below standard. The intepretability of the results might be affected""",
                        UserWarning)   
        gsea_df = att_df.merge(map_df, how="left", left_on="protIDs", right_on="qseqid")
        gsea_df["protein_description"] = gsea_df["stitle"].str.split(" ").str[1:].str.join(" ")
        rnk_series = MapProteins.rank_proteins(gsea_df=gsea_df)

        gsea_mapped = gsea_df.dropna(subset=["qseqid"])
        mapping_sseqid = gsea_mapped.set_index("qseqid")["sseqid"].to_dict()
        mapping_descr = gsea_mapped.set_index("qseqid")["protein_description"].to_dict()
        # Use sensible threads; keep your GSEA settings: min_size=5, permutation_num=10000
        max_size = len(rnk_series.index.unique())  # upper bound equals max possible overlap
        results = []
        for term in ["Entry ID", "GO Term"]:
            protein_set = MapProteins.create_proteinsets(gsea_df=gsea_df, terms=term)
            # rnk_series is your Series with index=genes (qseqid) and values=scores
            if len(protein_set) > 0:
                try:
                    pre_res = gp.prerank(
                        rnk=rnk_series,            # Series input is fastest for prerank
                        gene_sets=protein_set,
                        max_size=max_size,         # avoid upper-bound pruning
                        min_size=min_size,
                        permutation_num=10000,     # keep your setting
                        threads=threads,                 # faster than 100 due to overhead
                        no_plot=True,              # avoid figure creation cost
                        outdir=None,
                        seed=7,
                        verbose=False
                    )
                except LookupError:
                    warnings.warn(
                        """No gene sets passed through filtering condition for {}. """
                        """Lowering the min_size will probably solve it, but it has drawbacks on the """
                        """interpretability and hypothesis confirmation.""",
                        UserWarning)
                    gsea_results = pd.DataFrame([], columns=["Name","Term","ES","NES","NOM p-val",
                                                                "FDR q-val","FWER p-val","Tag %","Gene %","Lead_genes",
                                                                "Lead_genes_sseqid", "Lead_genes_description"])
                    gsea_results["Term_class"] = term
                    results.append(gsea_results)
                    continue

                gsea_results = pre_res.res2d
                gsea_results["Lead_genes_sseqid"] = gsea_results["Lead_genes"].str.split(";").apply(
                                                                                    lambda x: ";".join(mapping_sseqid.get(i, i) for i in x))
                gsea_results["Lead_genes_description"] = gsea_results["Lead_genes"].str.split(";").apply(
                                                                                    lambda x: ";".join(mapping_descr.get(i, i) for i in x))
                gsea_results["Term_class"] = term
                results.append(gsea_results)
            else:
                gsea_results = pd.DataFrame([], columns=["Name","Term","ES","NES","NOM p-val",
                                                                "FDR q-val","FWER p-val","Tag %","Gene %","Lead_genes",
                                                                "Lead_genes_sseqid", "Lead_genes_description"])
                gsea_results["Term_class"] = term
                results.append(gsea_results)
        return pd.concat(results, ignore_index=True)
    
    @staticmethod
    def reduce_fsa(fsa_dict, att_df, tmp_folder):
        att_top = att_df[att_df["top"]]
        filtered_dict = {k: v for k, v in fsa_dict.items() if k in set(att_top["protIDs"])}
        fsa_path = "{}/selected_prots.fsa".format(tmp_folder)
        MapProteins.write_protfasta(file_path=fsa_path,
                                    dict_prot=filtered_dict)
        return fsa_path


    def sort_aln_results(self, att_df: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
        att_top = att_df[att_df["top"]]
        att_top = att_top.rename(columns={"protIDs": "qseqid"})
        map_results = map_df.merge(att_top, how="right", on="qseqid")
        map_results = map_results.astype({
            "scovhsp": "float", "pident": "float", "sstart": "float", "send": "float",
            "length": "float", "slen": "float", "qstart": "float", "qend": "float"
        })
        map_results = map_results.fillna("-")
        map_results["stitle"] = map_results["stitle"].replace("-", "No Match Found")
        map_results = map_results.sort_values("attmax")
        map_results["taxname"] = map_results["stitle"].str.split("Tax", n=1).str[-1].str.replace("=", "", regex=False)
        map_results["taxid"] = map_results["stitle"].str.split("TaxID=").str[-1].str.split().str[0]
        return map_results


    def map_proteins(self, att_file: str, prot_file: str, gsea: bool = True,
                     top_proteins: int = 20, gsea_minsize: int = 15) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Run the full alignment (and optional GSEA) pipeline.

        Parameters
        ----------
        att_file:
            Path to the attention ``.npz`` file produced by the ensemble.
        prot_file:
            Path to the predicted protein FASTA file.
        gsea:
            If ``True``, run GSEA on alignment hits (requires *metadata*).
        top_proteins:
            Number of top-attention proteins to align.
        gsea_minsize:
            Minimum gene-set size passed to ``gseapy.prerank``.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame | None]
            Alignment results DataFrame and (if *gsea=True*) GSEA results
            DataFrame, otherwise ``None``.
        """
        fsa_dict = MapProteins.read_protfasta(file_path=prot_file)
        att_df = MapProteins.load_attfile(attfile=att_file, top_proteins=top_proteins)
        att_df[att_df["top"]].to_csv("{}/selected_prots.tsv".format(self.folder_tmp), sep="\t", index=False)
        if gsea:
            if self.metadata is None:
                raise ValueError("No protein database metadata was provided, which is necessary for running GSEA")
        else:
            prot_file = MapProteins.reduce_fsa(fsa_dict=fsa_dict, att_df=att_df, tmp_folder=self.folder_tmp)
        map_path = self.run_diamond(infile=prot_file, log_folder=self.log_folder, num_report=1)
        map_df = MapProteins.load_alnfile(map_path, metadata=self.metadata)
        if gsea:
            gsea_results = MapProteins.run_gsea(map_df=map_df, att_df=att_df, min_size=gsea_minsize)
            gsea_results.to_csv("{}/gsea_results.tsv".format(self.folder_out), sep="\t", index=False)
        else:
            gsea_results = None
        aln_results = self.sort_aln_results(map_df=map_df, att_df=att_df)
        aln_results.to_csv("{}/mapped_proteins.tsv".format(self.folder_out), sep="\t", index=False)
        return aln_results, gsea_results



def get_args():
    parser = argparse.ArgumentParser(description='Mapping your sequence to the Bacterial Pathogenic Landscape')
    ## GENERAL ##
    parser.add_argument('--diamond_path', help='Diamond path')
    parser.add_argument('--db_path', help='Diamond formatted db path', required=True)
    parser.add_argument('--prot_path', help='Path to protein fasta file')
    parser.add_argument('--att_path', help='Path to attention npz file')
    parser.add_argument('--log_folder', help='Folder for the logs of diamond')
    parser.add_argument("--out_folder", help='Folder where to output results', required=True)
    parser.add_argument("--tmp_folder", help="Folder where to output temporary files", default=None)
    #############
    parser.add_argument("--aln_top_genes", help="Align top genes")
    parser.add_argument('--amount_hits', help="Amount of hits reported", default=1)
    parser.add_argument('--amount_prots', help="Amount of proteins reported", default=20)
    ############
    parser.add_argument("--gsea", help="Perform GSEA")
    return parser.parse_args()


def main():
    args = get_args()
    mapprot = MapProteins(
        folder_out=args.out_folder,
        diamond_path=args.diamond_path,
        db_path=args.db_path,
        log_folder=args.log_folder,
        folder_tmp=args.tmp_folder,
    )
    mapprot.map_proteins(
        att_file=args.att_path,
        prot_file=args.prot_path,
        gsea=args.gsea,
        top_proteins=int(args.amount_prots),
    )




