import re
import requests
import os
import subprocess
from tqdm import tqdm
import pandas as pd
from goatools.obo_parser import GODag


class SetupSwissProt:

    REQUIRED = {"Entry", "Entry Name", "Gene Ontology (biological process)"}

    @staticmethod
    def check_metadata(df):

        missing = SetupSwissProt.REQUIRED - set(df.columns)

        if missing:
            raise KeyError(f"DF is missing required columns: {missing}")

    @staticmethod
    def stream_with_progress(url, params, out_path, desc, chunk_size=8192):
        with requests.get(url, params=params, stream=True) as r:
            r.raise_for_status()

            total = r.headers.get("Content-Length")
            total = int(total) if total is not None else None

            with open(out_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                leave=True,
            ) as pbar:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


    @staticmethod
    def download_go_basic(out_folder, out_path="go-basic.obo", fmt="obo"):
        """
        Download the Gene Ontology 'go-basic' ontology.

        Args:
            out_path: output file path (defaults to 'go-basic.obo').
            fmt: one of {'obo', 'json', 'owl'}; defaults to 'obo'.

        Returns:
            The path written.

        Notes:
            - go-basic is an acyclic subset of GO (safe for propagating annotations up the graph).
            - Official formats and explanation on GO docs.
        """
        if fmt not in {"obo", "json", "owl"}:
            raise ValueError("fmt must be 'obo', 'json', or 'owl'")

        base = "http://purl.obolibrary.org/obo/go/go-basic"
        url = f"{base}.{fmt}"  # e.g., .../go-basic.obo or .json or .owl
        out_path = "{}/{}".format(out_folder, out_path)
        # Auto-derive extension if caller didn't match filename to fmt
        if not out_path.endswith(f".{fmt}"):
            out_path = f"{out_path.rsplit('.', 1)[0]}.{fmt}"

        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return os.path.abspath(out_path)

    @staticmethod
    def download_swissprot_bacteria(out_folder,
            fasta_path="uniprot_swissprot_bacteria.fasta",
            tsv_path="uniprot_swissprot_bacteria.tsv"):
        """
        Download Swiss-Prot (reviewed) bacterial proteins from UniProt.

        Outputs:
        - FASTA (plain text, not compressed)
        - TSV with columns:
            Entry, Entry Name, Gene Ontology (biological process)
        """

        query = "(reviewed:true) AND (taxonomy_id:2)"

        # ---- FASTA (plain text) ----
        base = "https://rest.uniprot.org/uniprotkb/stream"

        fasta_path = "{}/{}".format(out_folder, fasta_path)
        tsv_path = "{}/{}".format(out_folder, tsv_path)

        SetupSwissProt.stream_with_progress(
                base,
                {"query": query, "format": "fasta"},
                fasta_path,
                desc="Downloading FASTA",
            )

        # ---- TSV ----

        SetupSwissProt.stream_with_progress(
            base,
            {"query": query, "format": "tsv", "fields": "accession,id,go_p"},
            tsv_path,
            desc="Downloading TSV",
        )
        return os.path.abspath(fasta_path), os.path.abspath(tsv_path)

    @staticmethod
    def bp_filter_and_deepest(
        go_ids,
        go_dag,
        prune_ancestors: bool = True,
        keep_all_ties: bool = False):
        """
        Given a list of GO IDs for one row:
        - return (bp_ids, deepest_bp_ids)

        Parameters
        ----------
        go_ids : List[str]
            GO IDs as provided in your row (e.g., ['GO:0007165', 'GO:0007166'])
        prune_ancestors : bool
            If True, drop any BP term that is an ancestor of another BP term.
        keep_all_ties : bool
            If True, keep all deepest terms if multiple share max depth; else pick one.

        Returns
        -------
        (bp_ids, deepest_bp_ids)
            bp_ids: BP-only list (filtered from go_ids)
            deepest_bp_ids: the deepest BP term(s) after optional pruning
        """
        # (a) keep only BP terms that exist in the DAG
        bp_ids = [g for g in go_ids if g in go_dag and go_dag[g].namespace == "biological_process"]

        # Optional pruning to keep the most specific terms present in this row
        if prune_ancestors:
            ids = bp_ids
            leaves = []
            for gid in ids:
                # drop gid if it is an ancestor of any other gid in the set
                if not any(
                    gid in go_dag[other].get_all_parents()
                    for other in ids if other != gid
                ):
                    leaves.append(gid)
            selected = leaves
        else:
            selected = bp_ids

        # (b) deepest by DAG depth
        if not selected:
            return bp_ids, []
        maxd = max(go_dag[g].depth for g in selected)
        deepest = [g for g in selected if go_dag[g].depth == maxd]
        return bp_ids, (deepest if keep_all_ties else deepest[:1])

    @staticmethod
    def diamond_index(fasta, db_name, diamond_path):
        try:
            subprocess.run(
                [diamond_path, "makedb", "--in", fasta, "--db", db_name],
                check=True
            )
            print(f"DIAMOND DB created: {db_name}.dmnd")
        except subprocess.CalledProcessError as e:
            raise RuntimeError("DIAMOND indexing failed") from e

    @staticmethod
    def set_goterm(go_dag, metadata, column_name="Gene Ontology (biological process)"):
        ancestors_terms_list = []
        for go_string in metadata[column_name]:
            go_string = str(go_string).strip()
            if not go_string or pd.isna(go_string):
                ancestors_terms_list.append("")
                continue

            go_ids = [match.group(0).strip("[]") for term in go_string.split(";")
                        if (match := re.search(r"\[GO:\d+\]", term))]      

            bp_ids, deepest_bp = SetupSwissProt.bp_filter_and_deepest(
                go_ids, go_dag, prune_ancestors=True, keep_all_ties=False
            )
            ancestors_terms_list.append(deepest_bp[0] if deepest_bp else "")
        return pd.Series(ancestors_terms_list)
    
    @staticmethod
    def format_metadata(uniprot_metadata, go_file, output_path):
        if os.path.isfile(os.path.abspath(output_path)):
            raise OSError("{} already exists".format(os.path.abspath(output_path)))
        metadata = pd.read_csv(uniprot_metadata, sep="\t", low_memory=False)
        SetupSwissProt.check_metadata(df=metadata)
        go_dag = GODag(go_file, load_obsolete=True, optional_attrs={'relationship'})
        entryid1 = metadata["Entry Name"].str.split("_").str[0]
        goterms = SetupSwissProt.set_goterm(go_dag, metadata=metadata, column_name="Gene Ontology (biological process)")
        out_df = pd.DataFrame({"Entry": metadata["Entry"], "Entry ID": entryid1, "GO Term": goterms})
        out_df.to_csv(os.path.abspath(output_path), index=False, sep="\t")
        return os.path.abspath(output_path)


if __name__ == "__main__":
    s = SetupSwissProt(uniprot_metadata="/work3/alff/PathogenFinder2/data/proteindb/uniprotkb_taxonomy_id_2_AND_reviewed_tr_2025_11_21.tsv",
                        go_file="/work3/alff/PathogenFinder2/data/proteindb/go-basic.obo")
    s.format_metadata("/work3/alff/PathogenFinder2/data/proteindb/uniprotkb_taxonomy_id_2_AND_reviewed_tr_2025_11_21_formated.tsv")
