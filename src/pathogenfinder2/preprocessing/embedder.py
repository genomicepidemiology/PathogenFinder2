#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProtT5 protein embedding module for PathogenFinder2.

:class:`ProtT5Embedder` loads the Rostlab/prot_t5_xl_half_uniref50-enc model
(from a local directory or the HuggingFace Hub) and computes per-protein
mean- or max-pooled embeddings from amino-acid sequences in FASTA format.
Results are written to an HDF5 file via :meth:`get_embeddings`.

For proteins longer than ``max_seq_len`` residues an OOM-safe k-mer sliding
window strategy is used (see :meth:`kmer_embed` / :meth:`embed_longbatch`).

Original author: mheinzinger  |  Edits: ffalfred
"""

import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import logging

logger = logging.getLogger(__name__)

import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer
from huggingface_hub import snapshot_download



class ProtT5Embedder:

    def __init__(self, model_dir=None,
              transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc",
              pool_mode="mean"):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model, self.vocab = self.get_T5_model(
            model_dir=model_dir,
            transformer_link=transformer_link,
        )
        if not pool_mode:
            self.pool_mode = None
        elif pool_mode in ("mean", "max"):
            self.pool_mode = pool_mode
        else:
            raise ValueError(
                f"The pool mode '{pool_mode}' is not an option. Only 'mean' and 'max' are available."
            )

    def get_T5_model(self, model_dir=None, transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
        if model_dir is not None and os.path.isdir(model_dir):
            use_local = True
            source = model_dir
            cache_dir = model_dir
        else:
            use_local = False
            source = transformer_link
            cache_dir = None

        # --- Check whether download is needed ---
        download_needed = False

        if not use_local:
            # We are loading from the HUB repo ID
            try:
                # Try to resolve snapshot locally (no download allowed)
                snapshot_download(
                    repo_id=transformer_link,
                    local_files_only=True
                )
                # If this succeeds, model is fully cached → no need to print
            except Exception:
                # Cache miss → download will happen
                download_needed = True

        # --- Print only if download is needed ---
        if download_needed:
            logging.getLogger(__name__).info("Loading ProtT5 from HuggingFace Hub (downloading…)")
        # --- Load model ---
        model = T5EncoderModel.from_pretrained(
            source,
            cache_dir=cache_dir,
            local_files_only=use_local,
        )

        vocab = T5Tokenizer.from_pretrained(
            source,
            do_lower_case=False,
            legacy=True,
            cache_dir=cache_dir,
            local_files_only=use_local,
        )

        # precision cast — torch.device.type gives the device string ('cpu'/'cuda')
        if self.device.type == 'cpu':
            model = model.float()
        else:
            model = model.half()

        model = model.to(self.device).eval()
        return model, vocab


    @staticmethod
    def read_fasta(fasta_path: str) -> dict[str, str]:
        """Read a FASTA file and return a mapping of header → sequence.

        Headers are stripped of ``>`` and leading/trailing whitespace; ``/``
        characters are replaced with ``_`` to produce filesystem-safe keys.
        Gaps are removed and sequences are upper-cased.

        Parameters
        ----------
        fasta_path:
            Path to the input FASTA file.
        """
        sequences = dict()
        with open(fasta_path, 'r') as fasta_f:
            for line in fasta_f:
                # get uniprot ID from header and create new entry
                if line.startswith('>'):
                    uniprot_id = line.replace('>', '').strip()
                    uniprot_id = uniprot_id.replace("/","_")
                    sequences[ uniprot_id ] = ''
                else:
                    # repl. all whie-space chars and join seqs spanning multiple lines
                    sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case

        return sequences

    def pool_embeddings(self, emb):
        if self.pool_mode == "mean":
            emb = emb.mean(dim=0)
        elif self.pool_mode == "max":
            emb = emb.max(dim=0).values
        else:
            raise ValueError("The pool mode '{}' is not an option."
                                  "Only 'mean' and 'max' are available.")
        emb = emb.detach().cpu().numpy().squeeze()
        return emb


    def kmer_embed(self, identifier, input, att_mask, max_len, kmer_overlap=30):
        start_split = 0
        end_split = max_len
        embed_repr_lst = []
        len_seq = sum(att_mask[0])
        stop_iter = False
        identifiers = []
        count = 1
        while True:
            input_slice = input[:, start_split:end_split].to(self.device)
            attmask_slice = att_mask[:, start_split:end_split].to(self.device)
            with torch.no_grad():
                embedding_repr_slice = self.model(input_slice,
                                            attention_mask=attmask_slice)
            s_len = attmask_slice[0].sum(dim=0)
            emb = embedding_repr_slice.last_hidden_state[0,:s_len]
            embedding_repr_detach = self.pool_embeddings(emb)
            subseq_name = "{}_Kmer-{}".format(identifier, count)
            count += 1
            identifiers.append(str(subseq_name))
            embed_repr_lst.append(embedding_repr_detach)
            start_split = end_split-kmer_overlap

            if stop_iter:
                break
            if ((end_split+max_len)-kmer_overlap) >= len_seq:
                end_split = len_seq
                stop_iter = True
            else:
                end_split = (end_split+max_len)-kmer_overlap
        return identifiers, embed_repr_lst

    def embed_longbatch(self, pdb_ids, seq_lens, token_encoding, max_len):
        input_ids      = torch.tensor(token_encoding['input_ids'])
        attention_mask = torch.tensor(token_encoding['attention_mask'])
        embedding_lst = []
        identifiers = []
        kmered_identifier = {}
        for batch_idx, identifier in enumerate(pdb_ids):
            input = input_ids[batch_idx, :]
            input = input[None, :]
            att_mask = attention_mask[batch_idx,:]
            att_mask = att_mask[None,:]
            oom_err = False
            try:
                with torch.no_grad():
                    embedding_repr = self.model(input, attention_mask=att_mask)
            except RuntimeError:
                oom_err = True
            if oom_err:
                identifier_kmer, embedding_repr = self.kmer_embed(identifier,
                                                      input, att_mask, max_len)
                kmered_identifier[str(identifier)] = identifier_kmer
                identifiers.extend(identifier_kmer)
                embedding_lst.extend(embedding_repr)
            else:
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[0, :s_len]
                embedding_repr = self.pool_embeddings(emb)
                embedding_lst.append(embedding_repr)
                identifiers.append(identifier)
        return identifiers, embedding_lst, kmered_identifier

    def get_embeddings(self, seq_path: str, emb_path: str,
                       pool_mode: str | None = None,
                       max_residues: int = 4000,
                       max_seq_len: int = 1000,
                       max_batch: int = 100,
                       split_kmer: bool = False) -> bool:
        """Compute and save per-protein embeddings for all sequences in *seq_path*.

        Parameters
        ----------
        seq_path:
            Path to a FASTA file of protein sequences.
        emb_path:
            Destination HDF5 file path for the computed embeddings.
        pool_mode:
            Override the instance-level pooling mode (``"mean"`` or ``"max"``).
            If ``None`` the mode set at construction time is used.
        max_residues:
            Maximum cumulative residues per tokenisation batch.
        max_seq_len:
            Sequences longer than this are handled one-by-one to avoid OOM.
        max_batch:
            Maximum sequences per tokenisation batch.
        split_kmer:
            If ``True``, very long sequences that cause OOM are split into
            overlapping k-mer windows and embedded independently.

        Returns
        -------
        bool
            Always ``True`` on success.
        """

        if pool_mode is not None:
            self.pool_mode = pool_mode
        embed_file = list()
        names_file = list()
        kmered_prot = {}

        # Read in fasta
        seq_dict = ProtT5Embedder.read_fasta( seq_path )

        n_sequences = len(seq_dict)
        seq_lengths = [len(seq) for seq in seq_dict.values()]
        avg_length = sum(seq_lengths) / n_sequences
        n_long     = sum(1 for l in seq_lengths if l > max_seq_len)

        logger.info("Amount of sequences: %d", len(seq_dict))
        logger.info("Average sequence length: %.1f", avg_length)
        logger.info("Number of sequences >%d: %d", max_seq_len, n_long)

        items   = sorted(seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )
        start = time.time()
        batch = []

        maxLen_embed = max_seq_len

        for seq_idx, (pdb_id, seq) in tqdm(enumerate(items, start=1), total=n_sequences,
                                            desc="Embedding proteins", unit="protein",
                                            dynamic_ncols=True, smoothing=0.05, mininterval=0.2,
                                            bar_format=("{desc} {percentage:>3.0f}% |{bar}| "
                                                        "{n_fmt}/{total_fmt} • {rate_fmt} • ETA {remaining}")):

            seq = seq.replace('U','X').replace('Z','X').replace('O','X')
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            batch.append((pdb_id,seq, seq_len))

            # count residues in current batch and add the last sequence length to
            # avoid that batches with (n_res_batch > max_residues) get processed
            n_res_batch = sum([s_len for  _, _, s_len in batch ]) + seq_len
            if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = list()
                token_encoding = self.vocab(seqs, add_special_tokens=True, padding="longest")
                input_ids      = torch.tensor(token_encoding['input_ids']).to(self.device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)
                oom_err = False
                try:
                    with torch.no_grad():
                        embedding_repr = self.model(input_ids, attention_mask=attention_mask)
                        if max(seq_lens) > maxLen_embed:
                            maxLen_embed = max(seq_lens)
                except RuntimeError:
                    if split_kmer:
                        oom_err = True
                    else:
                        logging.getLogger(__name__).warning(
                            "RuntimeError during embedding for %s (L=%d). Try lowering batch size. "
                            "If single sequence processing does not work, you need more vRAM to process your protein.",
                            pdb_id, seq_len)
                if oom_err:
                    identifiers, embs_detach, kmered_identifier = self.embed_longbatch(pdb_ids,
                                        seq_lens, token_encoding, maxLen_embed)
                    kmered_prot.update(kmered_identifier)
                    embed_file.extend(embs_detach)
                    names_file.extend(identifiers)
                else:
                    # batch-size x seq_len x embedding_dim
                    # extra token is added at the end of the seq
                    for batch_idx, identifier in enumerate(pdb_ids):
                        s_len = seq_lens[batch_idx]
                        # slice-off padded/special tokens
                        emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                        emb_detach = self.pool_embeddings(emb)
                        embed_file.append(emb_detach)
                        names_file.append(identifier)

        end = time.time()
        embeddings_array = np.array(embed_file, dtype=np.float32)
        names_array = np.array(names_file, dtype=object)
        len_file = len(seq_dict)
        len_embed = len(names_array)
        final_kmer = []
        for val in kmered_prot.values():
            final_kmer.append(", ".join(val))
        names_kmer = np.array(final_kmer, dtype=object)

        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(emb_path, "w") as hf:
            hf.attrs['Amount Proteins'] = len_file
            hf.attrs['Amount Embeddings'] = len_embed
            hf.attrs['K-mer Proteins'] = len(kmered_prot)
            hf.attrs['Pooled'] = self.pool_mode
            hf.create_dataset("Embeddings", data=embeddings_array)
            hf.create_dataset("Names", data=list(names_array), dtype=str_dtype)
            hf.create_dataset("K-mer Names", data=list(names_kmer), dtype=str_dtype)
          

        logger.info("Total embeddings: %d | time: %.2fs | time/prot: %.4fs | avg len: %.1f",
                    len_file, end - start, (end - start) / len_embed, avg_length)
        return True


def wrapper_multiple_core(embeder: "ProtT5Embedder", path_file: str, emb_path: str,
                          pool_mode: str, split_kmer: bool,
                          overwrite: bool = False) -> None:
    """Embed all FASTA files listed in *path_file*, writing HDF5s to *emb_path*.

    Parameters
    ----------
    embeder:
        A :class:`ProtT5Embedder` instance.
    path_file:
        Text file where each line is an absolute path to a protein FASTA.
    emb_path:
        Output directory for the ``.h5`` embedding files.
    pool_mode:
        Pooling strategy forwarded to :meth:`ProtT5Embedder.get_embeddings`.
    split_kmer:
        Whether to use k-mer splitting for OOM-prone long sequences.
    overwrite:
        If ``False``, skip files whose embedding already exists.
    """
    with open(path_file, "r") as pf:
        for line in pf:
            file_path = os.path.abspath(line.strip())
            if os.path.isfile(file_path):
                base_name = Path(file_path).stem
                emb_file = os.path.abspath("{}/{}.h5".format(emb_path, base_name))
                if overwrite or not os.path.isfile(emb_file):
                    try:
                        embeder.get_embeddings(file_path, emb_file,
                              pool_mode=pool_mode, split_kmer=split_kmer)
                    except ZeroDivisionError:
                        logger.error("Embedding failed for %s (ZeroDivisionError)", base_name)



def create_arg_parser():
    """Create and return the ArgumentParser object."""""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=(
            't5_embedder.py creates T5 embeddings for a given text '+
            ' file containing sequence(s) in FASTA-format.') )

    # Required positional argument
    parser.add_argument( '-i', '--input', type=str, default=None,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')
    # Required positional argument
    parser.add_argument( '-f', '--file_paths', type=str, default=None,
                    help='A file containing the paths to fasta-formatted text file containing protein sequence(s).')

    # Optional positional argument
    parser.add_argument( '-o', '--output', required=True, type=str,
                    help='A path for saving the created embeddings as NumPy npz file.')

    # Required positional argument
    parser.add_argument('-m', '--model', required=False, type=str,
                    default=None,
                    help='A path to a directory holding the checkpoint for a pre-trained model' )

    # Optional argument
    parser.add_argument('-p', '--pooling', type=str,
                    default=False,
                    help="Wether to pool by max or mean. If not use will return per aminoacid")
    # Optional argument
    parser.add_argument('-s', '--split_long', type=bool,
                    default=False,
                    help="Wether split the protein in overlapping kmers if a protein is too long to fit in GPU.")
        # Required positional argument
    parser.add_argument( '-c', '--cores', type=int,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')

    return parser

def main(embed_out, input_seq=None, input_txt=None, model_path=None,
                pool_mode="mean", split_kmer=True):
    if input_seq is None and input_txt is None:
        raise ValueError("Arguments input_seq and input_txt are mutually required — provide exactly one.")
    elif input_seq is not None and input_txt is not None:
        raise ValueError("Arguments input_seq and input_txt cannot be used at the same time — provide exactly one.")
    elif input_seq is not None and input_txt is None:
        seq_path = Path(input_seq)
        embeder = ProtT5Embedder(model_dir=model_path)
        embeder.get_embeddings(seq_path=seq_path,  emb_path=embed_out,
                           pool_mode=pool_mode, split_kmer=split_kmer)
    else:
        file_path = Path(input_txt)
        embeder = ProtT5Embedder(model_dir=model_path)
        wrapper_multiple_core(embeder=embeder, path_file=file_path,
                    emb_path=embed_out, pool_mode=pool_mode,
                    split_kmer=split_kmer)

if __name__ == '__main__':
    parser = create_arg_parser()

    args = parser.parse_args()
    emb_path = Path(args.output)
    model_dir = Path(args.model) if args.model is not None else None
    pool_mode = args.pooling
    split_kmer = args.split_long

    main(input_seq=args.input, input_txt=args.file_paths, embed_out=emb_path, model_path=model_dir,
                pool_mode=pool_mode, split_kmer=split_kmer)

