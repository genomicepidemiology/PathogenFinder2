#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:33:22 2020

@author: mheinzinger

@edits: ffalfred
"""

import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
import os

import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer


class ProtT5_Embedder:

    def __init__(self, model_dir=None,
              transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc",
              pool_mode="mean"):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model, self.vocab = self.get_T5_model(model_dir, transformer_link)
        if not pool_mode:
            self.pool_mode = None
        elif pool_mode == "mean" or pool_mode == "max":
            self.pool_mode = pool_mode
        else:
            raise ValueError("The pool mode '{}' is not an option."
                            "Only 'mean' and 'max' are available.")


    def get_T5_model(self, model_dir, transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
        print("Loading: {}".format(transformer_link))
        if model_dir is not None:
            print("##########################")
            print("Loading cached model from: {}".format(model_dir))
            print("##########################")
        model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
        model.full() if self.device=='cpu' else model.half() # only cast to full-precision if no GPU is available

        model = model.to(self.device)
        model = model.eval()
        vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )
        return model, vocab

    @staticmethod
    def read_fasta(fasta_path):
        '''
          Reads in fasta file containing multiple sequences.
          Returns dictionary of holding multiple sequences or only single
          sequence, depending on input file.
        '''

        sequences = dict()
        with open( fasta_path, 'r' ) as fasta_f:
            for line in fasta_f:
                # get uniprot ID from header and create new entry
                if line.startswith('>'):
                    uniprot_id = line.replace('>', '').strip()
                    # replace tokens that are mis-interpreted when loading h5
                    uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                    sequences[ uniprot_id ] = ''
                else:
                    # repl. all whie-space chars and join seqs spanning multiple lines
                    sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case

        return sequences

    def pool_embeddings(self, emb, s_len):
        # slice-off padded/special tokens
        if self.pool_mode == "mean":
            emb = emb.mean(dim=0)
        elif self.pool_mode == "max":
            emb = emb.max(dim=0)
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
            embedding_repr_detach = self.pool_embeddings(emb, s_len)
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
        len_batch = input_ids.shape[0]
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
                embedding_repr = self.pool_embeddings(embedding_repr, s_len)
                embedding_lst.append(embedding_repr)
                identifiers.append(identifier)
        return identifiers, embedding_lst, kmered_identifier

    def get_embeddings(self, seq_path, emb_path,
                    pool_mode, # whether to derive per-protein mean or max embeddings
                    max_residues=4000, # number of cumulative residues per batch
                    max_seq_len=1000, # max length after which we switch to single-sequence processing to avoid OOM
                    max_batch=100, # max number of sequences per single batch
                    split_kmer=False
                    ):

        seq_dict = dict()

        len_file = int(0)
        embed_file = list()
        names_file = list()
        if split_kmer:
            kmered_prot = {}
        else:
            kmered_prot = split_kmer

        # Read in fasta
        seq_dict = ProtT5_Embedder.read_fasta( seq_path )
        model, vocab = self.model, self.vocab

        #print('########################################')
        #print('Example sequence: {}\n{}'.format( next(iter(
              #seq_dict.keys())), next(iter(seq_dict.values()))) )
        #print('########################################')
        #print('Total number of sequences: {}'.format(len(seq_dict)))

        avg_length = sum([ len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
        n_long     = sum([ 1 for _, seq in seq_dict.items() if len(seq)>max_seq_len])
        seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )

        print("Amount of sequences: {}".format(len(seq_dict)))
        print("Average sequence length: {}".format(avg_length))
        print("Number of sequences >{}: {}".format(max_seq_len, n_long))

        start = time.time()
        batch = list()

        maxLen_embed = max_seq_len
        count = 0

        for seq_idx, (pdb_id, seq) in tqdm(enumerate(seq_dict,1)):
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
                token_encoding = self.vocab.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                input_ids      = torch.tensor(token_encoding['input_ids']).to(self.device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(self.device)
                oom_err = False
                try:
                    with torch.no_grad():
                        embedding_repr = self.model(input_ids, attention_mask=attention_mask)
                        if max(seq_lens) > maxLen_embed:
                            maxLen_embed = max(seq_lens)
                        else:
                            maxLen_embed = maxLen_embed
                except RuntimeError:
                    if split_kmer:
                        oom_err = True
                    else:
                        print("RuntimeError during embedding for {} (L={}). Try lowering batch size. ".format(pdb_id, seq_len) +
                               "If single sequence processing does not work, you need more vRAM to process your protein.")
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
                        emb_detach = self.pool_embeddings(emb, s_len)
                        embed_file.append(emb_detach)
                        names_file.append(identifier)

        end = time.time()
        embeddings_array = np.array(embed_file, dtype=np.float32)
        names_array = np.array(names_file, dtype=object)
        len_file = len(seq_dict)
        len_embed = len(names_array)
        final_kmer = []
        for k, val in kmered_prot.items():
            final_kmer.append(", ".join(val))
        names_kmer = np.array(final_kmer, dtype=object)

        with h5py.File(emb_path, "w") as hf:
            hf.attrs['Amount Proteins'] = len_file
            hf.attrs['Amount Embeddings'] = len_embed
            hf.attrs['K-mer Proteins'] = len(kmered_prot)
            hf.attrs['Pooled'] = pool_mode
            hf.create_dataset("Embeddings", data=embeddings_array)
            hf.create_dataset("Names", data=names_array)
            hf.create_dataset("K-mer Names", data=names_array)
          

        print('\n############# STATS #############')
        print('Total number of embeddings: {}'.format(len_file))
        print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(
                  end-start, (end-start)/len_embed, avg_length))
        return True


def wrapper_multipleCore(embeder, path_file, emb_path, pool_mode,
                         split_kmer, overwritte=False):
    with open(path_file, "r") as pf:
        for line in pf:
            file_path = os.path.abspath(line.strip())
            if os.path.isfile(file_path):
                base_name = Path(file_path).stem
                emb_file = os.path.abspath("{}/{}.h5".format(emb_path, base_name))
                if overwritte or not os.path.isfile(emb_file):
                    print(base_name)
                    try:
                        embeder.get_embeddings(file_path, emb_file,
                              pool_mode=pool_mode, split_kmer=split_kmer)
                    except ZeroDivisionError:
                        print("{} ERROR".format(base_name))



def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

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
        raise argparse.ArgumentParser.error("Arguments input and file_paths cannot be used at the same time.")
    elif input_seq is not None and input_txt is not None:
        raise argparse.ArgumentParser.error("The argument input or the argument file_paths are needed.")
    elif input_seq is not None and input_txt is None:
        seq_path = Path(input_seq)
        embeder = ProtT5_Embedder()
        embeder.get_embeddings(seq_path=seq_path,  emb_path=embed_out, pool_mode=pool_mode,
                           split_kmer=split_kmer)
    else:
        file_path = Path(input_txt)
        embeder = ProtT5_Embedder()
        wrapper_multipleCore(embeder=embeder, path_file=file_path,
                    emb_path=emb_path, pool_mode=pool_mode,
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

