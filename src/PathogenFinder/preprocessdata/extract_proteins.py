import os
from pathlib import Path
import subprocess
from utils.file_utils import get_filename

class Prodigal_EXEC:

    #TODO: Check if multiple sequence in one file also works

    def __init__(self, log_folder, output_folder, prodigal_path="pyrodigal"): # Change this to prodigal

        self.prodigal_path = prodigal_path
        self.log_folder = log_folder
        self.output_folder = output_folder


    def __call__(self, file_path):
        print("Running Prodigal for file '{}'".format(file_path))
        abs_filepath = os.path.abspath(file_path)
        seq_name, ext = get_filename(abs_filepath)
        seq_folder = os.path.dirname(abs_filepath)
        
        proteome_path = self.run_prodigal(seq_name=seq_name, abs_filepath=abs_filepath)
        return proteome_path

    def run_prodigal(self, seq_name, abs_filepath, aminoacid=True, cds=False,
                        stats=False, stdout=False, err=True):

        command = """{prodigal} -i {seqpath}"""
        if stdout:
            command += " -o {statsfold}/{seqname}.gbk".format(
                                statsfold=self.output_folder,
                                seqname=seq_name)
        if aminoacid:
            command += " -a {aminofold}/{seqname}.faa".format(
                                aminofold=self.output_folder,
                                seqname=seq_name)
        if cds:
            command += " -d {cdsfold}/{seqname}.fna".format(
                                cdsfold=self.output_folder,
                                seqname=seq_name)
        if stats:
            command += " -s {statsfold}/{seqname}.tsv".format(
                                statsfold=self.output_folder,
                                seqname=seq_name)

        command = command.format(prodigal=self.prodigal_path,
                        seqpath=abs_filepath, seqname=seq_name,
                        statsfold=self.output_folder).split(" ")
        print(" ".join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        output, err = process.communicate()
        if err:
            outstd_file = "{statsfold}/{seqname}.out".format(
                            statsfold=self.log_folder, seqname=seq_name)
            errstd_file = "{statsfold}/{seqname}.err".format(
                            statsfold=self.log_folder, seqname=seq_name)
            with open(outstd_file, "wb") as outfile:
                outfile.write(output)
            with open(errstd_file, "wb") as errfile:
                errfile.write(err)
        return "{aminofold}/{seqname}.faa".format(
                        aminofold=self.output_folder, seqname=seq_name)
