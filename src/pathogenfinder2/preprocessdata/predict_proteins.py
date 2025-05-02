import os
import subprocess
import logging


class Prodigal_Executable:

    #TODO: Check if multiple sequence in one file also works

    def __init__(self, output_folder, log_folder=False, prodigal_path="prodigal"): # Change this to prodigal

        self.prodigal_path = prodigal_path
        self.output_folder = output_folder
        if not log_folder:
            self.log_folder = output_folder
        else:
            self.log_folder = log_folder

    def __call__(self, input_filepath, prot_file=None, aminoacid=True, stats=True, stdout=True):
        logging.info("Running Prodigal for file '{}'".format(input_filepath))

        command = """{prodigal} -i {seqpath}"""
        if prot_file is None:
            prot_file = "{aminofold}/PredictedProteins.faa".format(
                                            aminofold=self.output_folder)
        else:
            prot_file = prot_file
        command += " -a {}".format(prot_file)
        if stdout:
            gbk_file = "{statsfold}/Prodigal_stdout.gbk".format(
                                statsfold=self.output_folder)
            command += " -o {}".format(gbk_file)
        else:
            gbk_file = False
        if stats:
            stats_file = "{statsfold}/Prodigal_stats.tsv".format(
                                statsfold=self.output_folder)
            command += " -s {}".format(stats_file)
        else:
            stats_file = False

        command = command.format(prodigal=self.prodigal_path,
                                    seqpath=input_filepath,
                                    statsfold=self.output_folder).split(" ")
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        output, err = process.communicate()
    
        outstd, errstd =  self.write_stderrout(output=output, err=err)

        return prot_file, gbk_file, stats_file, outstd, errstd

    def write_stderrout(self, output, err):
        outstd_file = "{statsfold}/prodigal.out".format(statsfold=self.log_folder)
        errstd_file = "{statsfold}/prodigal.err".format(statsfold=self.log_folder)
        with open(outstd_file, "wb") as outfile:
            outfile.write(output)
        with open(errstd_file, "wb") as errfile:
            errfile.write(err)
        return outstd_file, errstd_file
