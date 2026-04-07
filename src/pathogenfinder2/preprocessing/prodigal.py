"""
Prodigal protein-prediction wrapper for PathogenFinder2.

:class:`ProdigalRunner` is a callable that invokes the Prodigal gene-prediction
executable on a genome FASTA file and writes the predicted protein sequences,
GBK output, and statistics to the configured output directory.
"""
import os
import subprocess
import logging

logger = logging.getLogger(__name__)


class ProdigalRunner:
    """Wrapper around the Prodigal gene-prediction executable.

    Call an instance with a genome file path to run Prodigal and retrieve
    the paths to all output files.

    Parameters
    ----------
    output_folder:
        Directory where Prodigal output files (proteins, GBK, stats) are written.
    log_folder:
        Directory for Prodigal stdout/stderr logs.  Defaults to *output_folder*.
    prodigal_path:
        Path or name on ``$PATH`` of the Prodigal executable.

    Notes
    -----
    # TODO: Check if multiple sequences in one file (multi-contig genomes) also
    # work correctly end-to-end.
    """

    def __init__(self, output_folder: str, log_folder: str | None = None,
                 prodigal_path: str = "prodigal"):
        self.prodigal_path = prodigal_path
        self.output_folder = output_folder
        self.log_folder = output_folder if log_folder is None else log_folder

    def __call__(self, input_filepath: str, prot_file: str | None = None,
                 aminoacid: bool = True, stats: bool = True,
                 stdout: bool = True) -> tuple[str, str | None, str | None, str, str, str]:
        """Run Prodigal on *input_filepath* and return output file paths.

        Parameters
        ----------
        input_filepath:
            Path to the input genome FASTA file.
        prot_file:
            Destination path for the predicted protein FASTA.  Defaults to
            ``<output_folder>/PredictedProteins.faa``.
        aminoacid:
            Unused; reserved for future amino-acid mode flag.
        stats:
            If ``True``, write a per-gene statistics TSV.
        stdout:
            If ``True``, write Prodigal's GBK stdout to a file.

        Returns
        -------
        tuple
            ``(prot_file, gbk_file, stats_file, outstd_path, errstd_path, command_str)``
        """
        logger.info("Running Prodigal for file '%s'", input_filepath)

        if prot_file is None:
            prot_file = "{}/PredictedProteins.faa".format(self.output_folder)

        command = "{prodigal} -i {seqpath} -a {prot_file}".format(
            prodigal=self.prodigal_path,
            seqpath=input_filepath,
            prot_file=prot_file,
        )

        if stdout:
            gbk_file = "{}/Prodigal_stdout.gbk".format(self.output_folder)
            command += " -o {}".format(gbk_file)
        else:
            gbk_file = None

        if stats:
            stats_file = "{}/Prodigal_stats.tsv".format(self.output_folder)
            command += " -s {}".format(stats_file)
        else:
            stats_file = None

        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        output, err = process.communicate()
        outstd, errstd = self.write_stderrout(output=output, err=err)
        return prot_file, gbk_file, stats_file, outstd, errstd, command

    def write_stderrout(self, output: bytes, err: bytes) -> tuple[str, str]:
        """Write Prodigal stdout/stderr bytes to log files.

        Returns
        -------
        tuple[str, str]
            Paths to the stdout and stderr log files.
        """
        outstd_file = "{}/prodigal.out".format(self.log_folder)
        errstd_file = "{}/prodigal.err".format(self.log_folder)
        with open(outstd_file, "wb") as outfile:
            outfile.write(output)
        with open(errstd_file, "wb") as errfile:
            errfile.write(err)
        return outstd_file, errstd_file

    @staticmethod
    def count_proteins(prot_file: str) -> int:
        """Count the number of sequences in a FASTA file.

        Parameters
        ----------
        prot_file:
            Path to a FASTA-formatted protein file.

        Returns
        -------
        int
            Number of ``>`` header lines (i.e. number of sequences).
        """
        amount_prots = 0
        with open(prot_file, "r") as pf:
            for line in pf:
                if line.startswith(">"):
                    amount_prots += 1
        return amount_prots
