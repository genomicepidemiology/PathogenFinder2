import pandas as pd
import os
import subprocess
import urllib.request
import numpy as np
from tqdm import tqdm
import argparse
import gc


class PathogenDB:

    def __init__(self, out_folder, db_path, get_proteins=True, get_bpe=False,
                    prodigal_path=None, replace=False):

        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
#        if len(os.listdir(out_folder)) > 0:
            raise OSError("Folder {} has data in it.")

        self.define_folders(out_folder, prot_folders=get_proteins,
                                prodigal_path=prodigal_path)

        self.get_proteins = get_proteins
        self.get_bpe = get_bpe
        self.replace_mode = replace

        self.origin_db = pd.read_csv(db_path, sep="\t")

    def define_folders(self, out_folder, prot_folders, prodigal_path):
        self.genomes_folder = "{}/genome_files".format(out_folder)
        if not os.path.isdir(self.genomes_folder):
            os.mkdir(self.genomes_folder)
        self.subset_db = None
        self.out_folder = out_folder
        if prot_folders:
            self.protein_folder = "{}/protein_files".format(self.out_folder)
            self.cds_folder = "{}/cds_files".format(self.out_folder)
            self.prodigal_stats_folder = "{}/prodigal_stats".format(self.out_folder)
            self.prodigal_path = prodigal_path

        else:
            self.protein_folder = None
            self.cds_folder = None
            self.prodigal_stats_folder = None


    def download_fasta(self, file_out, address):
        a=urllib.request.urlretrieve(address, file_out)
        del a
        gc.collect()

    def representative(self, count, pathogenicity="No Pathogenic", relative=False):
        patho = self.origin_db[self.origin_db["PathoPhenotype"]==pathogenicity]
        patho_sp = patho["Species"].value_counts()[:count]
        if relative:
            total = patho_sp.sum()
            relative_df = (patho_sp/total)*count
            num_down = relative_df.round(0)
            num_down = num_down[num_down>0]
        else:
            num_down = pd.Series([1]*count, index=patho_sp.index)
        index_lst = []
        for index, row in num_down.items():
            specie_down = patho.loc[patho["Species"]==index]
            download_sample = specie_down.sample(n=int(row)).index
            index_lst.extend(download_sample)
        return index_lst

    def create_db(self, db_origin="all", number="all", patho_ratio=None, repr_mode="samples",
                    prodigal="light", section=None):
        if db_origin == "all":
            db_create = self.origin_db
        elif db_origin == "representative":
            if repr_mode == "samples":
                relative = True
            else:
                relative = False
            indexes_download = self.representative(count=number, relative=relative)
            db_create = self.origin_db.loc[indexes_download]
        else:
            index_subset = []
            for origin in db_origin:
                index_subset.extend(self.origin_db["Origin DB"].str.contains(origin).index)
            db_create = self.origin_db.loc[index_subset]
        if patho_ratio is None:
            if number == "all":
                indexes_download = db_create.index
            elif section is not None:
                indexes_download = db_create.iloc[section*1307:(section+1)*1307].index
            else:
                indexes_download = np.random.choice(len(db_create), number,
                                                    replace=False)
        else:
            patho_ind = db_create[db_create["PathoPhenotype"]=="Pathogenic"].index
            nonpatho_ind = db_create[db_create["PathoPhenotype"]=="No Pathogenic"].index
            ind_download_patho = np.random.choice(patho_ind, int((patho_ratio*number)),
                                                    replace=False)
            ind_download_nonpatho = np.random.choice(patho_ind,
                                        int((1-patho_ratio)*number), replace=False)
            indexes_download = []
            indexes_download.extend(ind_download_patho)
            indexes_download.extend(ind_download_nonpatho)
        print(indexes_download)
        db_download = db_create.iloc[indexes_download]
       # print(len(db_download), "SECTION: ", str(section*1307),str((section+1)*1307))
        if not self.get_proteins:
            db_download['ftp_path_refseq_txt'] = db_download['ftp_path_refseq'] + "/" + db_download['ftp_path_refseq'].str.split("/").str[-1] + "_genomic.fna.gz"
            db_download['ftp_path_refseq_txt'].to_csv('./download_wget_genome.txt', header=None, index=None, sep=' ')
            exit()
        paths = db_download["ftp_path_refseq"].tolist()
 #       for index, row in db_download.iterrows():
        del db_download
        wget_lst = ""
        count = 0
        for path in paths:
#            print(index)
            seq_name = path.split("/")[-1]
            filename = "{}_genomic.fna.gz".format(seq_name)
            file_out = "{}/{}".format(self.genomes_folder, filename)
            address_wget = "{}/{}".format(path, filename)
            #print("Sequence {}: {}".format(str(index), seq_name))
            if not os.path.isfile(file_out) and not self.replace_mode:
                self.download_fasta(file_out=file_out, address=address_wget)
                pass
            if self.get_proteins:
                print("{aminofold}/{seqname}_genomic.faa".format(aminofold=self.protein_folder,
                                seqname=seq_name))
                if not os.path.isfile("{aminofold}/{seqname}_genomic.faa".format(aminofold=self.protein_folder,
                                seqname=seq_name)) and not self.replace_mode:
                    unzip_file = PathogenDB.unzip_file(file=file_out)
                    if prodigal != "light":
                        self.get_proteinseq(seq_file=unzip_file)
                    else:
                        self.get_proteinseq(seq_file=unzip_file, aminoacid=True, cds=False, stats=False,
                                        stdout=False, err=True)
                    PathogenDB.delete_file(file="{}/{}".format(self.genomes_folder,unzip_file))
                    del unzip_file
            else:
         #       wget_lst += address_wget +"\n"
                print(count)
                count+=1
            del path, seq_name, filename, file_out, address_wget
            gc.collect()
#        if not self.get_proteins:
 #           with open("./download_wget_genome.txt", "r") as dwn_wget:
  #              dwn_wget.write(wget_lst)

#        db_download.to_csv("{}/db_downloaded.tsv".format(self.out_folder),
 #                           index=False)
    @staticmethod
    def unzip_file(file):
        command = """gunzip -k {}""".format(file)
#        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE,
 #                                   stderr=subprocess.PIPE)
        result = subprocess.run(command.split(), capture_output=True, text=True)
        output = result.stdout
        err = result.stderr
        del output, err, command, result
        gc.collect()
        file_name = os.path.basename(file).replace(".gz", "")
        return file_name

    @staticmethod
    def delete_file(file):
        command = """rm {}""".format(file)
        process = subprocess.run(command.split(), capture_output=True,
                                    text=True)
        output = process.stdout
        err = process.stderr

        del command, process, output, err
        gc.collect()

    def get_proteinseq(self, seq_file, aminoacid=True, cds=True, stats=False, stdout=True, err=True):
        command = """{prodigal} -i {infold}/{seqname} """
        if stdout:
            command += " -o {statsfold}/{seqname}.gbk".format(
                                statsfold=self.prodigal_stats_folder,
                                seqname=seq_file)
        if aminoacid:
            command += " -a {aminofold}/{seqname}".format(
                                aminofold=self.protein_folder,
                                seqname=seq_file.replace("fna","faa"))
        if cds:
            command += " -d {cdsfold}/{seqname}".format(
                                cdsfold=self.cds_folder, seqname=seq_file)
        if stats:
            command += " -s {statsfold}/{seqname}.tsv".format(
                                statsfold=self.prodigal_stats_folder,
                                seqname=seq_file)
        command = command.format(prodigal=self.prodigal_path,
                        infold=self.genomes_folder, seqname=seq_file,
                        statsfold=self.prodigal_stats_folder).split()
        process = subprocess.run(command, capture_output=True,
                                    text=True)
#        process = subprocess.Popen(command, stdout=subprocess.PIPE,
 #                                   stderr=subprocess.PIPE)
#        output, err = process.communicate()
        output = process.stdout
        err = process.stderr
        if err:
            outstd_file = "{statsfold}/{seqname}.out".format(
                            statsfold=self.prodigal_stats_folder, seqname=seq_file)
            errstd_file = "{statsfold}/{seqname}.err".format(
                            statsfold=self.prodigal_stats_folder, seqname=seq_file)
            with open(outstd_file, "w") as outfile:
                outfile.write(output)
            with open(errstd_file, "w") as errfile:
                errfile.write(err)
            del outstd_file, errstd_file, outfile, errfile
        del command, stdout, aminoacid, cds, stats, process, output, err, seq_file
        gc.collect()

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        #parser.add_argument("-s","--section", type=int, help="Section", default=None)
        parser.add_argument("-p", "--prodigal", help="Path to Prodigal", default=False)
        parser.add_argument("-t", "--tsv_path", help="Path to tsv input")
        parser.add_argument("-o", "--out_path", help="Path to out folder")
        parser.add_argument("-r", "--replace", help="Replace?", default=False)
        args = parser.parse_args()
        return args


if __name__ == '__main__':
    #instance = PathogenDB(out_folder="./nonpathogen_50repr/",
     #                   db_path="./humansnovelpathogenicity.tsv",
      #                  prodigal_path="/home/alfred/bio_tools/Prodigal/prodigal")
    #instance.create_db(number=50, db_origin="representative", repr_mode="repr")

    #instance = PathogenDB(out_folder="./nonpathogen_50samples/",
     #                   db_path="./humansnovelpathogenicity.tsv",
                        #prodigal_path="/home/alfred/bio_tools/Prodigal/prodigal")
    #instance.create_db(number=50, db_origin="representative", repr_mode="samples")
    args = PathogenDB.parse_args()
    if args.prodigal:
        get_proteins=True
    else:
        get_proteins=False
    create_refseq = PathogenDB(out_folder=args.out_path,
                            db_path=args.tsv_path,
                            prodigal_path=args.prodigal,
                            replace=args.replace, get_proteins=get_proteins)
    create_refseq.create_db(number="all", db_origin="all")
