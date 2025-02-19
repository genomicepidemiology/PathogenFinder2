import pandas as pd
from tqdm import tqdm
import os
import urllib.request
import gc


def get_db(excel_file, out_file, genbank_file, download_folder):
    dfexcel = pd.read_excel(excel_file, skiprows=[0,1])
    no_patho = dfexcel[dfexcel["Case or Control"]=="Control"]
    no_patho = no_patho[["Isolate","Location","Case or Control","Assembly Accession (GenBank)"]]
    no_patho["PathoPhenotype"] = "No Pathogenic"
    patho = dfexcel[dfexcel["Case or Control"]=="Case"]
    patho = patho[["Isolate","Location","Case or Control","Assembly Accession (GenBank)"]]
    patho["PathoPhenotype"] = "Pathogenic"
    print(patho, no_patho)
    df_ecoli = pd.concat((patho,no_patho))
    genbank_file = pd.read_csv(genbank_file, sep="\t", skiprows=[0])
    genbank_file.columns = genbank_file.columns.str.replace("#", "")
    ftp_files = []
    for index, rows in tqdm(df_ecoli.iterrows()):
        path = str(genbank_file.loc[genbank_file["wgs_master"].str.contains(rows["Assembly Accession (GenBank)"]), "ftp_path"].values[0])
        download_path = "{}/{}_genomic.fna.gz".format(path, os.path.basename(path))
        print(download_path)
        ftp_files.append(path)
        file_out = "{}/{}_genomic.fna.gz".format(download_folder, os.path.basename(path))
        a=urllib.request.urlretrieve(download_path, file_out)
        del a
        gc.collect()

    df_ecoli["ftp_links"] = ftp_files
    df_ecoli["Species"] = "Escherichia Coli"
    df_ecoli.to_csv(out_file)



#get_db(excel_file="../downloaded_metadata/dec2024/41467_2023_36337_MOESM4_ESM.xlsx", out_file="../filtered_metadata/dec2024/het_ecoli.tsv",
#        genbank_file="../misc_metadata/dec2024/assembly_summary_genbank.txt", download_folder="../het_ecoli_files/")
het_df = pd.read_csv("../filtered_metadata/dec2024/het_ecoli.tsv",index_col=0)
print(het_df)
filenames = []
pheno = []
for index, row in het_df.iterrows():
    filenames.append("{}_genomic.fna.gz".format(os.path.basename(str(row["ftp_links"]))))
    if row["PathoPhenotype"] == "Pathogenic":
        pheno.append(1)
    else:
        pheno.append(0)
pheno_data = pd.DataFrame({"Input File": filenames, "PathoPhenotype":pheno})
pheno_data.to_csv("het_ecoli_pheno.tsv", sep="\t", index=False)
