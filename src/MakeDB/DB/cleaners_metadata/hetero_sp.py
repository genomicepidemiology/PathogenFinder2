import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_species_het(all_df, train1, val1):
    hetero_tax_db = pd.crosstab(all_df["species_taxid_refseq"], all_df["PathoPhenotype"])
    hetero_tax_db = hetero_tax_db.reset_index()
    hetero_tax_db["Count"] = hetero_tax_db["No Pathogenic"] + hetero_tax_db["Pathogenic"]
    hetero_tax_db["Ratio"] = hetero_tax_db["Pathogenic"].astype(float)/hetero_tax_db["Count"].astype(float)
    hetero_tax_db = hetero_tax_db[(hetero_tax_db["Pathogenic"]!=0)&(hetero_tax_db["No Pathogenic"]!=0)]
    new_het = hetero_tax_db.merge(all_df[["species_taxid_refseq", "Species_Refseq"]], on='species_taxid_refseq', how='left')
    hetero_tax_db = new_het.drop_duplicates(subset=["species_taxid_refseq","No Pathogenic","Pathogenic","Count","Ratio"])
#   hetero_tax_db = hetero_tax_db.reset_index()
    hetero_tax_db.to_csv("../filtered_metadata/dec2024/heterogencity2_tax.tsv", sep="\t", index=False)
    hetero_complete = all_df.merge(hetero_tax_db, on="species_taxid_refseq", how="left")
    hetero_complete = hetero_complete[~hetero_complete["Ratio"].isnull()]
    hetero_complete.to_csv("../filtered_metadata/dec2024/refseqhumansnovelpathogenicity2_heterogencity2.tsv", sep="\t", index=False)
    trained_df = pd.concat([train1,val1],ignore_index=True)
    hetero_tax_test = hetero_tax_db.merge(trained_df, on="species_taxid_refseq", how="left", suffixes=('', '_y'))
    hetero_tax_test = hetero_tax_test[hetero_tax_test["File_Embedding"].isnull()]
    for col in hetero_tax_test.columns:
        if col not in ["Count", "Ratio", "No Pathogenic", "Pathogenic", "species_taxid_refseq", "Species_Refseq"]:
            del hetero_tax_test[col]
 #   new_het = hetero_tax_db.merge(all_df[["species_taxid_refseq", "Species_Refseq"]], on='species_taxid_refseq', how='left')
#    hetero_tax_db = new_het.drop_duplicates(subset=["species_taxid_refseq","No Pathogenic","Pathogenic","Count","Ratio"])
    hetero_tax_test.to_csv("../filtered_metadata/dec2024/heterogencity2_tax_test.tsv", sep="\t", index=False)
    test_hetero = all_df.merge(hetero_tax_test, on="species_taxid_refseq", how="left")
    test_hetero = test_hetero[~test_hetero["Ratio"].isnull()]
    test_hetero.to_csv("../filtered_metadata/dec2024/refseqhumansnovelpathogenicity2_heterogencity2_test.tsv", sep="\t", index=False)
    return hetero_tax_db

def graph_stacked(hetero_tax_db):
    hetero_tax_db = hetero_tax_db[hetero_tax_db["Count"]>10]
    hetero_tax_db = hetero_tax_db[(hetero_tax_db["Ratio"]<0.75)&(hetero_tax_db["Ratio"]>0.25)]
    hetero_tax_db = hetero_tax_db.sort_values(by="Count", ascending=False)
    weight_counts = {
        "Pathogenic": np.array(hetero_tax_db["Pathogenic"].tolist()),
        "No Pathogenic": np.array(hetero_tax_db["No Pathogenic"].tolist()),
        }
    species = hetero_tax_db["species_taxid_refseq"].astype(str).tolist()
    fig, ax = plt.subplots()
    bottom = np.zeros(len(species))
    for boolean, weight_count in weight_counts.items():
        ax.bar(species, weight_count, label=boolean, bottom=bottom)
        bottom += weight_count
    
    ax.set_title("Number of penguins with above average body mass")
    ax.legend(loc="upper right")

    plt.savefig("./meh.png")
    plt.close()




all_df = pd.read_csv("../filtered_metadata/dec2024/refseqhumansnovelpathogenicity2.tsv", sep="\t")
train1 = pd.read_csv("../../../../../database/METADATA_Train1DF_protLim_phageclean.tsv", sep="\t")
val1 = pd.read_csv("../../../../../database/METADATA_Val1DF_protLim_phageclean.tsv", sep="\t")

hetero_tax_db = get_species_het(all_df, train1, val1)

graph_stacked(hetero_tax_db)


#print(all_df)


