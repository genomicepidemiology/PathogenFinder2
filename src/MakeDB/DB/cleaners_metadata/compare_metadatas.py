import pandas as pd
import argparse




def extract_new_entries(old_db, new_db, out_path, dbtype):

    old_df = pd.read_csv(old_db, sep=",")
    new_df = pd.read_csv(new_db, sep=",")
    
    print(old_df, new_df)
    if dbtype == "ena":
        col_merge = "accession"
    elif dbtype == "ncbi":
        col_merge = "BioSample"
    else:
        col_merge = "Genome ID"
    print(new_df[col_merge])
    new_entries = new_df.merge(old_df, on=col_merge,
                   how='left', indicator=True)
    new_entries = new_entries[new_entries['_merge'] == 'left_only']
    for col in new_entries.columns:
        if col.endswith("_y"):
            del new_entries[col]
    new_entries.columns = new_entries.columns.str.replace("_x", "")
    new_entries.to_csv(out_path, sep="\t", index=False)





def arguments_compare():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--old_df", help="Path to old df")
    parser.add_argument("-n", "--new_df", help="Path to new df")
    parser.add_argument('-d', '--db_type', help="Type DB", choices=["ena", "ncbi", "patric"])
    parser.add_argument("-o", "--out_path", help="Path to out folder")
    args = parser.parse_args()
    return args


args =arguments_compare()

extract_new_entries(old_db=args.old_df, new_db=args.new_df, out_path=args.out_path, dbtype=args.db_type)
