import pandas as pd
import argparse
import datetime




def extract_new_entries(old_db, new_db, out_path):

    old_df = pd.read_csv(old_db, sep="\t")
    new_df = pd.read_csv(new_db, sep="\t")

    new_entries = new_df.merge(old_df.drop_duplicates(), left_on='assembly_accession_refseq', right_on='# assembly_accession_refseq', 
                   how='left', indicator=True)
    new_entries = new_entries[new_entries['_merge'] == 'left_only']
    for col in new_entries.columns:
        if col.endswith("_y"):
            del new_entries[col]
    new_entries.columns = new_entries.columns.str.replace("_x", "")
    new_entries.to_csv(out_path, sep="\t", index=False)

def extract_newdata_entries(new_db, date, out_path):
    new_df = pd.read_csv(new_db, sep="\t")
    print(new_df["seq_rel_date_refseq"])



def arguments_diff():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--old_df", help="Path to old df", default=False)
    parser.add_argument("-n", "--new_df", help="Path to new df")
    parser.add_argument('--date', type=datetime.date.fromisoformat)
    parser.add_argument("-o", "--out_path", help="Path to out folder")
    args = parser.parse_args()
    return args

args = arguments_diff()
if args.old_df:
    extract_new_entries(old_db=args.old_df, new_db=args.new_df, out_path=args.out_path)
else:
    extract_newdata_entries(new_db=args.new_df, date=args.date, out_path=args.out_path)

