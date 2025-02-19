import pandas as pd
import numpy as np
import os
import re
import sys
import json
import gc
import pickle
from itertools import islice
###import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import argparse
import nltk
import datetime

nlp = spacy.load('en_core_web_sm')

nltk.download("stopwords")


class DB_from_BVBRC:

    Check = {"Other Clinical": ["host_description", "host_disease_stage",
        "host_health_state", "treatment course", "Co-morbidity",
        "patient history | co-morbid medical conditions (up to 5 per patient)",
        "patient treatment history |other respiratory pathogens identified (up to 6 per patient)",
        "patient history | vital status at study completion",
        "patient treatment history | was patient in the icu?",
        "patient treatment history | any early complication of infection",
        "patient treatment history | other complications at any time during hospitalization",
        "comorbidity", "patient history | clinical symptoms | symptoms duration",
        "patient history | clinical presentation of current tb disease", "infection"],
        "Additional Metadata": ["phenotype", "passage_history", "outcome",
        "patient status", "outbreak", "patient_outcome", "disease characteristic"]}

    Unsure_patho = ["_health_"]
    Center_synonyms = ["Faculty", "Center", "University", "System", "Public",
                    "Private", "Laboratory", "Facility", "Department", "Dept."]
    VERBS_contrary = ["prevent", "avoid", "block", "impede", "inhibit", "limit", "prohibit",
                    "restrain", "restrict", "stop"]
    NEGATIVE_vocab = ["no", "no-", "neither", "anti", "vaccine", "non", "not"]

    def __init__(self):

        self.sw_nltk = stopwords.words('english') + ["str.", "sp."]
        self.lemmatizer = WordNetLemmatizer()
        self.db = None
        self.metadata_patho = None
        self.unhealthy_vocab = []

    def set_db(self, db):
        self.db = pd.read_csv(db, sep=",", dtype={"Genome ID":str,
            "Genome Name":str, "Other Names":str, "NCBI Taxon ID":int,
            "Taxon Lineage IDs":str, "Taxon Lineage Names":str,
            "Superkingdom":str, "Kingdom":str, "Phylum":str, "Class":str,
            "Order":str, "Family":str, "Genus":str, "Species":str,
            "Genome Status":str, "Strain":str, "Serovar":str, "Biovar":str,
            "Pathovar":str, "MLST":str, "Segment":str, "Subtype":str,
            "H_type":str, "N_type":str, "H1 Clade Global":str,
            "H1 Clade US":str, "H5 Clade":str, "pH1N1-like":str,
            "Lineage":str, "Clade":str, "Subclade":str, "Other Typing":str,
            "Culture Collection":str, "Type Strain":str, "Reference":str,
            "Genome Quality":str, "Completion Date":str, "Publication":str,
            "Authors":str, "BioProject Accession":str,
            "BioSample Accession":str, "Assembly Accession":str,
            "SRA Accession":str, "GenBank Accessions":str,
            "Sequencing Center":str, "Sequencing Status":str,
            "Sequencing Platform":str, "Sequencing Depth":str,
            "Assembly Method":str, "Chromosome":str, "Plasmids":str,
            "Contigs":str, "Size":str, "GC Content":str, "Contig L50":str,
            "Contig N50":str, "TRNA":str, "RRNA":str, "Mat Peptide":str,
            "CDS":str, "Coarse Consistency":str, "Fine Consistency":str,
            "CheckM Contamination":str, "CheckM Completeness":str,
            "Genome Quality Flags":str, "Isolation Source":str,
            "Isolation Comments":str, "Collection Date":str,
            "Collection Year":str, "Season":str, "Isolation Country":str,
            "Geographic Group":str, "Geographic Location":str,
            "Other Environmental":str, "Host Name":str, "Host Common Name":str,
            "Host Gender":str, "Host Age":str, "Host Health":str,
            "Host Group":str, "Lab Host":str, "Passage":str,
            "Other Clinical":str, "Additional Metadata":str, "Comments":str,
            "Date Inserted":str, "Date Modified":str})

    def select_fromdate(self, start_date):
        print(self.db["Date Modified"])
        self.db["year_col"]=None
        self.db["month_col"]=None
        self.db["day_col"] = None
        self.db["year_col"] = self.db["Date Inserted"].str.split("T").str[0].str.split("-").str[0]
        self.db["month_col"] = self.db["Date Inserted"].str.split("T").str[0].str.split("-").str[1]
        self.db["day_col"] = self.db["Date Inserted"].str.split("T").str[0].str.split("-").str[2]
        self.db["year_col"] = self.db["year_col"].astype(int)
        self.db["month_col"] = self.db["month_col"].astype(int)
        self.db["day_col"] = self.db["day_col"].astype(int)
        self.db = self.db[self.db["year_col"]>=int(start_date.year)]
        if start_date.month != 1:
            self.db = self.db[self.db["month_col"]!=None]
            self.db = self.db[self.db["month_col"]>=int(start_date.month)]
        if start_date.day != 1:
            self.db = self.db[self.db["day_col"]!=None]
            self.db = self.db[self.db["day_col"]>=int(start_date.day)]
        del self.db["year_col"]
        del self.db["month_col"]
        del self.db["day_col"]
        self.db.reset_index(inplace=True)


    def set_metadata_patho(self, path, unhealthy_cols=["Pathogen",
                            "Possible Pathogen", "Disease Caused By Pathogen",
                            "Disease maybe caused by pathogen",
                            "Unhealthy"]):
        with open(path, "r") as file:
            self.metadata_patho = json.load(file)
        for col in unhealthy_cols:
            self.unhealthy_vocab.extend(self.metadata_patho[col])
    
    @staticmethod
    def check_organizations_names():
        pass

    @staticmethod
    def analyze_sentence_pathogen(text, patho_regex, avoid_regex, exact):
        sick_sentence = False 
        match_ = None
        for sentence in nlp(text).sents:
            sick_sentence = False
            beginning_check = 0
            sentence = str(sentence)
            if exact:
                pattern_match = re.finditer(patho_regex, sentence)
            else:
                pattern_match = re.finditer(patho_regex, sentence, re.IGNORECASE)
            pattern_match = list(pattern_match)
            if len(pattern_match) > 0:
                for match_ in pattern_match:
                    sick_sentence = True
                    end_check = match_.span()
                    span_sentence = sentence[int(beginning_check):int(end_check[0])]
                    sick_sentence= DB_from_BVBRC.get_negative(text=span_sentence,
                                                                sick_sentence=sick_sentence)
                    beginning_check = end_check[0]
                    # Check if avoid is on the sentence
                    avoid_match = re.finditer(avoid_regex, span_sentence,  re.IGNORECASE)
                    avoid_match = list(avoid_match)
                    if len(avoid_match) > 0:
                        sentence_check_avoid = span_sentence[:int(avoid_match[-1].span()[0])]
                        sick_sentence = False
                        sick_sentence= DB_from_BVBRC.get_negative(text=sentence_check_avoid,
                                                            sick_sentence=sick_sentence)
                if sick_sentence:
                    break
        return sick_sentence, match_
    
    @staticmethod
    def get_negative(text, sick_sentence):
        if len(text.split()) > 0:
            text_prev = text.split()[-1]
            if text_prev in DB_from_BVBRC.NEGATIVE_vocab:
                sick_sentence = not sick_sentence
            else:
                sick_sentence = sick_sentence
        return sick_sentence

    @staticmethod
    def get_pattern(list_words, exact=False):
        regex_pattern_patho = r""
        for word in list_words:
            if exact:
                regex_pattern_patho+=r"\b{}\b|".format(word)
            else:
                regex_pattern_patho+=r"\b{}|".format(word)
        regex_pattern_patho = regex_pattern_patho[:-1]
        regex_pattern_avoid = r""
        for word in DB_from_BVBRC.VERBS_contrary:
            regex_pattern_avoid+=r"\b{}|".format(word)
        regex_pattern_avoid = regex_pattern_avoid[:-1]

        return regex_pattern_patho, regex_pattern_avoid

    @staticmethod
    def dict_to_json(out="./phenotype_patho.json"):
        from patric_vocab.phenotype_patho import PATHO_LABELS
        with open(out, "w") as outhandle:
            json.dump(PATHO_LABELS, outhandle)

    @staticmethod
    def get_diseaseexcel(excel_file):
        excel_df = pd.read_excel(excel_file, usecols=["Vocab",
                                                        "PathogenPhenotype"])
        body_data = excel_df[excel_df["PathogenPhenotype"]=="B"]["Vocab"].tolist()
        disease_data = excel_df[excel_df["PathogenPhenotype"]=="D"]["Vocab"].tolist()
        with open('disease_vocab_pre.py', 'w') as writefile:
            writefile.write('["' + ('", "'.join(disease_data)) + '"]')
        with open('body_vocab_pre.py', 'w') as writefile:
            writefile.write('["' + ('", "'.join(body_data)) + '"]')

    @staticmethod
    def mix_index(list_index):
        index_final = None
        for ind in list_index:
            if index_final is None:
                index_final = ind
            else:
                index_final = index_final.union(ind)
        return index_final

    @staticmethod
    def get_humanhost(data_file, save_file, mode="standard"):
        data = pd.read_csv(data_file)
        indices = []
        if mode == "all":
            columns_search = list(data)
        elif mode == "restringent":
            columns_search = ["Host Name", "Host Common Name", "Isolation Source"]
        else:
            columns_search = ["Host Common Name", "Phenotype", "Relevance", "Habitat", "Isolation", "Host Name"]

        prelist_indices = []

        for col in columns_search:
            try:
                indices_1 = data[col].dropna()[(data[col].dropna().str.contains("Human", case=False, na=False))&(~data[col].dropna().str.contains("Humanus", case=False, na=False))].index
                indices_2 = data[col].dropna()[(data[col].dropna().str.contains("Homo", case=False, na=False)) & (data[col].dropna().str.contains("Sapiens", case=False, na=False))].index
                prelist_indices.append(indices_1)
                prelist_indices.append(indices_2)
            except AttributeError:
                continue
        indices_final = DB_from_BVBRC.mix_index(prelist_indices)
        subset_db = data.loc[indices_final]
        subset_db.to_csv(save_file, sep=",", index=False)
        return subset_db

    @staticmethod
    def get_bacteria(data_file, save_file):
        data = pd.read_csv(data_file)
        bacteria_data = data[data["Superkingdom"]=="Bacteria"]
        bacteria_data.to_csv(save_file, sep=",", index=False)
        return bacteria_data

    @staticmethod
    def get_rid_contamination(data_file, save_file):
        data = pd.read_csv(data_file)
        data_c1 = data[~data["Genome Quality Flags"].str.contains("contamination", case=False, na=False)]
        data_c2 = data_c1[~data_c1["Host Health"].str.contains("contamination", case=False, na=False)]
        data_c2.to_csv(save_file, sep=",", index=False)
        return data_c2

    @staticmethod
    def split_data(string, delimiters):
        phrase = string.replace("]"," ").replace("[", " ").replace("(", " ")
        phrase = phrase.replace(")", " ").replace("-", " ").replace(".", " ")
        phrase = phrase.replace("_", " ").replace("?", " ").replace("'", " ")
        phrase = phrase.replace(r"-", " ").replace(":", " ")
        words = re.split(delimiters, phrase)
        return words


    def print_VOCAB_pathogen(self):
        pd.options.display.max_columns = None
        unique_health = self.db["Host Health"].unique()
        list_health = []
        list_health2 = []
        for health in unique_health:
            if not isinstance(health, str):
                continue
            health_split = DB_from_BVBRC.split_data(string=health,
                                                delimiters=' |,(|)|;|/')
            for hs1 in health_split:
                if not isinstance(hs1, str) or hs1.isnumeric() or not hs1.isalpha():
                    continue
                hs2 = hs1.lower()
                if(hs2 not in self.sw_nltk and len(hs2) > 1):
                    list_health.append(hs2)
        count_health = Counter(list_health).most_common()
        unique_health = list(set(list_health))
        data_vocab = pd.DataFrame()
        data_vocab["Vocab"] = unique_health
        data_vocab.to_excel("./sup_wo_lemma.xlsx")


    def print_VOCAB_contaminant(self, data):
        data_df = pd.read_csv(data)
        list_cols = list(data_df)
        dict_vocab = {}
        for col in list_cols:
            dict_vocab[col] = []

            try:
                lst_cont = list(data_df[col][data_df[col].str.contains(
                                    "contamin", case=False, na=False)])
            except AttributeError:
                continue

            str_cont = " ".join(lst_cont)
            wrds_cont = DB_from_BVBRC.split_data(string=str_cont,
                                                delimiters=' |,(|)|;|/')
            wrds_l_cont = []
            for wrds in wrds_cont:
                if not isinstance(wrds, str) or wrds.isnumeric() or not wrds.isalpha():
                    continue
                if wrds not in self.sw_nltk and len(wrds) > 1:
                    wrds = wrds.lower()
                    wrds_l_cont.append(wrds)
            dict_vocab[col].extend(wrds_l_cont)

        dict_unique = {}
        for col in list_cols:
            if len(dict_vocab[col]) > 0:
                dict_unique[col] = dict(Counter(dict_vocab[col]).most_common(1000))
        with open('contaminant_vocabulary.json', 'w') as fp:
            json.dump(dict_unique, fp)

    @staticmethod
    def mix_wronglemma(wronglemma, nolemma):
        wronglemma_df = pd.read_excel(wronglemma, usecols=["Vocab",
                                                        "PathogenPhenotype"])
        nolemma_df = pd.read_excel(nolemma, usecols=["Vocab"])
        #nolemma_df["PathogenPhenotype"] = "NaN"
        out_df = (nolemma_df.merge(wronglemma_df, left_on='Vocab', right_on='Vocab', how='left')
          .reindex(columns=['Vocab', 'PathogenPhenotype']))
        out_df.to_excel("./sup_nolemma_ANNOTATED.xlsx")

    def print_VOCAB_otherpatho(self, col="Other Clinical"):
        other_clinical = pd.Series(self.db[col].unique())
        split_other = other_clinical.str.split(";")
        keys = []
        for index, rows in split_other.items():
            if isinstance(rows, float):
                continue
            for element in rows:
                if len(element.split(":"))!=2:
                    continue
                key = element.split(":")[0]
                keys.append(key)
        counter_k = dict(Counter(keys).most_common())
        with open('{}_vocabularykeys.json'.format(col), 'w') as fp:
            json.dump(counter_k, fp)
        values = {}
        for k, v in counter_k.items():
            if k in DB_from_BVBRC.Check[col]:
                values[k] = []
        for index, rows in split_other.items():
            if isinstance(rows, float):
                continue
            for element in rows:
                if len(element.split(":"))!=2:
                    continue
                key,val = element.split(":")
                if key in DB_from_BVBRC.Check[col] and val not in values[key]:
                    values[key].append(element.split(":")[1])
        with open('{}_vocabularykeyvalues.json'.format(col), 'w') as fp:
            json.dump(values, fp)

    def select_pathometadata(self, restringent_lvl):
        # Select from all, patho, patho_dev, patho_disease, disease
        if restringent_lvl == "all":
            vocab_entries = ["Pathogen", "Possible Pathogen",
                "Disease Caused By Pathogen",
                "Disease maybe caused by pathogen"]
        elif restringent_lvl == "demo":
            vocab_entries = ["DEMO_nonpatho"]
        elif restringent_lvl == "infection":
            vocab_entries = ["Pathogen", "Possible Pathogen",
                "Disease Caused By Pathogen"]
        elif restringent_lvl == "pathogen related":
            vocab_entries = ["Pathogen", "Possible Pathogen"]
        elif restringent_lvl == "pathogen":
            vocab_entries = ["Pathogen"]
        elif restringent_lvl == "unhealthy":
            vocab_entries = ["Pathogen", "Possible Pathogen",
                "Disease Caused By Pathogen",
                "Disease maybe caused by pathogen", "Unhealthy"]
        elif restringent_lvl == "nonpathogen":
            vocab_entries = ["Non Pathogen"]
        elif restringent_lvl == "microbiome":
            vocab_entries = ["Flora Microbiome"]
        elif restringent_lvl == "probiotic":
            vocab_entries = ["Probiotic"]
        elif restringent_lvl == "extremophile":
            vocab_entries = ["Extreme Conditions"]
        else:
            raise KeyError("{} is not a key for the vocabulary metadata".format(
                                restringent_lvl))
        if self.metadata_patho is None:
            raise AttributeError("Metadata not loaded")
        vocabulary = {"vocabulary":[], "abbreviation":[]}
        for entries in vocab_entries:
            vocabulary["vocabulary"].extend(self.metadata_patho[entries]["vocabulary"])
            vocabulary["abbreviation"].extend(self.metadata_patho[entries]["abbreviation"])
        return vocabulary


    @staticmethod
    def filter_contain(db_excluded, col, vocab, stopwords, center_vocab=None,
                            type_vocab="vocabulary"):
        pre_index = []
        neg_voc = ["non{}", "non-{}", "non {}", "no{}", "no-{}", "no {}",
                        "{} vaccine", "un{}"]
        if type_vocab == "abbreviation":
            add_neg_voc = []
            for neg in neg_voc:
                add_neg_voc.extend(neg.capitalize())
        if type_vocab == "abbreviation":
            abbrev = True
            _voc = vocab
            ind = db_excluded.index[db_excluded[col].str.contains(r"\b"+_voc+r"\b",
                                                                   case=abbrev, na=False)]
            db_excluded = db_excluded.loc[ind]
            for neg in neg_voc:
                neg = neg.format(_voc)
                ind = db_excluded.index[~db_excluded[col].str.contains(neg,
                        case=abbrev, na=False)]
                db_excluded = db_excluded.loc[ind]
            final_ind = db_excluded.index
            pre_index.append(final_ind)
        else:
            abbrev = False
            for _voc in vocab.split(" "):
                ind = db_excluded.index[db_excluded[col].str.contains(r"\b"+_voc,
                        case=abbrev, na=False)]
                db_excluded = db_excluded.loc[ind]
                for neg in neg_voc:
                    neg = neg.format(_voc)
                    ind = db_excluded.index[~db_excluded[col].str.contains(neg,
                            case=abbrev, na=False)]
                    db_excluded = db_excluded.loc[ind]
                final_ind = db_excluded.index
                pre_index.append(final_ind)
        index_contain = list(set.intersection(*map(set, pre_index)))
        reason = db_excluded.loc[index_contain, col]
        if center_vocab is not None:
            index_contain, reason = DB_from_BVBRC.check_organizations(
                                            index_contain=index_contain,
                                            reason=reason, word=vocab,
                                            stopwords=stopwords,
                                            center_vocab=center_vocab)
        return index_contain, reason

    @staticmethod
    def check_organizations(index_contain, reason, word, stopwords,
                            center_vocab):
        word = word.capitalize()
        reason_split = reason.str.split(" ")
        delete_index = []
        for index, row in reason_split.items():
            sub_index = next((n for n in range(len(row)) if word in row[n]), None)
            suspicious_sentence = ""
            if sub_index is not None:
                while True:
                    if (sub_index) >= 0:
                        if row[sub_index] in stopwords or row[sub_index] == "":
                            sub_index-=1
                            continue
                        if row[sub_index][0].isupper():
                            suspicious_sentence += str(row[sub_index])
                            suspicious_sentence += " "
                            sub_index-=1
                        else:
                            break
                    else:
                        break
                while True:
                    if (sub_index) >= 0:
                        if row[sub_index] in stopwords:
                            sub_index+=1
                            continue
                        if row[sub_index][0].isupper():
                            suspicious_sentence += str(row[sub_index])
                            suspicious_sentence += " "
                            sub_index+=1
                        else:
                            break
                    else:
                        break
            for center in center_vocab:
                if center in suspicious_sentence:
                    delete_index.append(index)
                    break
        index_contain = pd.Series(index_contain, index=index_contain).drop(index=delete_index).tolist()
        reason.drop(index=delete_index, inplace=True)
        return index_contain, reason

    def select_nlp_containing(self, vocabulary, columns):
        df_selected = pd.DataFrame()
        for col in columns:
            if "Comments" in col:
                if not df_selected.empty:
                    index_selected = df_selected["Index"].tolist()
                    db_excluded = self.db.drop(index_selected, axis=0)
                else:
                    db_excluded = self.db
                comment_text = db_excluded[col].unique()
                for type_vocab, vocab in vocabulary.items():
                    if type_vocab == "abbreviation":
                        exact=True
                    else:
                        exact=False
                    sick_pattern, avoid_pattern = DB_from_BVBRC.get_pattern(list_words=vocab, exact=exact)
                    for text in tqdm(comment_text):
                        bool_patho, _ = DB_from_BVBRC.analyze_sentence_pathogen(text=text, 
                                                                            patho_regex=sick_pattern,
                                                                            avoid_regex=avoid_pattern,
                                                                            exact=exact)
                        if bool_patho:
                            cols_interest = self.db[self.db[col]==text]
                            ind_patho = cols_interest.index
                            reasons_sel = cols_interest[col]
                            pre_selected = pd.DataFrame({"Index":ind_patho,
                                                    "Reasons": reasons_sel})
                            df_selected = pd.concat([df_selected,pre_selected])
                            df_selected = (df_selected.groupby("Index").agg({"Reasons": " // ".join}).reset_index())
                        gc.collect()
            else:
                for type_vocab, vocab in tqdm(vocabulary.items()):
                    for words in tqdm(vocab):
                        if not df_selected.empty:
                            index_selected = df_selected["Index"].tolist()
                            db_excluded = self.db.drop(index_selected, axis=0)
                        else:
                            db_excluded = self.db
                        #print("{}".format(words))
                        db_excluded_copy = db_excluded.copy()
                        index_contain, reasons_selected = DB_from_BVBRC.filter_contain(
                                                    db_excluded=db_excluded_copy,
                                                    col=col, vocab=words,
                                                    stopwords=self.sw_nltk,
                                                    center_vocab=self.Center_synonyms,
                                                    type_vocab=type_vocab)
                        pre_selected = pd.DataFrame({"Index":index_contain,
                                                    "Reasons": reasons_selected})
                        df_selected = pd.concat([df_selected,pre_selected])
                        df_selected = (df_selected.groupby("Index").agg({"Reasons": " // ".join}).reset_index())
                        gc.collect()
        df_selected.sort_values(by=['Index'], inplace=True)
        index_selected = df_selected["Index"].tolist()
        reasons_selected = df_selected["Reasons"].tolist()
        subset_db = self.db.loc[index_selected]
        subset_db["Reason Phenotype"] = reasons_selected
        return subset_db        

    def select_containing(self, vocabulary, columns):
        df_selected = pd.DataFrame()
        for col in columns:
            for type_vocab, vocab in vocabulary.items():
                for words in tqdm(vocab):
                    if not df_selected.empty:
                        index_selected = df_selected["Index"].tolist()
                        db_excluded = self.db.drop(index_selected, axis=0)
                    else:
                        db_excluded = self.db
                    db_excluded_copy = db_excluded.copy()
                    index_contain, reasons_selected = DB_from_BVBRC.filter_contain(
                                                db_excluded=db_excluded_copy,
                                                col=col, vocab=words,
                                                stopwords=self.sw_nltk,
                                                center_vocab=self.Center_synonyms,
                                                type_vocab=type_vocab)
                    pre_selected = pd.DataFrame({"Index":index_contain,
                                                "Reasons": reasons_selected})
                    df_selected = pd.concat([df_selected,pre_selected])
                    df_selected = (df_selected.groupby("Index").agg({"Reasons": " // ".join}).reset_index())
                    gc.collect()
        df_selected.sort_values(by=['Index'], inplace=True)
        index_selected = df_selected["Index"].tolist()
        reasons_selected = df_selected["Reasons"].tolist()
        subset_db = self.db.loc[index_selected]
        subset_db["Reason Phenotype"] = reasons_selected
        return subset_db

    def remove_species(self, columns):
        new_columns = []
        species_col = self.db["Species"].fillna("").to_numpy()
        len_data = len(species_col)
        for col in columns:
            new_col = col+"_clean"
            new_columns.append(new_col)
            numpy_col = self.db[col].fillna("").to_numpy()
            new_nmp = []
            for indx in range(len_data):
                if species_col[indx] == "":
                    replaced_str = numpy_col[indx]
                    for word in str(self.db.loc[indx]["Genome Name"]).split(" "):
                        replaced_str = replaced_str.replace(word, "")
                else:
                    replaced_str = str(numpy_col[indx]).replace(str(species_col[indx]), "")
                new_nmp.append(replaced_str)
            self.db[new_col] = new_nmp
            for del_word in DB_from_BVBRC.Unsure_patho:
                self.db[new_col].str.replace(del_word, " ")
        return new_columns

    def select_cols(self, columns_mode):
        if columns_mode == "health":
            columns = ["Isolation Source","Host Health", "Other Clinical"]
        elif columns_mode == "healthdescription":
            columns = ["Isolation Source","Host Health", "Other Clinical",
                    "Comments", "Additional Metadata"]
        else:
            raise KeyError("The columns mode {} is not a mode".format(
                                                                columns_mode))
        cols_clean = self.remove_species(columns=columns)
        return cols_clean

    def exclude_nlp_pathogen(self, db, patho_voc, columns):
        index_selected = []
        for col in columns:
            if "Comments" in col:
                if len(index_selected) > 0:
                    #index_selected = df_selected["Index"].tolist()
                    db_excluded = db.loc[~db.index.isin(index_selected)]
                else:
                    db_excluded = db
                comment_text = db_excluded[col].unique()
                for type_vocab, vocab in patho_voc.items():
                    if type_vocab == "abbreviation":
                        exact=True
                    else:
                        exact=False
                    sick_pattern, avoid_pattern = DB_from_BVBRC.get_pattern(list_words=vocab, exact=exact)
                    for text in tqdm(comment_text):
                        bool_patho, shit = DB_from_BVBRC.analyze_sentence_pathogen(text=text, 
                                                                            patho_regex=sick_pattern,
                                                                            avoid_regex=avoid_pattern,
                                                                            exact=exact)
                        if bool_patho:
                            cols_interest = db[db[col]==text]
                            ind_patho = cols_interest.index
                            index_selected.extend(ind_patho)
                            index_selected = list(set(index_selected))
                        gc.collect()
            else:
                for type_vocab, vocab in patho_voc.items():
                    count = 0
                    for words in tqdm(vocab):
                        db_excluded = db.drop(index_selected, axis=0)
                        #print("{}".format(words), len(vocab), count)
                        db_excluded_copy = db_excluded.copy()
                        index_contain, _ = DB_from_BVBRC.filter_contain(
                                                db_excluded=db_excluded_copy, col=col,
                                                vocab=words, type_vocab=type_vocab,
                                                stopwords=None, center_vocab=None)

                        index_selected.extend(index_contain)
                        index_selected = list(set(index_selected))
                        #print(len(index_selected))
                        count += 1
                        gc.collect()
        return db.loc[~db.index.isin(index_selected)]



    def get_pathogen(self, out_file, restringent_lvl="pathogen", columns_mode="health"):
        columns_analyze = self.select_cols(columns_mode=columns_mode)
        vocabulary = self.select_pathometadata(restringent_lvl=restringent_lvl)
        pathogen_db = self.select_nlp_containing(vocabulary=vocabulary,
                                                   columns=columns_analyze)
        #pathogen_db = self.select_containing(vocabulary=vocabulary,
         #                                       columns=columns_analyze)
        pathogen_db.to_csv(
            "{}/BVBRC_bacteria_pathogens-selNLP_{}-col_{}.csv".format(out_file,
                                                restringent_lvl, columns_mode))
  #      pathogen_db.to_csv(out_file)


    def remove_sp_probiotic(self, db):
        species_remove = ["Streptococcus sp.", "Klebsiella pneumoniae", "Clostridium sp.",
            "Pseudomonas sp.", "Listeria innocua", "Klebsiella sp.", "Salmonella enterica",
            "Staphylococcus aureus", "Acinetobacter baumannii", "Escherichia coli",
            "Listeria monocytogenes"]
        index_remove = []
        for sp in species_remove:
            ind_rm = db[db["Species"].str.contains(sp, na=False)].index
            index_remove.extend(ind_rm)
        db = db.loc[~db.index.isin(index_remove)]
        return db

    def get_nonpatho(self, out_file, nonpatho_type="nonpathogen", columns_mode="health"):
        columns_analyze = self.select_cols(columns_mode=columns_mode)

        vocabulary_nonpatho = self.select_pathometadata(
                                                restringent_lvl=nonpatho_type)
        vocabulary_patho = self.select_pathometadata(
                                                restringent_lvl="unhealthy")
        nonpathogen_db = self.select_containing(vocabulary=vocabulary_nonpatho,
                                                columns=columns_analyze)
        clean_nonpatho_db = self.exclude_nlp_pathogen(db=nonpathogen_db,
                                                patho_voc=vocabulary_patho,
                                                columns=columns_analyze)
        if nonpatho_type == "probiotic":
            clean_nonpatho_db = self.remove_sp_probiotic(clean_nonpatho_db)
 #       clean_nonpatho_db.to_csv(out_file)
        clean_nonpatho_db.to_csv(
            "{}/BVBRC_bacteria_nonpathogen-selNLP_{}-col_{}.csv".format(out_file,
                                                nonpatho_type, columns_mode))

def get_arguments():
    parser = argparse.ArgumentParser(description='Clean Patric and create different datasets')
    parser.add_argument('-i','--in_metadataFile', help='Input metadatafile', required=True)
    parser.add_argument('-f','--format_in', help='Data format metadata', required=True,
                                        choices=["original", "bacteria_only", "bacteria_human_only",
                                                 "bacteria_human_clean_only"])
    parser.add_argument('-s', '--subset_out', help="Where the subsets go")
    parser.add_argument('-j', '--save_jsonvocab', required=True, help="Save json vocab")
    parser.add_argument('-t', '--type_vocab', required=True, choices=["all", "infection", "pathogen related",
                                                                    "pathogen", "unhealthy", "nonpathogen", "microbiome",
                                                                      "probiotic", "extremophile"])
    parser.add_argument('-c', '--columns_check', required=True, choices=["health","healthdescription"])
    parser.add_argument('--from_date', help="From Date", default=False, type=datetime.date.fromisoformat)
    parser.add_argument('-o', '--out_file', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arguments()
    ## TODO: Be careful with study names (like Virulence Study of Humans)
    ## Be careful with gallstones diseases
#    DB_from_BVBRC.get_diseaseexcel(excel_file="sup_nolemma_ANNOTATED_done.xlsx")
    instance = DB_from_BVBRC()
    if not os.path.isfile(args.save_jsonvocab):
        DB_from_BVBRC.dict_to_json(out=args.save_jsonvocab)
    if args.format_in == "original":
        orig_file = args.in_metadataFile
        bacteria_file = "{}/BVBRC_bacteria_genome.csv".format(args.subset_out)
        bacteria_clean_file = "{}/BVBRC_bacteria_clean_genome.csv".format(args.subset_out)
        bacteria_clean_hh_file = "{}/BVBRC_bacteria_clean_hh_genome.csv".format(args.subset_out)
        DB_from_BVBRC.get_bacteria(data_file=orig_file, save_file=bacteria_file)
        DB_from_BVBRC.get_rid_contamination(data_file=bacteria_file,
                                            save_file=bacteria_clean_file)
        DB_from_BVBRC.get_humanhost(data_file=bacteria_clean_file, save_file=bacteria_clean_hh_file, mode="restringent")
    elif args.format_in == "bacteria_only":
        bacteria_file = args.in_metadataFile
        bacteria_clean_file = "{}/BVBRC_bacteria_clean_genome.csv".format(args.subset_out)
        bacteria_clean_hh_file = "{}/BVBRC_bacteria_clean_hh_genome.csv".format(args.subset_out)
        DB_from_BVBRC.get_rid_contamination(data_file=bacteria_file,
                                            save_file=bacteria_clean_file)
        DB_from_BVBRC.get_humanhost(data_file=bacteria_clean_file, save_file=bacteria_clean_hh_file, mode="restringent")
    elif args.format_in == "bacteria_clean_only":
        bacteria_clean_file = args.in_metadataFile
        bacteria_clean_hh_file = "{}/BVBRC_bacteria_clean_genome.csv".format(args.subset_out)
        DB_from_BVBRC.get_humanhost(data_file=bacteria_clean_file, save_file=bacteria_clean_hh_file, mode="restringent")
    else:
        bacteria_hh_clean_file = args.in_metadataFile

    if args.type_vocab == "probiotic" or args.type_vocab == "extremophile":
        instance.set_db(db=bacteria_clean_file)
    else:
        instance.set_db(db=bacteria_clean_hh_file)
    if args.from_date:
        instance.select_fromdate(start_date=args.from_date)
    instance.set_metadata_patho(path=args.save_jsonvocab)
    
    if args.type_vocab in ["all", "infection", "pathogen related", "pathogen", "unhealthy"]:
        instance.get_pathogen(out_file=args.out_file, restringent_lvl=args.type_vocab, columns_mode=args.columns_check)    
    else:
        instance.get_nonpatho(out_file=args.out_file, nonpatho_type=args.type_vocab, columns_mode=args.columns_check)
    #instance.get_pathogen(restringent_lvl="all", columns_mode="healthdescription")


    #instance.print_VOCAB_nonpatho(data=bacteria_file, keyword="human",
    #                                        col_sel="Comments")
    #instance.print_otherpatho(col="Other Clinical")
    #instance.print_otherpatho(col="Additional Metadata")
