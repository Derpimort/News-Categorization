import os
import re
import nltk
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from urllib.parse import urlparse

tqdm.pandas()

nlp=spacy.load('en_core_web_lg')
DATA_DIR="data/cleaned/"
loc_labels=['LOC', 'GPE']

def get_locs(docents):
    res=[]
    for entity in docents:
        if entity.label_ in loc_labels:
            res.append(entity.text)
    if not res:
        return [''], ''
    else:
        res, counts=np.unique(res, return_counts=True)
        return res.tolist(), res[np.argmax(counts)]

def loc_destroyer(row):
    sentence=row['long_description']
    try:
        if pd.isna(sentence):
            #tags=url_parse(row['link'])
            tags=['']
            toptag=''
        else:
            sentence=re.sub("[\\\\\'\"]", "", sentence)
            tokenized=nlp(sentence)
            tags, toptag=get_locs(tokenized.ents)

    except Exception as e:
        tags=['']
        toptag=''
    return row.name, tags, toptag


if __name__=="__main__":

    df_langs=pd.read_csv(DATA_DIR+"langs.csv")
    df_langs=df_langs.set_index('id')
    df_train=pd.read_pickle(DATA_DIR+"train_cleaned.pkl")
    df_train=df_train.set_index('id')

    cols=['title', 'description', 'long_description']
    for col in cols:
        df_train[col][df_langs[col+"_lang"]!='en']=None

    df_train['long_description'].fillna(df_train['description'], inplace=True)
    df_train['long_description'].fillna(df_train['title'], inplace=True)

    result=df_train.progress_apply(loc_destroyer, axis=1, result_type='expand')
    result.columns=['id', 'locs', 'top_loc']
    result=result.set_index('id')
    result.to_csv(DATA_DIR+"locations_%d.csv")
    result.to_pickle(DATA_DIR+"locations_%d.pkl")


