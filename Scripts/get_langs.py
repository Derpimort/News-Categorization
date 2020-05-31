import numpy as np
import pandas as pd
from langdetect import detect
from tqdm import tqdm
from langdetect import DetectorFactory
DetectorFactory.seed = 169

DATA_DIR="data/cleaned/"

def get_lang(row):
    try:
        lang=detect(row)
    except Exception as e:
        lang='unknown'
    return lang

if __name__=="__main__":
    df_train=pd.read_pickle(DATA_DIR+"train_cleaned.pkl")
    df_train=df_train.set_index('id')

    lang_list=[]

    for index, row in tqdm(df_train.iterrows()):
        lang_list.append((index, 
            get_lang(row['title']), 
            get_lang(row['description']),
            get_lang(row['long_description'])))

    lang_df=pd.DataFrame(lang_list, 
        columns=['id', 'title_lang', 'description_lang', 'long_description_lang'])

    lang_df=lang_df.set_index('id')
    lang_df.to_csv(DATA_DIR+"langs.csv")
    lang_df.to_pickle(DATA_DIR+"langs.pkl")
