import numpy as np
import pandas as pd
from pandarallel import pandarallel
from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 169

DATA_DIR="data/cleaned/"
pandarallel.initialize(nb_workers=10, progress_bar=True, use_memory_fs=False)

def get_lang(row):
    ret_l=[row['id']]
    for ind in ['title', 'description', 'long_description']: 
        try:
            ret_l.append(detect(row[ind]))
        except Exception as e:
            ret_l.append('unknown')
    return ret_l

if __name__=="__main__":
    df_train=pd.read_pickle(DATA_DIR+"train_cleaned.pkl")
    lang_df=df_train.parallel_apply(get_lang, axis=1, result_type='expand')
    lang_df.columns=['id', 'title_lang', 'description_lang', 'long_description_lang']

    lang_df=lang_df.set_index('id')
    lang_df.to_csv(DATA_DIR+"langs.csv")
    lang_df.to_pickle(DATA_DIR+"langs.pkl")
