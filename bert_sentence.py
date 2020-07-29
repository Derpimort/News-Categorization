import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import argparse

DATA_DIR="data/cleaned/"

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--target", default="eng", help="output suffix")
    parser.add_argument("--dir", default=DATA_DIR, help="source data directory")
    parser.add_argument("--df", default="train_cleaned.pkl", help="pickle of preprocessed dataframe")
    parser.add_argument("--df_lang", default="langs.csv", help="csv of languages df")
    parser.add_argument("--model_dir", default="data/models/", help="Directory containing sentencetransformer modeli, or could be model name")
    args=parser.parse_args()
    DATA_DIR=args.dir
    df_langs=pd.read_csv(DATA_DIR+args.df_lang)
    df_langs=df_langs.set_index('id')
    df_train=pd.read_pickle(DATA_DIR+args.df)
    df_train=df_train.set_index('id')

    cols=['title', 'description', 'long_description']
    for col in cols:
        df_train[col][df_langs[col+"_lang"]!='en']=None

    df_train['long_description'].fillna(df_train['description'], inplace=True)
    df_train['long_description'].fillna(df_train['title'], inplace=True)
    df_train=df_train[df_train['long_description'].notna()]
    model=SentenceTransformer(args.model_dir)
    embeddings=model.encode(df_train['long_description'].values, batch_size=256)
    print(len(embeddings)) 
    df_train['long_description'].to_csv(DATA_DIR+"bert_index_%s.csv"%args.target, header=True)
    np.save(DATA_DIR+"bert_embeddings_%s.npy"%args.target, embeddings)

