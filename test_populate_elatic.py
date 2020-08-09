import os
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from tqdm import tqdm

DATA_DIR="data/cleaned/"

def transform_data(df, df_locs):
    es_data_keys = ('text', 'link', 'timestamp', 'points', 'states')
    for article_id, keywords, url, date in tqdm(df.itertuples(), total=len(df)):
        if not pd.isna(url):
            locs=None
            states=None
            try:
                temp = df_locs.loc[article_id]
                locs = temp[['long','lat']].values.tolist()
                states = np.unique(temp.dropna()['state_name']).tolist()
            except KeyError as e:
                pass
            es_data_values = (
                keywords, 
                url,
                date if not pd.isna(date) else "", 
                locs, 
                states if states else "")

            yield article_id, dict(zip(es_data_keys, es_data_values))

if __name__=="__main__":
    
    df_locs = pd.read_csv(DATA_DIR+"latlong.csv")
    state_names = pd.read_csv(DATA_DIR+"../admin1CodesASCII.txt", delimiter="\t", names=['state_name', 'state_name2', 'pincode'])
    df_locs['state_code'] = df_locs['state_code'].apply(lambda x: 'IN.%.2d'%x)
    df_locs = df_locs.join(state_names['state_name'], on='state_code')
   
    df_sub = pd.read_pickle(DATA_DIR+"submission_full.pkl")
    df_time = pd.read_pickle(DATA_DIR+"time_full.pkl")
    df_links = pd.read_pickle(DATA_DIR+"links.pkl")

    df = df_sub.join(df_links).join(df_time)

    del df_sub, df_time, df_links
    
    es = Elasticsearch(hosts=[{'host':'localhost', 'port':'9200'}])
    
    data_iter = ({
                "_index":       "article",
                "_type":        "_doc",
                "_id":          idx,
                "_source":      es_article,
                } for idx, es_article in transform_data(df, df_locs))

    helpers.bulk(es, data_iter)
    
    
     
