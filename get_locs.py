import os
import re
import queue
from queue import Queue
from threading import Thread
import csv
import nltk
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from urllib.parse import urlparse

tqdm.pandas()

name=input("Enter output suffix: ")
nlp=spacy.load('en_core_web_lg')
DATA_DIR="/data/techgig/cleaned/"
loc_labels=['LOC', 'GPE']
all_locs_cols = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature class",
    "feature code",
    "country code",
    "cc2",
    "admin1 code",
    "admin2 code",
    "admin3 code",
    "admin4 code",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modification date"]

all_locs = pd.read_csv(DATA_DIR+"../IN.txt", delimiter="\t", names=all_locs_cols, index_col=0)
all_locs = all_locs[["name","latitude", "longitude", "admin1 code"]]
#states = pd.read_csv(DATA_DIR+"../admin1CodesASCII.txt", delimiter="\t", names=['state_name', 'state_name2', 'pincode'])
all_locs['name'] = all_locs['name'].str.lower()
all_locs['state_name'] = all_locs['admin1 code'].fillna(0.0).round(0).astype(int)
#all_locs['admin1 code'] = all_locs['admin1 code'].apply(lambda x: 'IN.'+str(round(x)))
#all_locs = all_locs.join(states['state_name'], on='admin1 code')

queue1 = Queue(maxsize=0)

if not os.path.isfile(DATA_DIR+"latlong.csv"):
    with open(DATA_DIR+"latlong.csv", "w") as f:
        f.write("article_id,lat,long,state_code\n")

class CoordWorker(Thread):
    def __init__(self, queue, num, df):
        Thread.__init__(self)
        self.queue = queue
        self.f = open(DATA_DIR+"latlong_%d.csv"%num, "a")
        self.df = df

    def close(self):
        self.f.close()
    
    def run(self):
        while True:
            try:
                article_id, tags = self.queue.get()
                for i in self.df[self.df['name'].isin(tags)].itertuples(index=False, name=None):
                    self.f.write("%d,%f,%f,%d\n"%(article_id,i[1],i[2],i[-1]))
                self.queue.task_done()
            except queue.Empty:
                pass

def write_coords(article_id, tags):
    for i in all_locs[all_locs['name'].isin(tags)].itertuples(index=False, name=None):
        coords.write("%d,%f,%f,%d\n"%(article_id,i[1],i[2],i[-1]))
    #coords_writer.writerows(((article_id,i[1],i[2],int(i[-1])) for i in all_locs[all_locs['name'].isin(tags)].itertuples(index=False, name=None)))

    
def get_states(tags):
    res=all_locs[all_locs['name'].isin(tags)]['state_name'].dropna().unique().tolist()
    return res

def get_locs(docents):
    res=[]
    for entity in docents:
        if entity.label_ in loc_labels:
            res.append(re.sub("[^a-z, ]","",entity.text.lower()))
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
            #sentence=re.sub("[\\\\\'\"]", "", sentence)
            tokenized=nlp(sentence)
            tags, toptag=get_locs(tokenized.ents)
            queue1.put((row.name, tags))
            #write_coords(row.name, tags)
            #states=get_states(tags)

    except Exception as e:
        print(e)
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
    print(df_train.head())
    workers = []
    print("Initializing workers")
    for i in range(10):
        worker = CoordWorker(queue1, i+1, all_locs)
        worker.daemon = True
        worker.start()
        workers.append(worker)


    result=df_train[:10000].progress_apply(loc_destroyer, axis=1, result_type='expand')
    queue1.join()
    for worker in workers:
        worker.close()
    result.columns=['id', 'locs', 'top_loc']
    result=result.set_index('id')
    result.to_csv(DATA_DIR+"locations_%s.csv"%name)
    result.to_pickle(DATA_DIR+"locations_%s.pkl"%name)
    coords.close()

