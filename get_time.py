import os
import re
import sys
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm

DATA_DIR="data/cleaned/"


PUBLISHED_L=[("meta", {"property": re.compile(".*published.*")}),
             ("meta", {"name": re.compile(".*published.*")}),
             ("meta", {"property": re.compile(".*created.*")}),
             ("meta", {"name": re.compile(".*created.*")}),
             ("meta", {"property": re.compile(".*date.*")}),
             ("meta", {"name": re.compile(".*date.*")}),
             ("meta", {"property": re.compile(".*time.*")}),
             ("meta", {"name": re.compile(".*time.*")})]

def iterTillHit(soup, arglist, target=None):
        for arg in arglist:
            cont=soup.find(*arg)
            if cont:
                if not target:
                    return cont
                elif cont.text:
                    return cont.text
                else:
                    return cont[target]
        else:
            return None

def get_time(file):
    with open(file,"r") as f:
        cont=f.readline()
        if not cont.startswith("Error"):
            return iterTillHit(BeautifulSoup(f.read(), 'lxml'), PUBLISHED_L, 'content')
    return None

if __name__=="__main__":

    df_train = pd.read_pickle(DATA_DIR+"train_cleaned.pkl")
    df_train['date']=None
    df=df_train[['id', 'date']]
    del df_train
    df=df.set_index('id')
    for dirname, _, filenames in os.walk(sys.argv[1]):
        for filename in tqdm(filenames):
            try:
                ntime=get_time(os.path.join(dirname,filename))
                if ntime:
                    df.loc[int(filename.split(".")[0])]['date']=ntime
            except Exception as e:
                pass
            
    df.dropna().to_pickle(DATA_DIR+"time_full.pkl")
