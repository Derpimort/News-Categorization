import os
import math
from multiprocessing import Process
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

start_time=time.time()

tqdm.pandas()
num_procs=10
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
if not os.path.isfile(DATA_DIR+"latlong.csv"):
    with open(DATA_DIR+"latlong.csv", "w") as f:
        f.write("article_id,lat,long,state_code\n")


def get_coords(locs, all_loc, num):
    with open(DATA_DIR+"latlong_%d.csv"%num, "a") as f:
        for index, loc in locs['locs'].iteritems():
            for i in all_loc[all_loc['name'].isin(loc)].itertuples(index=False, name=None):
                f.write("%d,%f,%f,%d\n"%(index,i[1],i[2],i[-1]))
            

if __name__=="__main__":
    locs = pd.read_pickle(DATA_DIR+"locations_newfull.pkl")

    print("Queueing items")
    per_chunk = math.ceil(len(locs)/num_procs)+1 
    procs = []
    for i in range(num_procs):
        proc = Process(target=get_coords, args=(locs[i*per_chunk:(i+1)*per_chunk], all_locs, i+1,))
        procs.append(proc)
        proc.start()
    print("Processing")
    for proc in procs:
        proc.join()
    print("DONE! in %.5f seconds"%(time.time()-start_time))


        
