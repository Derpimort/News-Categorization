import os
import numpy as np
import pandas as pd
import re
from io import StringIO
import csv
import pandas as pd
from tqdm.auto import tqdm

DATA_DIR="data/"

with open(DATA_DIR+"Train_data.csv", "r") as f:
        print(f.readline().strip("\n"))

badf=open(DATA_DIR+"bad_rows_simple.csv", "w", newline='')
bad_csv_writer=csv.writer(badf, delimiter="|", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
output = StringIO()
csv_writer = csv.writer(output, delimiter=u'\u0001')

bad_count=0

with open(DATA_DIR+"Train_data.csv", "r") as f:
    header=f.readline().strip("\n")
    csv_writer.writerow(header.split("|"))
    rc=csv.reader(f, delimiter="|", quotechar='"')
    for row in tqdm(rc, total=1411103):
        if len(row)!=5:
            bad_count+=1
            bad_csv_writer.writerow(row)
        else:
            csv_writer.writerow(row)
        

output.seek(0)
df_first = pd.read_csv(output, sep=u'\u0001')
print(df_first.head())

df_first.to_csv(DATA_DIR+"train_cleaned_simple.csv", sep=u'\u0001', index=False)
df_first.to_pickle(DATA_DIR+"train_cleaned_simple.pkl")
print("Bad files", bad_count)
