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


#checks=["'", '"']
checks=['"']
def unbalanced(s):
    for check in checks:
        if (s.count(check)-s.count(f'\\{check[0]}'))%2!=0:
            return True
    return False
def lineCleaner(line):
    line=line.strip("\n")
    splitl=line.split("|")
    slen=len(splitl)
    ids=re.findall(r'\|([0-9]+)$',line)
    if slen==5:
        return splitl
    else:
        try:
            lindex=0
            res=[]
            while(lindex<slen):
                if unbalanced(splitl[lindex]) and lindex<slen-1:
                    rindex=lindex+1
                    while not unbalanced(splitl[rindex]) and rindex<slen-2:
                        rindex+=1
                    res+=[" ".join(splitl[lindex:rindex+1])]
                    lindex=rindex
                else:
                    res+=[splitl[lindex]]
                lindex+=1
#             if len(res)==6:
#                 res[3]=res[3]+res[4]
#                 res[]
            if len(res)>5:
                res[3]=" ".join(res[3:])
                res=res[:4]
            if len(res)==4:
                res+=[ids[0].strip("|")]
#             if len(res)==3:
#                 long_desc=re.search(r'\|([^\|]*)\|[0-9]*$', line)
            if len(res)!=5:
                res=[]
                url=re.search(r'\|http.*?\|', line)
                if url is not None:
                    res+=[line[:url.start()]]
                    res+=[line[url.start():url.end()]]
                else:
                    res+=[None, None]
                long_desc=re.search(r'\|([^\|]*)\|[0-9]*$', line)
                if long_desc is not None:
                    res+=[line[url.end():long_desc.start()]]
                    res+=[line[long_desc.start():long_desc.end()]]
                else:
                    res+=[None, None]
                res+=[ids[0]]
                for i in range(5):
                    if res[i] is not None:
                        res[i]=res[i].strip("|")
            return res
        except Exception as e:
            return [None]*5
#             res=[ids[0].strip("|")]
#             url=re.search(r'\|http.*?\|', line).start()
            
#             long_desc=re.search(r'\|([^\|]*)\|[0-9]*$', line)

badf=open(DATA_DIR+"bad_rows.csv", "w", newline='')
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
            try:
                row_new=lineCleaner("|".join(row))
                int(row_new[-1])
            except Exception as e:
                bad_count+=1
                bad_csv_writer.writerow(row)
                row_new=[None, None, None, None, row[-1]]
            finally:
                row=row_new
        csv_writer.writerow(row)
        

output.seek(0)
df_first = pd.read_csv(output, sep=u'\u0001')
print(df_first.head())
DATA_DIR+="cleaned/"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
df_first.to_csv(DATA_DIR+"train_cleaned.csv", sep=u'\u0001', index=False)
df_first.to_pickle(DATA_DIR+"train_cleaned.pkl")
df_first['link'].to_csv(DATA_DIR+"links.csv")
df_first['link'].to_pickle(DATA_DIR+"links.pkl")
print("Bad files", bad_count)
