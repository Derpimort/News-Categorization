import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

DATA_DIR="data/cleaned/"

df=pd.read_pickle(DATA_DIR+"train_cleaned.pkl")
df=df.set_index('id')
model=SentenceTransformer('data/models/')
embeddings=model.encode(df['long_description'].dropna().values, batch_size=256)
print(len(embeddings))
np.save(DATA_DIR+"bert_embeddings.npy", embeddings)
