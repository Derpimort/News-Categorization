import os
import re
import nltk
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
from preprocess import lemma_pp

tqdm.pandas()

startnum, subsetnum=[int(i) for i in input("Enter subsetNum: ").split(" ")]
model=SentenceTransformer('data/models/')
nlp=spacy.load('en_core_web_sm')
DATA_DIR="data/cleaned/"
considered_tags = set(['NN', 'NNS', 'NNP', 'NNPS', 'JJ'])
propnoun_tags=set(['NNP', 'NNPS'])

def url_parse(sent):
    sent=urlparse(sent).path
    sent=re.sub(r' .{2} '," "," ".join(lemma_pp(sent)))
    return sent.split(" ")

# From https://github.com/swisscom/ai-research-keyphrase-extraction
def get_topk(doc, X, candidates, N=13, beta=0.6, alias_threshold=0.8):
    N=min(N, len(candidates))
    doc_sim=cosine_similarity(X, doc)
    doc_sim_norm = doc_sim/np.max(doc_sim)
    doc_sim_norm = 0.5 + (doc_sim_norm - np.average(doc_sim_norm)) / np.std(doc_sim_norm)
    sim_between = cosine_similarity(X)
    np.fill_diagonal(sim_between, np.NaN)
    sim_between_norm = sim_between/np.nanmax(sim_between, axis=0)
    sim_between_norm = 0.5 + (sim_between_norm - np.nanmean(sim_between_norm, axis=0)) / np.nanstd(sim_between_norm, axis=0)
    selected_candidates = []
    unselected_candidates = [c for c in range(len(X))]
    j = np.argmax(doc_sim)
    selected_candidates.append(j)
    unselected_candidates.remove(j)
    for _ in range(N-1):
        selec_array = np.array(selected_candidates)
        unselec_array = np.array(unselected_candidates)
        distance_to_doc = doc_sim_norm[unselec_array, :]
        dist_between = sim_between_norm[unselec_array][:, selec_array]
        if dist_between.ndim == 1:
            dist_between = dist_between[:, np.newaxis]
        j = np.argmax(beta * distance_to_doc - (1 - beta) * np.max(dist_between, axis=1).reshape(-1, 1))
        item_idx = unselected_candidates[j]
        selected_candidates.append(item_idx)
        unselected_candidates.remove(item_idx)
    return candidates[selected_candidates].tolist()

def tokenize(sent):
    ret_l=[]
    for token in nlp(sent):
        ret_l.append((token.text, token.tag_))
    return ret_l

def extract_phrases(sent):
    keyphrase_candidate = set()

    np_parser = nltk.RegexpParser("""  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)""")  # Noun phrase parser
    tree = np_parser.parse(sent)  # Generator with one tree per sentence

    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
        # Concatenate the token with a space
        keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))

    keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 5}

    keyphrase_candidate = list(keyphrase_candidate)
    return keyphrase_candidate

def extract_candidates(sent):
    ret_l=[]
    propnoun_flag=False
    propnouns_l=[]
    temp=[]
    for token in sent:
        if token[1] in propnoun_tags:
            if propnoun_flag:
                temp.append(token[0])
            else:
                temp=[token[0]]
                propnoun_flag=True
        elif propnoun_flag:
            propnouns_l.append(" ".join(temp))
            propnoun_flag=False
        if token[1] in considered_tags and len(token[0])>3:
            ret_l.append(token[0])
    return np.unique(ret_l), np.unique(propnouns_l)

def the_destructor(row):
    sentence=row['long_description']
    try:
        if pd.isna(sentence):
            tags=url_parse(row['link'])
        else:
            sentence=re.sub("[\\\\\'\"]", "", sentence.lower())
            tokenized=tokenize(sentence)
            selected_phrases=np.array(extract_phrases(tokenized))
            sentence_emb=model.encode([sentence], show_progress_bar=False)
            embeddings=np.array(model.encode(selected_phrases, show_progress_bar=False))
            tags=get_topk(sentence_emb, embeddings, selected_phrases)
    except Exception as e:
        tags=['']
    return row.name, tags


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

    result=df_train[startnum:subsetnum].progress_apply(the_destructor, axis=1, result_type='expand')
    result.columns=['id', 'keywords']
    result=result.set_index('id')
    result.to_csv(DATA_DIR+"submission_%d.csv"%subsetnum)
    result.to_pickle(DATA_DIR+"submission_%d.pkl"%subsetnum)


