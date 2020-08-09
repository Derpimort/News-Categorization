from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from tqdm.auto import tqdm

STOPWORDS = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def lemma_pp(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(lemma.lemmatize(token))
    return result

def get_sentences(corpus, progress=False):
    sentences=[]
    if progress:
        corpus=tqdm(corpus)
    for desc in corpus:
        sentences.append(" ".join(lemma_pp(desc)))
    return sentences
