# News-Categorization
Unsupervised categorization of news articles

* [Setup](#setup)
* [Run](#run)
* [Files](#files)
* [Data cleaning](#data-cleaning)
* [Preprocessing and language detection](#preprocessing-and-language-detection)
* [Keyword extraction](#keyword-extraction)
* [References](#references)


## Setup

0. Install PyTorch
    - `pip install torch torchvision` if cuda gpu available (Recommended).
    - `pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html` for cpu only.
1. Run `pip install -r requirements.txt`
2. Open a python interpreter and run
    ```
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Run

1. Put the dataset inside a folder `data/` and run `data_clean.py`
2. Copy the cleaned `train_cleaned.pkl` into `data/cleaned` and run `get_langs_fast.py`
3. Run `ultimate_destruction.py`, when asked for input enter the `start end` values for DF subpart processing.
4. Results will be saved in `data/cleaned/submission_*.csv`

## Files

- `data_clean` used to get a pandas DataFrame from the dataset csv
- `get_langs_fast.py` used to get a languages DataFrame for title, desc and long_desc in the processsed DataFrame
- `preprocess.py` used to apply lemmatization and simple preprocessing on text.
- `ultimate_destruction.py` uses the combination of both of the above dataframes with the complete implementation to generate a DataFrame of keywords.

## Data cleaning

- `csv.reader` used to read the given dataset.
- The read lines are directly fed to a `csv.writer` with a different separator to avoid collisions.
- The lines which don't have exactly 5 rows are fed into a `lineCleaner` function.
- `lineCleaner` uses regex to find the id, url and long_description. The rest is found using a combination of separator logic based on unbalanced `quotechars`
- If even the `lineCleaner` fails, the row is written into the badrows.csv file with None values for all except `id` in the main DF. We only had 693 bad rows in this run(<0.05%), so we didn't waste time on it anymore.

## Preprocessing and language detection

- `langdetect` library was used to get languages for each content. Parallelized the detection for faster results.
- `RoBERTa` requires minimal to no preprocessing so the only sentence preprocessing (lemmatization and stopwords removal) was done for the url parser.

## Keyword extraction

- All of the non-english rows in the `long_description` column are filled with english `title` or `description` (if available). The remaining non english rows are set to `None`.
- A `the_destructor` function is applied on each row and it does the following:
    - Iterate through each row and give links for Null content to url parses to get keywords
    - Non-null rows have their sentences tokenized.
    - Keywords are selected from the tokenized content using `RegexpParser` tree.
    - All candidate keywords generated from the tree are then processed with `RoBERTa` to get their embeddings alongwith main content embeddings.
    - The candidate keywords and main content embeddings are then passed to the `get_topk` function which applies `MMR` algo to return top k keywords.
    - `MMR` automatically eliminates duplicates (i.e. keywords with similarity over a given threshold).
    - `the_destructor` return id and keywords for each row.
- The id and keywords are then written in a submission dataframe.
- `url_parse` is a simple function to get preprocessed tags from the url of the article.
- Amongst all the data about 3080(<0.25%) rows had empty keywords as a result of a combination of no english content and empty urls.

## References

- Basically [this](https://github.com/swisscom/ai-research-keyphrase-extraction) with `RoBERTa embeddings`


```
Bennani-Smires K., Musat C., Hossman A., Baeriswyl M., Jaggi M., 2018 Simple Unsupervised Keyphrase Extraction using Sentence Embeddings. arXiv:1801.04470
```


```
Liu Y., Ott M., Goyal N., Du J., Joshi M., Chen D., Levy O., Lewis M., Zettlemoyer L., Stoyanov V., 2019  RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692
```
