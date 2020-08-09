# News-Categorization
Unsupervised categorization of news articles

* [Setup](#setup)
* [Run](#run)
* [Files](#files)
* [Data cleaning](#data-cleaning)
* [Preprocessing and language detection](#preprocessing-and-language-detection)
* [Keyword extraction](#keyword-extraction)
* [References](#references)


## Setup (TODO: update with single DOCKERFILE asap)

0. Install PyTorch
    - `pip install torch torchvision` if cuda gpu available (Recommended).
    - `pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html` for cpu only.
    - https://pytorch.org - or just go to this site to get the best version for you.
1. Run `pip install -r requirements.txt` or `conda env create -f environment.yml` if conda is installed.
2. Open a python interpreter and run
    ```
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```
3. Download the following and extract inside data/
    ```
    https://download.geonames.org/export/dump/IN.zip
    https://download.geonames.org/export/dump/admin1CodesASCII.txt
    ```
4. Install docker and run following commands for elastic and kibana(development) if needed
    ```
    docker pull docker.elastic.co/elasticsearch/elasticsearch-oss:7.8.1
    docker pull docker.elastic.co/kibana/kibana-oss:7.8.1
    docker run -d --name elastic -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch-oss:7.8.1
    docker run -d --name kibana --link elastic:elasticsearch -p 5601:5601 docker.elastic.co/kibana/kibana-oss:7.8.1
    ```
    Run the curl command in `create_index.txt` to create elasticsearch index

## Run

1. Put the dataset inside a folder `data/` and run `data_clean.py`
2. Run `get_langs_fast.py`.
3. Run `ultimate_destruction.py`, when asked for input enter the `start end` values for DF subpart processing.
4. Run `get_locs_people.py`, `get_time.py`, `get_coords.py` for extracting locs, people and time.
5. Results will be saved in `data/cleaned/submission_*.csv`.
6. Run `populate_elastic.py` (or `populate_elastic_notime.py` if web scraped html are not available), default elastic port used on `localhost`.

## Files

- `data_clean` used to get a pandas DataFrame from the dataset csv.
- `get_langs_fast.py` used to get a languages DataFrame for title, desc and long_desc in the processsed DataFrame.
- `get_locs_people.py`, `get_time.py`, `get_coords.py` used to extraction location, time and coordinates from articles and some scraped urls.
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
- `RoBERTa` requires minimal to no preprocessing.

## Keyword extraction

- A `the_destructor` function is applied on each row and it does the following:
    - Iterate through each row and give `long_description` to `RoBERTa` and `title` or `description` for Null content to the `multilingual bert`.
    - Sentences tokenized.
    - Keywords are selected from the tokenized content using `RegexpParser` tree.
    - All candidate keywords generated from the tree are then processed with `RoBERTa` or `BERT` to get their embeddings alongwith main content embeddings.
    - The candidate keywords and main content embeddings are then passed to the `get_topk` function which applies `MMR` algo to return top k keywords.
    - `MMR` automatically eliminates duplicates (i.e. keywords with similarity over a given threshold).
    - `the_destructor` return id and keywords for each row.

## Categorization

- Embeddings generated for all categories in the tree.
- Category embeddings and article embeddings fed into MMR to get top 2 matching categories.

## References

- [swisscom/ai-research-keyphrase-extraction](https://github.com/swisscom/ai-research-keyphrase-extraction)



```
Bennani-Smires K., Musat C., Hossman A., Baeriswyl M., Jaggi M., 2018 Simple Unsupervised Keyphrase Extraction using Sentence Embeddings. arXiv:1801.04470
```


```
Liu Y., Ott M., Goyal N., Du J., Joshi M., Chen D., Levy O., Lewis M., Zettlemoyer L., Stoyanov V., 2019  RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692
```
