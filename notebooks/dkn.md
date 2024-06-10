---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: 'Python 3.7.11 64-bit (''tf2'': conda)'
    language: python
    name: python3
---

<i>Copyright (c) Recommenders contributors.</i>

<i>Licensed under the MIT License.</i>

```python
# !cd .. && pip install -e .
```

<!-- #region -->
# DKN : Deep Knowledge-Aware Network for News Recommendation

DKN \[1\] is a deep learning model which incorporates information from knowledge graph for better news recommendation. Specifically, DKN uses TransX \[2\] method for knowledge graph representation learning, then applies a CNN framework, named KCNN, to combine entity embedding with word embedding and generate a final embedding vector for a news article. CTR prediction is made via an attention-based neural scorer. 

## Properties of DKN:

- DKN is a content-based deep model for CTR prediction rather than traditional ID-based collaborative filtering. 
- It makes use of knowledge entities and common sense in news content via joint learning from semantic-level and knowledge-level representations of news articles.
- DKN uses an attention module to dynamically calculate a user's aggregated historical representation.


## Data format

DKN takes several files as input as follows:

- **training / validation / test files**: each line in these files represents one instance. Impressionid is used to evaluate performance within an impression session, so it is only used when evaluating, you can set it to 0 for training data. The format is : <br> 
`[label] [userid] [CandidateNews]%[impressionid] `<br> 
e.g., `1 train_U1 N1%0` <br> 

- **user history file**: each line in this file represents a users' click history. You need to set `history_size` parameter in the config file, which is the max number of user's click history we use. We will automatically keep the last `history_size` number of user click history, if user's click history is more than `history_size`, and we will automatically pad with 0 if user's click history is less than `history_size`. the format is : <br> 
`[Userid] [newsid1,newsid2...]`<br>
e.g., `train_U1 N1,N2` <br> 

- **document feature file**: It contains the word and entity features for news articles. News articles are represented by aligned title words and title entities. To take a quick example, a news title may be: <i>"Trump to deliver State of the Union address next week"</i>, then the title words value may be `CandidateNews:34,45,334,23,12,987,3456,111,456,432` and the title entitie value may be: `entity:45,0,0,0,0,0,0,0,0,0`. Only the first value of entity vector is non-zero due to the word "Trump". The title value and entity value is hashed from 1 to `n` (where `n` is the number of distinct words or entities). Each feature length should be fixed at k (`doc_size` parameter), if the number of words in document is more than k, you should truncate the document to k words, and if the number of words in document is less than k, you should pad 0 to the end. 
the format is like: <br> 
`[Newsid] [w1,w2,w3...wk] [e1,e2,e3...ek]`

- **word embedding/entity embedding/ context embedding files**: These are `*.npy` files of pretrained embeddings. After loading, each file is a `[n+1,k]` two-dimensional matrix, n is the number of words(or entities) of their hash dictionary, k is dimension of the embedding, note that we keep embedding 0 for zero padding. 

In this experiment, we used GloVe\[4\] vectors to initialize the word embedding. We trained entity embedding using TransE\[2\] on knowledge graph and context embedding is the average of the entity's neighbors in the knowledge graph.<br>

## MIND dataset

MIND dataset\[3\] is a large-scale English news dataset. It was collected from anonymized behavior logs of Microsoft News website. MIND contains 1,000,000 users, 161,013 news articles and 15,777,377 impression logs. Every news article contains rich textual content including title, abstract, body, category and entities. Each impression log contains the click events, non-clicked events and historical news click behaviors of this user before this impression.

A smaller version, [MIND-small](https://azure.microsoft.com/en-us/services/open-datasets/catalog/microsoft-news-dataset/), is a small version of the MIND dataset by randomly sampling 50,000 users and their behavior logs from the MIND dataset.

The datasets contains these files for both training and validation data:

#### behaviors.tsv

The behaviors.tsv file contains the impression logs and users' news click hostories. It has 5 columns divided by the tab symbol:

+ Impression ID. The ID of an impression.
+ User ID. The anonymous ID of a user.
+ Time. The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM".
+ History. The news click history (ID list of clicked news) of this user before this impression.
+ Impressions. List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click).

One simple example: 

`1    U82271    11/11/2019 3:28:58 PM    N3130 N11621 N12917 N4574 N12140 N9748    N13390-0 N7180-0 N20785-0 N6937-0 N15776-0 N25810-0 N20820-0 N6885-0 N27294-0 N18835-0 N16945-0 N7410-0 N23967-0 N22679-0 N20532-0 N26651-0 N22078-0 N4098-0 N16473-0 N13841-0 N15660-0 N25787-0 N2315-0 N1615-0 N9087-0 N23880-0 N3600-0 N24479-0 N22882-0 N26308-0 N13594-0 N2220-0 N28356-0 N17083-0 N21415-0 N18671-0 N9440-0 N17759-0 N10861-0 N21830-0 N8064-0 N5675-0 N15037-0 N26154-0 N15368-1 N481-0 N3256-0 N20663-0 N23940-0 N7654-0 N10729-0 N7090-0 N23596-0 N15901-0 N16348-0 N13645-0 N8124-0 N20094-0 N27774-0 N23011-0 N14832-0 N15971-0 N27729-0 N2167-0 N11186-0 N18390-0 N21328-0 N10992-0 N20122-0 N1958-0 N2004-0 N26156-0 N17632-0 N26146-0 N17322-0 N18403-0 N17397-0 N18215-0 N14475-0 N9781-0 N17958-0 N3370-0 N1127-0 N15525-0 N12657-0 N10537-0 N18224-0 `

#### news.tsv

The news.tsv file contains the detailed information of news articles involved in the behaviors.tsv file. It has 7 columns, which are divided by the tab symbol:

+ News ID
+ Category
+ SubCategory
+ Title
+ Abstract
+ URL
+ Title Entities (entities contained in the title of this news)
+ Abstract Entities (entites contained in the abstract of this news)

One simple example: 

`N46466    lifestyle    lifestyleroyals    The Brands Queen Elizabeth, Prince Charles, and Prince Philip Swear By    Shop the notebooks, jackets, and more that the royals can't live without.    https://www.msn.com/en-us/lifestyle/lifestyleroyals/the-brands-queen-elizabeth,-prince-charles,-and-prince-philip-swear-by/ss-AAGH0ET?ocid=chopendata    [{"Label": "Prince Philip, Duke of Edinburgh", "Type": "P", "WikidataId": "Q80976", "Confidence": 1.0, "OccurrenceOffsets": [48], "SurfaceForms": ["Prince Philip"]}, {"Label": "Charles, Prince of Wales", "Type": "P", "WikidataId": "Q43274", "Confidence": 1.0, "OccurrenceOffsets": [28], "SurfaceForms": ["Prince Charles"]}, {"Label": "Elizabeth II", "Type": "P", "WikidataId": "Q9682", "Confidence": 0.97, "OccurrenceOffsets": [11], "SurfaceForms": ["Queen Elizabeth"]}]    [] `

#### entity_embedding.vec & relation_embedding.vec

The entity_embedding.vec and relation_embedding.vec files contain the 100-dimensional embeddings of the entities and relations learned from the subgraph (from WikiData knowledge graph) by TransE method. In both files, the first column is the ID of entity/relation, and the other columns are the embedding vector values.

One simple example: 

`Q42306013  0.014516 -0.106958 0.024590 ... -0.080382`


## DKN architecture

The following figure shows the architecture of DKN.

![](https://recodatasets.z20.web.core.windows.net/images/dkn_architecture.png)

DKN takes one piece of candidate news and one piece of a user’s clicked news as input. For each piece of news, a specially designed KCNN is used to process its title and generate an embedding vector. KCNN is an extension of traditional CNN that allows flexibility in incorporating symbolic knowledge from a knowledge graph into sentence representation learning. 

With the KCNN, we obtain a set of embedding vectors for a user’s clicked history. To get final embedding of the user with
respect to the current candidate news, we use an attention-based method to automatically match the candidate news to each piece
of his clicked news, and aggregate the user’s historical interests with different weights. The candidate news embedding and the user embedding are concatenated and fed into a deep neural network (DNN) to calculate the predicted probability that the user will click the candidate news.
<!-- #endregion -->

## Global settings and imports

```python
import os
import sys
from tempfile import TemporaryDirectory
import logging
import numpy as np
from pathlib import Path

import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# logging.basicConfig(level=logging.INFO)

from pathlib import Path
import polars as pl

from recommenders.datasets.download_utils import maybe_download
from recommenders.datasets.mind import (download_mind, 
                                     extract_mind, 
                                     read_clickhistory, 
                                     get_train_input, 
                                     get_valid_input, 
                                     get_user_history,
                                     get_words_and_entities,
                                     generate_embeddings) 
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.models.dkn import DKN
from recommenders.models.deeprec.io.dkn_iterator import DKNTextIterator
from recommenders.utils.notebook_utils import store_metadata

print(f"System version: {sys.version}")
print(f"Tensorflow version: {tf.__version__}")
```

```python tags=["parameters"]
# DKN parameters
epochs = 2
history_size = 5
batch_size = 1000

DATASET_NAME = "small" # one of: demo, small, large

# Paths
tmp_path = os.path.join("..", "tmp")
tmp_pathlib = Path(tmp_path)
(tmp_pathlib / DATASET_NAME / "validation").mkdir(exist_ok=True, parents=True)
(tmp_pathlib / DATASET_NAME / "train").mkdir(exist_ok=True, parents=True)



test_raw_file = Path().resolve().parent/ "downloads" / "ebnerd_testset" / "test" / "behaviors.parquet"
test_file = os.path.join(tmp_path, "test_behavior.txt")

data_path = os.path.join("..", "downloads", DATASET_NAME)
train_file = os.path.join(tmp_path, DATASET_NAME, "train", "behaviours.txt")
valid_file = os.path.join(tmp_path, DATASET_NAME, "validation", "behaviours.txt")
user_history_file = os.path.join(tmp_path, DATASET_NAME, "user_history.txt")
infer_embedding_file = os.path.join(tmp_path, DATASET_NAME, "infer_embedding.txt")

LOG.info(tmp_path)
```

## Data preparation

In this example, let's go through a real case on how to apply DKN on a raw news dataset from the very beginning. We will download a copy of open-source MIND dataset, in its original raw format. Then we will process the raw data files into DKN's input data format, which is stated previously. 

```python
def transform_behaviors(input_file: str, output_file: str, skip_impression=False):
    LOG.info(f"Starting transform_behaviors for input: {input_file}, writing to: {output_file}")
    df = (
        pl.scan_parquet(input_file) 
        .select('article_ids_inview', 'article_ids_clicked', 'impression_id', 'user_id') 
        .with_columns(
            pl.struct(['article_ids_inview', 'article_ids_clicked']) 
                .map_elements(lambda x: [(article, 1) if article in x['article_ids_clicked'] else (article, 0) for article in set(x['article_ids_inview'])]) 
                .alias('inview_label_combined')) 
        .select('impression_id', 'user_id', 'inview_label_combined') 
       # .head(n=13113) # first bad row when exploding both cols in one step
       # .filter(pl.col("impression_id") == 845918)
        .explode(['inview_label_combined']) # workaround, got error when exploding both columns directly
        .with_columns(pl.col('inview_label_combined').list[0].alias('article_id'))
        .with_columns(pl.col('inview_label_combined').list[1].alias('label'))
        .select('impression_id', 'user_id', 'article_id', 'label')
    )

    if skip_impression:
        df = df.with_columns(pl.col('article_id').alias('article_impression'))
    else:
        df = df.with_columns(
            pl.struct(['article_id', 'impression_id']) \
                .map_elements(lambda x: f"{x['article_id']}%{x['impression_id']}")
                .alias('article_impression')) \
            .select('label', 'user_id', 'article_impression')
    df.sink_csv(output_file, separator=' ', quote_style='never', include_header=False)
```


```python
def transform_history(train_input_file: str, val_input_file: str, test_input_file, output_file: str):
    LOG.info(f"Starting transform_history for train input: {train_input_file} and validation input: {val_input_file}, writing to: {output_file}")
    df = (
        pl.concat([
            pl.scan_parquet(train_input_file),
            pl.scan_parquet(val_input_file),
            pl.scan_parquet(test_input_file)
            ])
        .select('user_id', 'article_id_fixed') #.collect()
        .with_columns(pl.col('article_id_fixed').map_elements(lambda ids: ','.join(map(str, ids))).alias('article_id_fixed'))
    )
    df.sink_csv(output_file, separator=' ', quote_style='never', include_header=False)
```


```python
pl.Config.set_streaming_chunk_size(5_000_000)
force_reload = False

if not Path(train_file).exists() or force_reload:
    transform_behaviors(os.path.join(data_path, 'train', 'behaviors.parquet'), train_file)

if not Path(valid_file).exists() or force_reload:
    transform_behaviors(os.path.join(data_path, 'validation', 'behaviors.parquet'), valid_file)

if not Path(user_history_file).exists() or force_reload:
    transform_history(os.path.join(data_path, 'train', 'history.parquet'), os.path.join(data_path, 'validation', 'history.parquet'), os.path.join("../downloads/ebnerd_testset/test/history.parquet"), user_history_file)
```

```python
pl.threadpool_size()
# train_zip, valid_zip = download_mind(size=MIND_SIZE, dest_path=data_path)
# train_path, valid_path = extract_mind(train_zip, valid_zip)
```

```python
# train_session, train_history = read_clickhistory(train_path, "behaviors.tsv")
# valid_session, valid_history = read_clickhistory(valid_path, "behaviors.tsv")
# get_train_input(train_session, train_file)
# get_valid_input(valid_session, valid_file)
# get_user_history(train_history, valid_history, user_history_file)
```

```python
# train_news = os.path.join(train_path, "news.tsv")
# valid_news = os.path.join(valid_path, "news.tsv")
# news_words, news_entities = get_words_and_entities(train_news, valid_news)
```

```python
# train_entities = os.path.join(train_path, "entity_embedding.vec")
# valid_entities = os.path.join(valid_path, "entity_embedding.vec")
# news_feature_file, word_embeddings_file, entity_embeddings_file = generate_embeddings(
#     data_path,
#     news_words,
#     news_entities,
#     train_entities,
#     valid_entities,
#     max_sentence=10,
#     word_embedding_dim=100,
# )
```
Format:
`[Newsid] [w1,w2,w3...wk] [e1,e2,e3...ek]`

```python

from nltk import RegexpTokenizer

def tokenize_articles(path: str):
    # word_tokens_path = os.path.join(data_path, "word_tokens.txt")
    tokenizer = RegexpTokenizer(r"\w+")
    articles = (
        pl.concat([
            pl.scan_parquet(path),])
        .select("article_id", "title")
        .with_columns(pl.col("article_id").cast(pl.String))
        .with_columns(pl.col("title").map_elements(lambda ids: tokenizer.tokenize(ids)).alias("word_tokens"))
        .with_columns(pl.col("title").map_elements(lambda ids: []).alias("entities")) # TODO: add entities in lambda
        .drop("title")
    )

    word_tokens = {}
    entities = {}
    for tup in articles.collect().iter_rows():
        word_tokens[tup[0]] = tup[1]
        entities[tup[0]] = tup[2]

    return word_tokens, entities

# articles.select("article_id", "word_tokens").collect().write_csv(os.path.join(data_path, "word_tokens.txt"), separator=' ', quote_style='never', include_header=False)

# articles.select("article_id", "entities").collect().write_csv(entities, separator=' ', quote_style='never', include_header=False)

entities_path = os.path.join(tmp_path, "entities.txt")

dataset_articles = os.path.join(data_path, "articles.parquet")
testset_articles = os.path.join("../downloads/ebnerd_testset/articles.parquet")

word_tokens, entities = tokenize_articles(testset_articles)

Path(entities_path).touch()

word_tokens
```



```python
news_feature_file, word_embeddings_file, entity_embeddings_file = generate_embeddings(
    data_path,
    word_tokens,
    entities,
    entities_path, # empty file for now
    entities_path,
    max_sentence=10,
    word_embedding_dim=100,
)
```

```python
file = np.load(word_embeddings_file)
file.shape
file[1]
```

## Create hyper-parameters

```python
yaml_file = maybe_download(url="https://recodatasets.z20.web.core.windows.net/deeprec/deeprec/dkn/dkn_MINDsmall.yaml", 
                           work_directory=data_path)
hparams = prepare_hparams(yaml_file,
                          show_step=100,
                          news_feature_file=news_feature_file,
                          user_history_file=user_history_file,
                          wordEmb_file=word_embeddings_file,
                          entityEmb_file=entity_embeddings_file,
                          epochs=epochs,
                          save_model=True,
                          MODEL_DIR="../tmp/model",
                          history_size=history_size,
                          batch_size=batch_size)
```

## Train the DKN model

```python
model = DKN(hparams, DKNTextIterator)
```

small:   2_585_747
large: 133_810_641
--> factor of 51 between small & large

on small:
    batch 1000 -> 2600 steps
    hist 50
    takes 50 min / epoch

on small:
    batch 1000
    hist 5
    takes 10 min / epoch
```python
model.fit(train_file, valid_file)
```

## Evaluate the DKN model

```python
pl.scan_parquet(str(test_raw_file)).select( "article_ids_inview", "impression_id", "user_id").head().collect()
```


```python
valid_file

def transform_behaviors_test(input_file: str, output_file: str):
    LOG.info(f"Starting transform_behaviors for input: {input_file}, writing to: {output_file}")
    df = (
        pl.scan_parquet(input_file) 
            .select('article_ids_inview', 'impression_id', 'user_id') 
            .explode(['article_ids_inview'])
            .rename({'article_ids_inview': 'article_id'}) 
            .select('impression_id', 'user_id', 'article_id')
    )
    df = df.with_columns(
        pl.struct(['article_id', 'impression_id']) 
            .map_elements(lambda x: f"{x['article_id']}%{x['impression_id']}")
            .alias('article_impression')) \
        .with_columns(label = pl.lit(0)) \
        .select('label', 'user_id', 'article_impression')
    df.sink_csv(output_file, separator=' ', quote_style='never', include_header=False)

#    df = (
#        pl.scan_parquet(input_file)
#            .select('article_ids_inview', 'impression_id', 'user_id')
#            .collect()
#            .explode(['article_ids_inview'])
#    )
#
#    if skip_impression:
#        df = df.with_columns(pl.col('article_ids_inview').alias('article_impression'))
#    else:
#        df = df.with_columns(
#            pl.struct(['article_ids_inview', 'impression_id'])
#                .map_elements(lambda x: f"{x['article_ids_inview']}%{x['impression_id']}")
#                .alias('article_impression')
#        )
#
#    df = df.select('user_id', 'article_impression')
#    df.write_csv(output_file, separator=' ', quote_style='never', include_header=False)

if not Path(test_file).exists():
    transform_behaviors_test(str(test_raw_file), test_file)
```

```python
res = model.run_eval(str(valid_file))
print(res)
```

```python
prediction_file = "../tmp/prediction.csv"
model.predict(str(test_file), prediction_file)

# Record results for tests - ignore this cell
# store_metadata("auc", res["auc"])
# store_metadata("group_auc", res["group_auc"])
# store_metadata("ndcg@5", res["ndcg@5"])
# store_metadata("ndcg@10", res["ndcg@10"])
# store_metadata("mean_mrr", res["mean_mrr"])
```

```python
# model.saver.save()
```

```python
df: pl.LazyFrame = (
    pl.concat([
        pl.scan_csv(prediction_file, has_header=False, new_columns=["score"]),
        pl.scan_csv(test_file, has_header=False, separator=" ", new_columns=["label", "user", "article_impression"],)],
        how = "horizontal")
    .with_columns([pl.col("article_impression")
        .str.split_exact("%", 1)
        .struct.rename_fields(["article", "impression"]).alias("split")])
    .unnest("split")
    .drop("article_impression", "user", "label")
)

df = (df.head(1000)
    .group_by("impression")
    .agg((pl.col("score"), pl.col("article")))

    
)

df.head().collect()
```
```python
impression_article_raw = (
    pl.concat([
    pl.scan_parquet(test_raw_file)
    .select('article_ids_inview', 'impression_id'),
    df], how="horizontal")
)

impression_article_raw.head().collect()
```

## Document embedding inference API

After training, you can get document embedding through this document embedding inference API. The input file format is same with document feature file. The output file fomrat is: `[Newsid] [embedding]`

```python
model.run_get_embedding(news_feature_file, infer_embedding_file)
```

<!-- #region -->
## Results on large MIND dataset

Here are performances using the large MIND dataset (1,000,000 users, 161,013 news articles and 15,777,377 impression logs). 

| Models | g-AUC | MRR |NDCG@5 | NDCG@10 |
| :------| :------: | :------: | :------: | :------ |
| LibFM | 0.5993 | 0.2823 | 0.3005 | 0.3574 |
| Wide&Deep | 0.6216 | 0.2931 | 0.3138 | 0.3712 |
| DKN | 0.6436 | 0.3128 | 0.3371 | 0.3908|


Note that the results of DKN are using Microsoft recommender and the results of the first two models come from the MIND paper \[3\].
We compare the results on the same test dataset. 

One epoch takes 6381.3s (5066.6s for training, 1314.7s for evaluating) for DKN on GPU. Hardware specification for running the large dataset: <br>
GPU: Tesla P100-PCIE-16GB <br>
CPU: 6 cores Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
<!-- #endregion -->

## References

\[1\] Hongwei Wang, Fuzheng Zhang, Xing Xie and Minyi Guo, "DKN: Deep Knowledge-Aware Network for News Recommendation", in Proceedings of the 2018 World Wide Web Conference (WWW), 2018, https://arxiv.org/abs/1801.08284. <br>
\[2\] Knowledge Graph Embeddings including TransE, TransH, TransR and PTransE. https://github.com/thunlp/KB2E <br>
\[3\] Fangzhao Wu et al., "MIND: A Large-scale Dataset for News Recommendation", Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, https://msnews.github.io/competition.html. <br>
\[4\] GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/projects/glove/
