from os import PathLike
from pathlib import Path
from nltk import RegexpTokenizer
from requests import HTTPError, get
import numpy as np
import polars as pl
import logging
import dacy
from gensim.models import KeyedVectors


LOG = logging.getLogger(__name__)
entity_embeddings = {}
context_embeddings = {}
word2vec: dict | None = None
nlp = dacy.load("large")


def transform_behaviors(behaviors_raw: pl.LazyFrame, skip_impression=False):
    LOG.info(f"Starting transform_behaviors")
    df = (behaviors_raw
        .select('article_ids_inview', 'article_ids_clicked', 'impression_id', 'user_id')
        .with_columns(
            pl.struct(['article_ids_inview', 'article_ids_clicked'])
                .map_elements(lambda x: [(article, 1) if article in x['article_ids_clicked'] else (article, 0) for article in set(x['article_ids_inview'])])
                .alias('inview_label_combined'))
        .select('impression_id', 'user_id', 'inview_label_combined')
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
    return df


def transform_history(*input_files):
    LOG.info(f"Starting transform_history for input files: {input_files}")
    df = (
        pl.concat([pl.scan_parquet(file) for file in input_files])
        .select('user_id', 'article_id_fixed')
        .with_columns(pl
            .col('article_id_fixed')
            .map_elements(lambda ids: ','.join(map(str, ids)))
            .alias('article_id_fixed'))
    )
    return df


def tokenize_articles(articles_file: Path, tokenized_articles_file: Path):
    tokenizer = RegexpTokenizer(r"\w+")
    articles = (pl.scan_parquet(articles_file)
        .select("article_id", "title")
        .with_columns(pl.col("title").str.to_lowercase().alias("title"))
        .with_columns(pl.col("title")
            .map_elements(lambda title: tokenizer.tokenize(title))
            .alias("word_tokens")
        )
        .with_columns(pl.col("title")
            .map_elements(lambda title: title or " ")
            .map_batches(lambda titles: pl.Series(nlp.pipe(titles)))
            .map_elements(lambda doc: [{"id": ent.kb_id_, "start": ent.start, "end": ent.end} for ent in doc.ents if ent.kb_id_ != 'NIL'])
            .alias("entities")
        )
        .drop("title")
    )

    articles.collect(streaming=True).write_parquet(tokenized_articles_file)


def get_entity_embedding(entity: str):
    if entity not in entity_embeddings:
        try:
            response = get(f'https://wembedder.toolforge.org/api/vector/{entity}')
            response.raise_for_status()

            entity_embeddings[entity] = np.array(response.json()['vector'])
        except HTTPError as e:
            if e.response.status_code != 404:
                raise e

            entity_embeddings[entity] = None

    return entity_embeddings[entity]


def get_context_embedding(entity: str):
    if entity not in context_embeddings:
        try:
            response = get(f'https://www.wikidata.org/w/rest.php/wikibase/v0/entities/items/{entity}/statements')
            response.raise_for_status()

            statements = [statement for property_statements in response.json().values() for statement in property_statements]
            context = {statement['value']['content'] for statement in statements if statement['property']['data-type'] == 'wikibase-item' and 'value' in statement and 'content' in statement['value']}
            embeddings = [get_entity_embedding(entity) for entity in context]
            embeddings = [embedding for embedding in embeddings if embedding is not None]

            context_embeddings[entity] = np.mean(embeddings, axis=0) if embeddings else None
        except HTTPError as e:
            if e.response.status_code not in [400, 404]:
                raise e

            context_embeddings[entity] = None

    return context_embeddings[entity]


def create_embeddings(tokens, get_embedding, default_embedding=np.zeros(100)):
    embedding_tokens = [None]
    embeddings = [default_embedding]

    for token in tokens:
        embedding = get_embedding(token)

        if embedding is not None:
            embedding_tokens.append(token)
            embeddings.append(embedding)

    embeddings = np.stack(embeddings)
    token2id = {token: i for i, token in enumerate(embedding_tokens)}

    return embeddings, token2id


def entities_to_ids(entities, entity2id):
    ids = []

    for entity in entities:
        if entity["id"] in entity2id and entity["end"] > len(ids):
            if entity["start"] > len(ids):
                ids.extend([0] * (entity["start"] - len(ids)))

            ids.extend([entity2id[entity["id"]] for _ in range(entity["end"] - len(ids))])

    return ids


def create_feature_file(word2vec_path, tokenized_articles_file, test_tokenized_articles_file, word_embeddings_file, entity_embeddings_file, context_embeddings_file, news_feature_file, doc_size):
    articles = pl.concat([
        pl.scan_parquet(tokenized_articles_file),
        pl.scan_parquet(test_tokenized_articles_file)
    ])

    words = (articles
        .select("word_tokens")
        .explode("word_tokens")
        .unique("word_tokens")
        .collect()
    )["word_tokens"]

    entities = (articles
        .select("entities")
        .explode("entities")
        .drop_nulls("entities")
        .with_columns(pl.col("entities").map_elements(lambda e: e["id"]).alias("entities"))
        .select("entities")
        .unique()
        .collect()
    )["entities"]

    if not word2vec:
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    word_embeddings, word2id = create_embeddings(words, get_word_embedding)
    np.save(word_embeddings_file, word_embeddings)
    del word_embeddings

    entity_embeddings, entity2id = create_embeddings(entities, get_entity_embedding)
    np.save(entity_embeddings_file, entity_embeddings)
    del entity_embeddings

    context_embeddings, entity2id = create_embeddings(entities, get_context_embedding)
    np.save(context_embeddings_file, context_embeddings)
    del context_embeddings

    encoded_articles = (articles
        .with_columns(
            pl.col("word_tokens")
                .map_elements(lambda tokens: [word2id[token] if token in word2id else 0 for token in tokens])
                .map_elements(lambda tokens: list(tokens[:doc_size]) + [0] * (doc_size - len(tokens)))
                .map_elements(lambda tokens: ','.join(map(str, tokens)))
                .alias("word_tokens"),
            pl.col("entities")
                .map_elements(lambda entities: entities_to_ids(entities, entity2id))
                .map_elements(lambda entities: list(entities[:doc_size]) + [0] * (doc_size - len(entities)))
                .map_elements(lambda entities: ','.join(map(str, entities)))
                .alias("entities")
        )
    )

    encoded_articles.sink_csv(news_feature_file, separator=' ', quote_style='never', include_header=False)


def get_word_embedding(word):
    return word2vec[word] if word in word2vec else None


def transform_behaviors_test(test_raw_file: str, indexed_behaviors_file: str, test_file: str):
    LOG.info(f"Writing index behaviors data to: {indexed_behaviors_file}")
    indexed_behaviors = (pl
        .scan_parquet(test_raw_file)
        .with_row_index()
        .select('index', 'article_ids_inview', 'impression_id', 'user_id')
        .rename({'article_ids_inview': 'article_id'})
        .with_columns(pl.col('article_id').map_elements(lambda ids: list(range(len(ids)))).alias('article_index'))
        .explode('article_id', 'article_index')
    )
    indexed_behaviors.collect(streaming=True).write_parquet(indexed_behaviors_file)

    LOG.info(f"Starting transform_behaviors for input: {test_raw_file}, writing to: {test_file}")
    test_data = (pl
        .scan_parquet(indexed_behaviors_file)
        .with_columns(
            pl.struct(['article_id', 'impression_id'])
                .map_elements(lambda x: f"{x['article_id']}%{x['impression_id']}")
                .alias('article_impression'),
            pl.lit(0).alias('label')
        )
        .select('label', 'user_id', 'article_impression')
    )
    test_data.sink_csv(test_file, separator=' ', quote_style='never', include_header=False)


def calculate_rankings(indexed_behaviors_file: str, scores_file: str):
    """
    Combine the prediction file (contains only a single column of scores) with the previously generated index file, which contains a column that specifies to which row of the original testfile a score belongs and the impression id.
    The resulting LazyFrame of the concatenation is then grouped by the impression_id, where all scores of the same impression id are concatenated, resulting in the same layout as the original test file with the same order of rows.

    Calculate the rankings for the given score column and return the resulting LazyFrame which
    has the correct schema for a handin at the RecSys challenge (columns: impression_id, ranking).
    """
    rankings = (
        pl.concat([
            pl.scan_csv(scores_file, has_header=False, new_columns=["score"]),
            pl.scan_parquet(indexed_behaviors_file),
        ], how="horizontal")
        .group_by("index", "impression_id")
        .agg(pl.col("article_index"), pl.col("score"))
        .with_columns(
            pl.struct(["article_index", "score"])
                .map_elements(lambda x: (np.array(x["article_index"])[np.argsort(x["score"])[::-1]] + 1).tolist())
                .map_elements(lambda el: str(el.to_list()).replace(" ", ""))
                .alias("ranking")
        )
        .drop("article_index", "score")
        .collect(streaming=True)
        .sort("index")
        .select("impression_id", "ranking")
    )

    return rankings
