import pickle
import polars as pl
import numpy as np
import re

import group_33.constants as Constants
from recommenders.models.newsrec.newsrec_utils import word_tokenize

pl.Config.set_tbl_rows(100)
pl.Config.set_streaming_chunk_size(500_000)

def transfrom_behavior_file(
    behaviors_path, history_path, result_path, history_size=None, sample_size=1
):
    def transform_row(row):
        impression_id = row[Constants.COL_IMPRESSION_ID_IDX]
        user_id = row[Constants.COL_USER_ID_IDX]
        impression_time = row[Constants.COL_IMPRESSION_TIME_IDX]
        clicked_articles = user_history.get(user_id, {}).get("article_id", [])
        timestamps = user_history.get(user_id, {}).get("impression_time", [])

        # Filter click history to include only clicks before the impression time
        user_click_history = [
            f"{article_id}"
            for article_id, timestamp in zip(clicked_articles, timestamps)
            if timestamp < impression_time
        ]

        user_click_history = user_click_history[-history_size:]
        user_click_history_str = " ".join(user_click_history)

        # Prepare impression news
        inview_articles = row[Constants.COL_INVIEW_ARTICLE_IDS_IDX]
        clicked_articles = row[Constants.COL_CLICKED_ARTICLE_IDS_IDX]
        impression_news = [
            f"{article_id}-{1 if article_id in clicked_articles else 0}"
            for article_id in inview_articles
        ]
        impression_news_str = " ".join(impression_news)

        return (
            impression_id,
            user_id,
            impression_time,
            user_click_history_str,
            impression_news_str,
        )

    behaviors = pl.read_parquet(behaviors_path)
    history = pl.read_parquet(history_path)

    if history_size is None:
        history_size = history.shape[0]

    # Transform history to a dictionary for fast lookup
    user_history = {}
    for row in history.iter_rows(named=True):
        user_history[row['user_id']] = {
            "article_id": row["article_id_fixed"],
            "impression_time": row["impression_time_fixed"],
        }


    result_behavior_df = pl.DataFrame(behaviors.map_rows(transform_row))
    result_behavior_df.columns = [
        Constants.COL_IMPRESSION_ID,
        Constants.COL_USER_ID,
        Constants.COL_IMPRESSION_TIME,
        Constants.COL_USER_CLICK_HISTORY,
        Constants.COL_IMPRESSION_ARTICLES,
    ]
    result_behavior_df.sample(fraction=sample_size).write_csv(
        result_path, quote_style="never", include_header=False, separator="\t"
    )
    return result_behavior_df

def transform_articles_file(articles_path, result_path):
    def clean_text(column):
        return column.str.replace_all("\n", "").str.replace_all("\t", " ")

    articles = pl.read_parquet(articles_path)

    # Select relevant columns and apply the cleaning function to 'title' and 'body'
    articles = articles.select(
        [
            Constants.COL_ARTICLE_ID,
            Constants.COL_ARTICLE_CATEGORY,
            Constants.COL_ARTICLE_TITLE,
            Constants.COL_ARTICLE_BODY,
            Constants.COL_ARTICLE_URL,
        ]
    ).with_columns(
        [
            clean_text(articles[Constants.COL_ARTICLE_TITLE]),
            clean_text(articles[Constants.COL_ARTICLE_BODY]),
        ]
    )

    articles.write_csv(
        result_path, quote_style="never", include_header=False, separator="\t"
    )
    return articles


def generate_user_mapping(behavior_df, user_dict_file):
    user_id_mapping = {
        user_id: i
        for i, user_id in enumerate(behavior_df[Constants.COL_USER_ID].unique())
    }

    # Dump the dictionary as a pkl file
    with open(user_dict_file, "wb") as f:
        pickle.dump(user_id_mapping, f)

def generate_word_dict(df_articles, word_dict_file):
    # Tokenize the words
    tokens = df_articles['title'].map_elements(word_tokenize)
    words_id_mapping = {word: i for i, word in enumerate(tokens.explode().unique())}
    
    # Dump the dictionary as a pkl file
    with open(word_dict_file, 'wb') as f:
        pickle.dump(words_id_mapping, f)

    return words_id_mapping

def generate_word_embeddings(words_id_mapping, embedding_path, result_path):
    
    # Function to load GloVe embeddings
    def load_embeddings(file_path):
        embeddings = {}
        with open(file_path, 'r', encoding='iso-8859-1') as file:
            for line in file:
                # Split on the first occurrence of a number
                key, values = re.split(r' (?=\d)', line.strip(), maxsplit=1)
                if key in words_id_mapping:
                    values = [float(x) for x in values.split()]
                    embeddings[key] = values
        return embeddings

    embeddings_index = load_embeddings(embedding_path)

    # Create embedding matrix
    embedding_dim = 100
    embedding_matrix = np.zeros((len(words_id_mapping), embedding_dim))

    for word, idx in words_id_mapping.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    # Save the embedding matrix to a file
    np.save(result_path, embedding_matrix)