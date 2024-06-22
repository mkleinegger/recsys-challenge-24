import polars as pl
import numpy as np


def calculate_rankings(scores_file: str, indexed_behaviors_file: str):
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
        .group_by("impression_id")
        .agg(pl.col("index"), pl.col("score"))
        .with_columns(
            pl.struct(["index", "score"])
                .map_elements(lambda x: (np.array(x["index"])[np.argsort(x["score"])[::-1]] + 1).tolist())
                .map_elements(lambda el: str(el.to_list()).replace(" ", ""))
                .alias("ranking")
        )
        .select("impression_id", "ranking")
    )

    return rankings
