import polars as pl
import numpy as np


def group_scores(prediction_file: str, index_file: str):
    """Combine the prediction file (contains only a single column of scores) with the previously generated index file, which contains a column that specifies to which row of the original testfile a score belongs and the impression id.
    The resulting LazyFrame of the concatenation is then grouped by the impression_id, where all scores of the same impression id are concatenated, resulting in the same layout as the original test file with the same order of rows.
    """
    grouped_scores = (
        pl.concat(
            [
                pl.scan_csv(prediction_file, has_header=False, new_columns=["score"]),
                pl.scan_parquet(index_file),
            ],
            how="horizontal",
        )
        .group_by(["index", "impression_id"])
        .agg(pl.col("score"))
        .sort("index")
        .drop("index")
    )

    return grouped_scores


def calculate_rankings(grouped_scores: pl.LazyFrame):
    """Calculate the rankings for the given score column and return the resulting LazyFrame which
    has the correct schema for a handin at the RecSys challenge (columns: impression_id, ranking).
    """
    rankings = (
        grouped_scores.with_columns(
            pl.col("score")
            .map_elements(
                lambda scores: [
                    rank + 1
                    for rank, index in sorted(
                        enumerate(np.argsort(np.array(scores))[::-1]),
                        key=lambda el: el[1],
                    )
                ]
            )
            .alias("ranking")
        )
        .drop("score")
        .with_columns(
            pl.col("ranking").map_elements(
                lambda el: str(el.to_list()).replace(" ", "")
            )
        )
        .select("impression_id", "ranking")
    )

    return rankings
