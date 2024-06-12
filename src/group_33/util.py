import polars as pl
from polars.lazyframe import LazyFrame


def train_test_split(df: LazyFrame, fraction_train: float = 0.8) -> tuple[LazyFrame, LazyFrame]:
    df = df.with_columns(pl.all().shuffle(seed=1)).with_row_count()
    df_train = df.filter(pl.col("row_nr") < pl.col("row_nr").max() * fraction_train).drop("row_nr")
    df_test = df.filter(pl.col("row_nr") >= pl.col("row_nr").max() * fraction_train).drop("row_nr")

    return df_train, df_test
