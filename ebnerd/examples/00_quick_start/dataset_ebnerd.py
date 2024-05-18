# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting started with the EB-NeRD

# %%
from pathlib import Path
import polars as pl

from ebrec.utils._descriptive_analysis import (
    min_max_impression_time_behaviors, 
    min_max_impression_time_history
)
from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    truncate_history,
)
from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_USER_COL
)

# %% [markdown]
# ## Load dataset:

# %%
PATH = Path("downloads/demo")
data_split = "train"

# %%
df_behaviors = pl.scan_parquet(PATH.joinpath(data_split, "behaviors.parquet"))
df_history = pl.scan_parquet(PATH.joinpath(data_split, "history.parquet"))

# %% [markdown]
# ### Check min/max time-stamps in the data-split period

# %%
print(f"History: {min_max_impression_time_history(df_history).collect()}")
print(f"Behaviors: {min_max_impression_time_behaviors(df_behaviors).collect()}")

# %% [markdown]
# ## Add History to Behaviors

# %%
df_history = df_history.select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL).pipe(
    truncate_history,
    column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    history_size=30,
    padding_value=0,
    enable_warning=False,
)
df_history.head(5).collect()

# %%
df = slice_join_dataframes(
    df1=df_behaviors.collect(),
    df2=df_history.collect(),
    on=DEFAULT_USER_COL,
    how="left",
)
df.head(5)

# %% [markdown]
# ## Generate labels

# %% [markdown]
# Here's an example how to generate binary labels based on ``article_ids_clicked`` and ``article_ids_inview``

# %%
df.select(DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL).pipe(
    create_binary_labels_column, shuffle=True, seed=123
).with_columns(pl.col("labels").list.len().name.suffix("_len")).head(5)

# %% [markdown]
# An example using the downsample strategy employed by Wu et al.

# %%
NPRATIO = 2
df.select(DEFAULT_CLICKED_ARTICLES_COL, DEFAULT_INVIEW_ARTICLES_COL).pipe(
    sampling_strategy_wu2019, npratio=NPRATIO, shuffle=False, with_replacement=True, seed=123
).pipe(create_binary_labels_column, shuffle=True, seed=123).with_columns(pl.col("labels").list.len().name.suffix("_len")).head(5)

# %%
