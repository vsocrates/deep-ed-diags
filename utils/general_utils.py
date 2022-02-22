import pandas as pd
import numpy as np


def column_index(df, query_cols):
    """Returns the idxs of the list of strs in query_cols that are in df

    NOTE: They must exist, or else we'll get an error
    """
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


def flatten(t):
    return [item for sublist in t for item in sublist]
