import warnings

import pandas as pd
import numpy as np

try:
    import utils.utils as wb_utils
except:
    import whitebox.utils.utils as wb_utils


def autoformat_types(inputdf):
    # convert categorical dtypes to strings
    catcols = inputdf.select_dtypes(include=['category']).columns
    inputdf[catcols] = inputdf[catcols].apply(lambda x: x.astype(str))
    return inputdf


def format_inputs(input, format_dict):
    # format string
    if isinstance(input, str):
        return format_dict.get(input, input)
    # format pandas dataframe
    if isinstance(input, pd.DataFrame):
        return input.rename(columns=format_dict)
    # format list
    if isinstance(input, list):
        return [format_dict.get(list_val, list_val) for list_val in input]