#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import io
from sklearn.datasets import make_blobs, make_regression
import random
import logging


class Settings(object):
    """ module wide parameter settings """
    supported_agg_errors = ['MSE', 'MAE', 'RMSE', 'MEAN', 'MED']
    # placeholder class swithc - if Sensitivity then pull in html_sensitivity code
    # if Error then pull in html_error code
    html_type = {'WhiteBoxSensitivity': 'html_sensitivity',
                 'WhiteBoxError': 'html_error'}

    # define desired output percentiles
    output_percentiles = [0, .01, .1, .25, .5, .75, .9, 1]
    # formatted percentiles
    formatted_percentiles = [int(percent * 100) for percent in output_percentiles]
    # remove 100th percentile if present for percentiles._percentiles_out
    fmt_percentiles_out = [percent for percent in formatted_percentiles if percent != 100]
    # specify supported output types
    supported_out_types = ['html', 'raw_data', 'agg_data', None]
    # setup log verbose lookup
    verbose2log = {None: logging.NOTSET,
                   0: logging.DEBUG,
                   1: logging.WARNING,
                   2: logging.ERROR}


class ErrorWarningMsgs(object):
    """ module wide error and warning messages """
    # specify groupbyvars error
    groupbyvars_error = ValueError(
        """groupbyvars must be a list of grouping 
            variables and cannot be None""")

    # specify supported errors message
    error_type_error = ValueError(
        """Supported values for error_type are [MSE, MAE, RMSE]"""
    )

    # raise cat_df error if model shapes don't align
    cat_df_shape_error = """cat_df and model_df must have same number of observations.
                            \ncat_df shape: {}
                            \nmodel_df shape: {}"""
    # modelobj prediction method error
    predict_model_obj_error = """modelObj does not have predict method. 
                                WhiteBoxError only works with model 
                                objects with predict method"""

    # missing keepfeaturelist error message
    missing_keepfeaturelist = """featuredict keys missing from assigned cat_df
                                    \ncheck featuredict keys and reassign.
                                    \nMissing keys: {}"""

    # run wb error before calling .save error
    run_wb_error = """Must run {}.run() before calling save method"""

    # user defined aggregation function error
    agg_func_error = """aggregate_func must work on 
                            arrays of data and yield scalar
                            \nError: {}"""

    # hold all error messages that are raised based on value or type errors
    error_msgs = {'groupbyvars': groupbyvars_error,
                  'error_type': error_type_error,
                  'cat_df': cat_df_shape_error,
                  'modelobj': predict_model_obj_error,
                  'wb_run_error': run_wb_error,
                  'agg_func': agg_func_error,
                  'keepfeaturelist': missing_keepfeaturelist}

    # convert category dtypes to object dtypes warning message
    cat_df_warning = """model_df being used for processing. Given that most 
                        sklearn models cannot directly handle 
                        string objects and they need to be converted to numbers, 
                        the use of model_df for processing may not behave as expected. 
                        For best results, use cat_df with string columns directly"""

    # autoformat usage warning message
    auto_format = """Please note autoformat is currently experimental and may have unintended consequences."""

    warning_msgs = {'cat_df': cat_df_warning,
                    'auto_format': auto_format}


def prob_acc(true_class=0, pred_prob=0.2):
    """
    return classification prediction accuracy

    :param true_class: integer containing true label 0 or 1
    :param pred_prob: float - predicted probability
    :return scalar - prediction accuracy
    :rtype float
    """
    return (true_class * (1-pred_prob)) + ((1-true_class)*pred_prob)


def create_wine_data(cat_cols):
    """
    create UCI wine machine learning dataset
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality

    :param cat_cols: columns to convert to categories
    :return UCI wine machine learning dataset
    :rtype pd.DataFrame
    """

    if not cat_cols:
        cat_cols = ['alcohol', 'fixed acidity']

    red_raw = requests.get(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv').content
    red = pd.read_csv(io.StringIO(red_raw.decode('utf-8-sig')),
                      sep=';')
    red['Type'] = 'Red'

    white_raw = requests.get(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv').content
    white = pd.read_csv(io.StringIO(white_raw.decode('utf-8-sig')),
                        sep=';')
    white['Type'] = 'White'

    # read in wine quality dataset
    wine = pd.concat([white, red])

    # create category columns
    # create categories
    for cat in cat_cols:
        wine.loc[:, cat] = pd.cut(wine.loc[:, cat], bins=3, labels=['low', 'medium', 'high'])

    return wine


def create_synthetic(nrows=100,
                     ncols=10,
                     ncat=5,
                     num_groupby=3,
                     max_levels=10,
                     mod_type='regression'):
    """
    synthetic dataset creation for testing whitebox

    :param nrows: int number of observations
    :param ncols: int number of features
    :param ncat: int number of categories
    :param num_groupby: int number of groupby variables
    :param max_levels: int number of max bin levels
    :param mod_type: str specifying modeling dataset type (regression, classification)
    :return synthetic dataset
    :rtype pd.DataFrame
    """

    if mod_type == 'classification':
        df = pd.DataFrame(make_blobs(n_samples=nrows,
                                     n_features=ncols,
                                     random_state=5)[0])
    else:
        df = pd.DataFrame(make_regression(n_samples=nrows,
                                          n_features=ncols,
                                          random_state=5)[0])

    cols = ['col{}'.format(idx) for idx in list(range(ncols))]

    df.columns = cols

    # reserve col0 for target
    cols = cols[1:]
    # randomly select ncat cols
    cats = list(set([random.choice(cols) for _ in range(ncat)]))

    for cat in cats:
        num_bins = max(1, random.choice(list(range(max_levels))))
        bin_labels = ['level_{}'.format(level) for level in list(range(num_bins))]
        df.loc[:, cat] = pd.cut(df.loc[:, cat], bins=num_bins,
                                labels=bin_labels)
        df.loc[:, cat] = df.loc[:, cat].astype(str)

    if mod_type == 'classification':
        df.loc[:, 'col0'] = pd.cut(df.loc[:, 'col0'], bins=2,
                                   labels=[0, 1])
        df.loc[:, 'col0'] = df.loc[:, 'col0'].astype(int)

    df.rename(columns={'col0': 'target'}, inplace=True)

    if not num_groupby:
        num_groupby = max(1, random.choice(list(range(ncat))))

    catcols = df.loc[:, df.columns != 'target'].select_dtypes(include=['O']).columns.values.tolist()

    random.shuffle(catcols)

    groupby = catcols[0:num_groupby]

    return 'target', groupby, df


def create_insights(
                    group,
                    group_var=None,
                    error_type='RMSE'):
    """
    aggregates user specified error metric from raw errors

    :param group: dataframe containing errors
    :param group_var: str specificying groupby variable
    :param error_type: str specifying error metric
    :return error metric dataframe
    :rtype pd.DataFrame
    """
    assert error_type in Settings.supported_agg_errors, """{} unspported error type""".format(error_type)
    errors = group['errors']
    error_dict = {'MSE': np.mean(errors ** 2),
                  'RMSE': (np.mean(errors ** 2)) ** (1 / 2),
                  'MAE': np.sum(np.absolute(errors))/group.shape[0],
                  'MEAN': np.mean(errors),
                  'MED': np.median(errors)}

    msedf = pd.DataFrame({'groupByValue': group.name,
                          'groupByVarName': group_var,
                          error_type: error_dict[error_type],
                          'Total': float(group.shape[0])}, index=[0])
    return msedf


def create_accuracy(model_type,
                    cat_df,
                    error_type,
                    groupby=None):
    """
    create error metric results by groupby variable

    :param model_type: str specifying model type used (regression, classification)
    :param cat_df: pd.DataFrame specifying formatting dataframe
    :param error_type: str specifying error metric (RMSE, MSE, MAE)
    :param groupby: str specifying groupby variable
    :return accuracy dataset by groupby variable
    :rtype pd.DataFrame
    """
    # use this as an opportunity to capture error metrics for the groupby variable
    if model_type == 'classification':
        error_type = 'MEAN'

    acc = cat_df.groupby(groupby).apply(create_insights,
                                        group_var=groupby,
                                        error_type=error_type)
    # drop the grouping indexing
    acc.reset_index(drop=True, inplace=True)
    # append to insights_df
    return acc

def util_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('%asctime)s %(name)-12s %(levelname)-8s %(message)s')
    formatter = logging.Formatter("""%(asctime)s:[%(filename)s:%(lineno)s - 
                                        %(funcName)20s()]
                                        %(levelname)s:\n%(message)s""")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger