#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import logging
import requests
import math
import pandas as pd
import numpy as np
import pkg_resources
import io


class Settings(object):
    # currently supported aggregate metrics
    supported_agg_errors = ['MSE', 'MAE', 'RMSE', 'RAW']
    # placeholder class swithc - if Sensitivity then pull in html_sensitivity code
    # if Error then pull in html_error code
    html_type = {'WhiteBoxSensitivity': 'html_sensitivity',
                 'WhiteBoxError': 'html_error'}

class ErrorWarningMsgs(object):
    # specify groupbyvars error
    groupbyvars_error = ValueError(
        """groupbyvars must be a list of grouping 
            variables and cannot be None""")

    # specify supported errosr message
    error_type_error = ValueError(
        """Supported values for error_type are [MSE, MAE, RMSE]"""
    )

    cat_df_shape_error = """cat_df and model_df must have same number of observations.
                            \ncat_df shape: {}
                            \nmodel_df shape: {}"""

    predict_model_obj_error = """modelObj does not have predict method. 
                                WhiteBoxError only works with model 
                                objects with predict method"""

    run_wb_error = """Must run {}.run() before calling save method"""

    agg_func_error = """aggregate_func must work on 
                            arrays of data and yield scalar
                            \nError: {}"""

    # hold all error messages that are raised based on value or type errors
    error_msgs = {'groupbyvars': groupbyvars_error,
                  'error_type': error_type_error,
                  'cat_df': cat_df_shape_error,
                  'modelobj': predict_model_obj_error,
                  'wb_run_error': run_wb_error,
                  'agg_func': agg_func_error}

    cat_df_warning = """model_df being used for processing. Given that most 
                        sklearn models cannot directly handle 
                        string objects and they need to be converted to numbers, 
                        the use of model_df for processing may not behave as expected. 
                        For best results, use cat_df with string columns directly"""

    auto_format = """Please note autoformat is currently experimental and may have unintended consequences."""

    warning_msgs = {'cat_df': cat_df_warning,
                    'auto_format': auto_format}




def getvectors(dataframe):
    """
    getVectors calculates the percentiles 1 through 100 for data in
    each columns of dataframe.
    For categorical data, the mode is used.
    :param dataframe: pandas dataframe object
    :return: pandas dataframe object with percentiles
    """
    # ensure dataframe is pandas dataframe object
    assert isinstance(dataframe, pd.DataFrame), """Supports pandas 
                                                    dataframe, current class 
                                                    type is {}""".format(type(dataframe))
    # calculate the percentiles for numerical data and use the mode for
    # categorical, string data
    allresults = dataframe.describe(percentiles=np.linspace(0.01, 1, num=100),
                                    include=[np.number])

    # if top (or mode) in index
    if 'top' in allresults.index:
        # Pull out the percentiles for numerical data otherwise use the mode
        tempvec = allresults.apply(lambda x: x['top'] if math.isnan(x['mean']) else x.filter(regex='[0-9]{1,2}\%',
                                                                                             axis=0), axis=0)
    else:
        tempvec = allresults.filter(regex='[0-9]{1,2}\%', axis=0)

    return tempvec


def convert_categorical_independent(dataframe):
    """
    convert pandas dtypes 'categorical' into numerical columns
    :param dataframe: dataframe to perform adjustment on
    :return: dataframe that has converted strings to numbers
    """
    # we want to change the data, not copy and change
    dataframe = dataframe.copy(deep=True)
    # convert all strings to categories and format codes
    for str_col in dataframe.select_dtypes(include=['O', 'category']):
        dataframe.loc[:, str_col] =pd.Categorical(dataframe.loc[:, str_col])
    # convert all category datatypes into numeric
    cats = dataframe.select_dtypes(include=['category'])
    # warn user if no categorical variables detected
    if cats.shape[1] == 0:
        logging.warn("""Pandas categorical variable types not detected""")
        warnings.warn('Pandas categorical variable types not detected', UserWarning)
    # iterate over these columns
    for category in cats.columns:
        dataframe.loc[:, category] = dataframe.loc[:, category].cat.codes

    return dataframe


def create_insights(
                    group,
                    group_var=None,
                    error_type='MSE'):
    """
    create_insights develops various error metrics such as MSE, RMSE, MAE, etc.
    :param group: the grouping object from the pandas groupby
    :param group_var: the column that is being grouped on
    :return: dataframe with error metrics
    """
    assert error_type in ['MSE', 'RMSE', 'MAE', 'RAW'], """Currently only supports
                                                 MAE, MSE, RMSE, RAW"""
    errors = group['errors']
    error_dict = {'MSE': np.mean(errors ** 2),
                  'RMSE': (np.mean(errors ** 2)) ** (1 / 2),
                  'MAE': np.sum(np.absolute(errors))/group.shape[0],
                  'RAW': np.mean(errors)}

    msedf = pd.DataFrame({'groupByValue': group.name,
                          'groupByVarName': group_var,
                          error_type: error_dict[error_type],
                          'Total': float(group.shape[0])}, index=[0])
    return msedf

def to_json(
                dataframe,
                vartype='Continuous',
                html_type='error',
                incremental_val=None):
    # convert dataframe values into a json like object for D3 consumption
    assert vartype in ['Continuous', 'Categorical', 'Accuracy','Percentile'], """Vartypes should only be continuous, 
                                                                                categorical,
                                                                                Percentile or accuracy"""
    assert html_type in ['error', 'sensitivity',
                         'percentile'], 'html_type must be error or sensitivity'
    # prepare for error
    if html_type in ['error', 'percentile']:
        # specify data type
        json_out = {'Type': vartype}
    # prepare for sensitivity
    if html_type == 'sensitivity':
        # convert incremental_val
        if isinstance(incremental_val, float):
            incremental_val = round(incremental_val, 2)
        json_out = {'Type': vartype,
                    'Change': str(incremental_val)}
    # create data records from values in df
    json_out['Data'] = dataframe.to_dict(orient='records')

    return json_out


def create_group_percentiles(df,
                             groupbyvars,
                             wanted_percentiles=[0, .01, .1, .25, .5, .75, .9, 1]):
    """
    create percentile buckets for based on groupby for numeric columns
    :param df: dataframe
    :param groupbyvars: groupby variable list
    :param wanted_percentiles: desired percnetile lines for user intereface
    :return: json formatted percentile outputs
    """
    groupbyvars = list(groupbyvars)
    # subset numeric cols
    num_cols = df.select_dtypes(include=[np.number])
    final_out = {'Type': 'PercentileGroup'}
    final_list = []
    # iterate over
    for col in num_cols:
        data_out = {'variable': col}
        groupbylist = []
        # iterate groupbys
        for group in groupbyvars:
            # iterate over each slice of the groups
            for name, group in df.groupby(group):
                # get col of interest
                group = group.loc[:, col]
                # start data out for group
                group_out = {'groupByVar': name}
                # capture wanted percentiles
                group_percent = group.quantile(wanted_percentiles).reset_index().rename(columns = {'index': 'percentiles',                                                                         col: 'value'})
                # readjust percentiles to look nice
                group_percent.loc[:, 'percentiles'] = group_percent.loc[:, 'percentiles'].apply(lambda x: str(int(x*100))+'%')
                # convert percnetile dataframe into json format
                group_out['percentileValues'] = group_percent.to_dict(orient='records')
                # append group out to group placeholder list
                groupbylist.append(group_out)
        # assign groupbylist out
        data_out['percentileList'] = groupbylist
        final_list.append(data_out)
    final_out['Data'] = final_list
    return final_out


def flatten_json(dictlist):
    """
    flatten lists of dictionaries of the same variable into one dict
    structure. Inputs: [{'Type': 'Continuous', 'Data': [fixed.acid: 1, ...]},
    {'Type': 'Continuous', 'Data': [fixed.acid : 2, ...]}]
    outputs: {'Type' : 'Continuous', 'Data' : [fixed.acid: 1, fixed.acid: 2]}}
    :param dictlist: current list of dictionaries containing certain column elements
    :return: flattened structure with column variable as key
    """
    # make copy of dictlist
    copydict = dictlist[:]
    if len(copydict) > 1:
        for val in copydict[1:]:
            copydict[0]['Data'].extend(val['Data'])
        # take the revised first element of the list
        toreturn = copydict[0]
    else:
        if isinstance(copydict, list):
            # return the dictionary object if list type
            toreturn = copydict[0]
        else:
            # else return the dictionary itself
            toreturn = copydict
    assert isinstance(toreturn, dict), """flatten_json output object not of class dict.
                                        \nOutput class type: {}""".format(type(toreturn))
    return toreturn

def prob_acc(true_class=0, pred_prob=0.2):
    """
    return the prediction error
    :param true_class: true class label (0 or 1)
    :param pred_prob: predicted probability
    :return: error
    """
    return (true_class * (1-pred_prob)) + ((1-true_class)*pred_prob)


class HTML(object):
    @staticmethod
    def get_html(htmltype='html_error'):
        assert htmltype in ['html_error', 'html_sensitivity'], 'htmltype must be html_error or html_sensitivity'
        html_path = pkg_resources.resource_filename('whitebox', '{}.txt'.format(htmltype))
        # utility class to hold whitebox files
        try:
            wbox_html = open('{}.txt'.format(htmltype), 'r').read()
        except IOError:
            wbox_html = open(html_path, 'r').read()
        return wbox_html

def createmlerror_html(
                        datastring,
                        dependentvar,
                        htmltype='html_error'):
    """
    create WhiteBox error plot html code
    :param datastring: json like object containing data
    :param dependentvar: name of dependent variable
    :return: html string
    """
    assert htmltype in ['html_error', 'html_sensitivity'], """htmltype must be html_error 
                                                                or html_sensitivity"""
    output = HTML.get_html(htmltype=htmltype).replace('<***>',
                                                        datastring
                                                        ).replace('Quality', dependentvar)

    return output

def create_wine_data(cat_cols):
    """
    helper function to grab UCI machine learning wine dataset, convert to
    pandas dataframe, and return
    :return: pandas dataframe
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
