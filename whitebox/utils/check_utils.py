import logging
import warnings

import pandas as pd
from sklearn.utils.validation import (check_consistent_length,
                                      check_is_fitted)

try:
    import utils.utils as wb_utils
except:
    import whitebox.utils.utils as wb_utils

class CheckInputs(object):
    @staticmethod
    def is_regression(modelobj):
        """
        determined whether users modelobj is regression or classification based on
        presence of predict_proba
        :return: NA
        """
        # determine if in classification problem or regression problem
        if hasattr(modelobj, 'predict_proba'):
            # if classification setting, secure the predicted class probabilities
            predict_engine = getattr(modelobj, 'predict_proba')
            model_type = 'classification'
        else:
            # use the regular predict function
            predict_engine = getattr(modelobj, 'predict')
            model_type = 'regression'
        return predict_engine, model_type

    @staticmethod
    def check_featuredict(featuredict, cat_df):
        """
        check user defined featuredict - if blank assign all dataframe columns
        :param featuredict: user defined featuredict mapping original col names to cleaned col names
        :return: NA
        """
        # featuredict blank
        if not featuredict:
            featuredict = {col: col for col in cat_df.columns}
        else:
            if not all([key in cat_df.columns for key in featuredict.keys()]):
                # identify missing keys
                missing = list(set(featuredict.keys()).difference(set(cat_df.columns)))
                raise ValueError(wb_utils.ErrorWarningMsgs.error_msgs['featuredict'].format(missing))
        return featuredict

    @staticmethod
    def check_agg_func(aggregate_func):
        """
        check user defined aggregate function
        :param aggregate_func: user defined aggregate function
        :return: NA
        """

        try:
            agg_results = aggregate_func(list(range(100)))
            if hasattr(agg_results, '__len__'):
                raise ValueError("""aggregate_func must return scalar""")
        except Exception as e:
            raise TypeError(wb_utils.ErrorWarningMsgs.error_msgs['agg_func'].format(e))

        return aggregate_func

    @staticmethod
    def check_verbose(verbose):
        """
        assign user defined verbosity level to logging
        :param verbose: verbosity level
        :return: NA
        """
        if verbose:

            if verbose not in [0, 1, 2]:
                raise ValueError(
                    """Verbose flag must be set to 
                    level 0, 1 or 2.
                    \nLevel 0: Debug
                    \nLevel 1: Warning
                    \nLevel 2: Info""")

            # create log dict
            log_dict = {0: logging.DEBUG,
                        1: logging.WARNING,
                        2: logging.INFO}

            logging.basicConfig(
                format="""%(asctime)s:[%(filename)s:%(lineno)s - 
                                        %(funcName)20s()]
                                        %(levelname)s:\n%(message)s""",
                level=log_dict[verbose])
            logging.info("Logger started....")

    @staticmethod
    def check_cat_df(cat_df, model_df):
        """
        ensure validity of assigned cat_df - must have same length of model_df
        and if None, is replaced by model_df
        :param value: user defined cat_df
        :return: NA
        """
        # if cat_df not assigned, use model_df
        if cat_df is None:
            warnings.warn(wb_utils.ErrorWarningMsgs.warning_msgs['cat_df'])
            cat_df = model_df
        else:
            # check both model_df and cat_df have the same length
            check_consistent_length(cat_df, model_df)
            # check index's are equal
            if not all(cat_df.index == model_df.index):
                raise ValueError("""Indices of cat_df and model_df are not aligned. Ensure Index's are 
                                            \nexactly the same before WhiteBox use.""")
            # reset users index in case of multi index or otherwise
            return cat_df

    @staticmethod
    def check_modelobj(value):
        """
        check user defined model object has been fit before use within WhiteBox
        :param value: user defined model object
        :return: NA
        """
        # basic parameter checks
        if not hasattr(value, 'predict'):
            raise ValueError(wb_utils.ErrorWarningMsgs.error_msgs['modelobj'])

        # ensure modelobj has been previously fit
        check_is_fitted(value, 'base_estimator_')

        return value
