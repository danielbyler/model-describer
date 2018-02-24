#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
import logging

import numpy as np
import pandas as pd

try:
    import utils.utils as wb_utils
    import utils.check_utils as checks
    import utils.percentiles as percentiles
    import utils.formatting as formatting
    import modelconfig.fmt_sklearn_preds as fmt_sklearn_preds
except ImportError:
    import whitebox.utils.utils as wb_utils
    import whitebox.utils.check_utils as checks
    import whitebox.utils.percentiles as percentiles
    import whitebox.utils.formatting as formatting
    from whitebox.utils.fmt_model_outputs import fmt_sklearn_preds


class WhiteBoxBase(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(
                    self,
                    modelobj,
                    model_df,
                    ydepend,
                    cat_df=None,
                    keepfeaturelist=None,
                    groupbyvars=None,
                    aggregate_func=np.mean,
                    error_type='RMSE',
                    autoformat_types=False,
                    verbose=None):
        """
        WhiteBox base class instantiation and parameter checking

        :param modelobj: fitted sklearn model object
        :param model_df: dataframe used for training sklearn object
        :param ydepend: str dependent variable
        :param cat_df: dataframe formatted with categorical dtypes specified,
                       and non-dummy categories
        :param keepfeaturelist: list of features to keep in output
        :param groupbyvars: list of groupby variables
        :param error_type: str aggregate error metric i.e. MSE, MAE, RMSE, MED, MEAN
                MSE - Mean Squared Error
                MAE - Mean Absolute Error
                RMSE - Root Mean Squared Error
                MED - Median Error
                MEAN - Mean Error
        :param autoformat: experimental feature for formatting dataframe columns and dtypes
        :param verbose: set verbose level -- 0 = debug, 1 = warning, 2 = error
        """

        # check error type is supported format
        if error_type not in wb_utils.Settings.supported_agg_errors:
            raise wb_utils.ErrorWarningMsgs.error_msgs['error_type']

            # check groupby vars
        if not groupbyvars:
            raise wb_utils.ErrorWarningMsgs.error_msgs['groupbyvars']

        # make copy, reset index and assign model dataframe
        self._model_df = model_df.copy(deep=True).reset_index(drop=True)
        # check cat_df
        cat_df = checks.CheckInputs.check_cat_df(cat_df.copy(deep=True).reset_index(drop=True),
                                                        self._model_df)

        # check keepfeaturelist
        self.keepfeaturelist = checks.CheckInputs.check_keepfeaturelist(keepfeaturelist, cat_df)

        # subset dataframe down based on user input
        self._cat_df = formatting.subset_input(cat_df,
                                               self.keepfeaturelist,
                                               ydepend)

        # check modelobj
        self._modelobj = checks.CheckInputs.check_modelobj(modelobj)
        # check verbosity
        if verbose:
            checks.CheckInputs.check_verbose(verbose)
        # check for classification or regression
        self.predict_engine, self.model_type = checks.CheckInputs.is_regression(modelobj)

        # check aggregate func
        self.aggregate_func = checks.CheckInputs.check_agg_func(aggregate_func)
        # check error type supported
        self.error_type = error_type
        # assign dependent variable
        self.ydepend = ydepend
        # if user specified keepfeaturelist, use column mappings otherwise use original groupby
        self.groupbyvars = groupbyvars
        # determine the calling class (WhiteBoxError or WhiteBoxSensitivity)
        self.called_class = self.__class__.__name__
        # create percentiles
        self.Percentiles = percentiles.Percentiles(self._cat_df,
                                                   self.groupbyvars)
        # get population percentiles
        self.Percentiles.population_percentiles()

        if autoformat_types:
            self._cat_df = formatting.autoformat_types(self._cat_df)

        # store results
        self.agg_df = pd.DataFrame()
        # raw results
        self.raw_df = pd.DataFrame()

    @property
    def modelobj(self):
        return self._modelobj

    @property
    def cat_df(self):
        return self._cat_df

    @property
    def model_df(self):
        return self._model_df

    @abstractmethod
    def _transform_function(self,
                            group,
                            groupby_var=None,
                            col=None,
                            vartype='Continuous'):
        # method to operate on slices of data within groups
        pass

    @abstractmethod
    def _var_check(
                    self,
                    col=None,
                    groupby_var=None):
        """
        _var_check tests for categorical or continuous variable and performs operations
        dependent upon the var type
        :param col: current column being operated on
        :param groupby: current groupby level
        :return: NA
        """
        pass

    #TODO add weighted schematics function
    def _continuous_slice(
                        self,
                        group,
                        col=None,
                        groupby_var=None,
                        vartype='Continuous'):
        """
        _continuous_slice operates on portions of the data that correspond
        to a particular group of data from the groupby
        variable. For instance, if working on the wine quality dataset with
        Type representing your groupby variable, then
        _continuous_slice would operate on 'White' wine data
        :param group: slice of data with columns, etc.
        :param col: current continuous variable being operated on
        :param vartype: continuous
        :return: transformed data with col data max, errPos mean, errNeg mean,
                and prediction means for this group
        """
        # check right vartype
        assert vartype in ['Continuous', 'Categorical'], """Vartype must be 
                            Categorical or Continuous"""
        continuous_group = group.reset_index(drop=True).copy(deep=True)
        # pull out col being operated on
        group_col_vals = continuous_group.loc[:, col]
        # if more than 100 values in the group, use percentile bins
        if group.shape[0] > 100:
            logging.info("""Creating percentile bins for current 
                            continuous grouping""")
            # digitize needs monotonically increasing vector
            group_percentiles = sorted(list(set(percentiles.create_percentile_vecs(group_col_vals))))
            # create percentiles for slice
            continuous_group['fixed_bins'] = np.digitize(group_col_vals,
                                              group_percentiles,
                                              right=True)
        else:
            logging.warning("""Slice of data less than 100 continuous observations, 
                                using raw data as opposed to percentile groups.
                                \nGroup size: {}""".format(group.shape))
            continuous_group['fixed_bins'] = group_col_vals

        logging.info("""Applying transform function to continuous bins""")
        # group by bins
        errors_out = continuous_group.groupby('fixed_bins').apply(self._transform_function,
                                                                  col=col,
                                                                  groupby_var=groupby_var,
                                                                  vartype=vartype)
        return errors_out

    def run(self,
            output_type=None,
            output_path=''):
        """
        main run engine. Iterate over columns specified in keepfeaturelist,
        and perform anlaysis
        :param output_type - output type - current support for html
        :param output_path - fpath to save output
        :return: None - does put outputs in place
        """
        # ensure supported output types
        supported = ['html', None]
        if output_type not in supported:
            raise ValueError("""Output type {} not supported.
                                \nCurrently support {} output""".format(output_type, supported))
        # run the prediction function first to assign the errors to the dataframe
        self._cat_df, self._model_df = fmt_sklearn_preds(self.predict_engine,
                                                       self.modelobj,
                                                       self._model_df,
                                                       self._cat_df,
                                                       self.ydepend,
                                                       self.model_type)
        # create placeholder for outputs
        placeholder = []
        # create placeholder for all insights
        insights_df = pd.DataFrame()
        logging.info("""Running main program. Iterating over 
                    columns and applying functions depednent on datatype""")


        not_in_cols = ['errors', 'predictedYSmooth', self.ydepend]
        # filter columns to iterate through
        to_iter_cols = self._cat_df.columns[~self._cat_df.columns.isin(not_in_cols)]
        # iterate over each col
        for col in to_iter_cols:

            # column placeholder
            colhold = []

            for groupby_var in self.groupbyvars:
                # if current column is the groupby variable,
                # create error metrics
                if col != groupby_var:
                    json_out = self._var_check(
                                                col=col,
                                                groupby_var=groupby_var)
                    # append to placeholder
                    colhold.append(json_out)

                else:
                    logging.info(
                                """Creating accuracy metric for 
                                groupby variable: {}""".format(groupby_var))
                    # create error metrics for slices of groupby data
                    acc = wb_utils.create_accuracy(self.model_type,
                                                   self._cat_df,
                                                   self.error_type,
                                                   groupby=groupby_var)
                    # append to insights dataframe placeholder
                    insights_df = insights_df.append(acc)

            # map all of the same columns errors to the first element and
            # append to placeholder
            # dont append if placeholder is empty due to col being the same as groupby
            if len(colhold) > 0:
                placeholder.append(formatting.FmtJson.flatten_json(colhold))

        logging.info('Converting accuracy outputs to json format')
        # finally convert insights_df into json object
        insights_json = formatting.FmtJson.to_json(insights_df,
                                                   html_type='accuracy',
                                                   vartype='Accuracy',
                                                   err_type=self.error_type)
        # append to outputs
        placeholder.append(insights_json)
        # append percentiles
        placeholder.append(self.Percentiles.percentiles)
        # append groupby percnetiles
        placeholder.append(self.Percentiles.group_percentiles_out)
        # assign placeholder final outputs to class instance
        self.outputs = placeholder
        # save outputs if specified
        if output_type == 'html':
            self._save(fpath=output_path)

    def _save(self, fpath=''):
        """
        save html output to disc
        :param fpath: file path to save html file to
        :return: None
        """
        logging.info("""creating html output for type: {}""".format(wb_utils.Settings.html_type[self.called_class]))

        # tweak self.ydepend if classification case (add dominate class)
        if self.model_type == 'classification':
            ydepend_out = '{}: {}'.format(self.ydepend, self._modelobj.classes_[1])
        else:
            # regression case
            ydepend_out = self.ydepend

        # create HTML output
        html_out = formatting.HTML.fmt_html_out(
                                            str(self.outputs),
                                            ydepend_out,
                                            htmltype=wb_utils.Settings.html_type[self.called_class])
        # save html_out to disk
        with open(fpath, 'w') as outfile:
            logging.info("""Writing html file out to disk""")
            outfile.write(html_out)
