#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta
import logging
import sys

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

logger = wb_utils.util_logger(__name__)

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
                    aggregate_func=np.nanmedian,
                    error_type='RMSE',
                    autoformat_types=False,
                    round_num=2,
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
        :param autoformat_types: boolean to auto format categorical columns to objects
        :param round_num: round numeric columns to specified level for output
        :param verbose: set verbose level -- 0 = debug, 1 = warning, 2 = error
        """
        logger.setLevel(wb_utils.Settings.verbose2log[verbose])
        logger.info('Initilizing {} parameters'.format(self.__class__.__name__))
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
        self._keepfeaturelist = checks.CheckInputs.check_keepfeaturelist(keepfeaturelist, cat_df)

        # subset dataframe down based on user input
        self._cat_df = formatting.subset_input(cat_df,
                                               self._keepfeaturelist,
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
                                                   self.groupbyvars,
                                                   round_num=round_num)
        # get population percentiles
        self.Percentiles.population_percentiles()

        if autoformat_types is True:
            self._cat_df = formatting.autoformat_types(self._cat_df)

        self.agg_df = pd.DataFrame()
        self.raw_df = pd.DataFrame()
        self.round_num = round_num

    @property
    def modelobj(self):
        return self._modelobj

    @property
    def cat_df(self):
        return self._cat_df

    @property
    def model_df(self):
        return self._model_df
    
    @property
    def keepfeaturelist(self):
        return self._keepfeaturelist

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
        :param groupby_var: current groupby level
        """
        pass

    # TODO add weighted schematics function
    def _continuous_slice(
                        self,
                        group,
                        col=None,
                        groupby_var=None):
        """
        _continuous_slice operates on portions of continuous data that correspond
        to a particular group specified by groupby. If current group
        has more than 100 observations, create 100 percentile buckets.
        Otherwise, use raw data values. Data is only assigned to percentile bucket if
        data actually exists. In cases where there are big gaps between data points,
        even with more than 100 observations, there could be far less percentile buckets
        that align with the actual data.


        :param group: slice of data with columns, etc.
        :param col: current continuous variable being operated on
        :param groupby_var: str current groupby variable
        :return: transformed data with col data max, errPos mean, errNeg mean,
                and prediction means for this group
        :rtype: pd.DataFrame
        """
        logger.info("""Creating percentile bins
                    \nCol: {}
                    \nGroupby_var: {}
                    \nGroup shape: {}""".format(col, groupby_var, group.shape))

        continuous_group = group.reset_index(drop=True).copy(deep=True)
        # pull out col being operated on
        group_col_vals = continuous_group.loc[:, col]
        # if more than 100 values in the group, use percentile bins
        if group.shape[0] > 100:
            # digitize needs monotonically increasing vector
            group_percentiles = sorted(list(set(percentiles.create_percentile_vecs(group_col_vals))))
            # create percentiles for slice
            continuous_group['fixed_bins'] = np.digitize(group_col_vals, 
                                                         group_percentiles, 
                                                         right=True)
        else:
            logger.info("""Slice of data less than 100 continuous observations, 
                                using raw data as opposed to percentile groups - Group size: {}""".format(group.shape))
            continuous_group['fixed_bins'] = group_col_vals

        logger.info("""Applying transform function to continuous bins""")
        # group by bins
        errors_out = continuous_group.groupby('fixed_bins').apply(self._transform_function,
                                                                  col=col,
                                                                  groupby_var=groupby_var,
                                                                  vartype='Continuous')
        return errors_out

    def _create_preds(self,
                      df):
        """
        create predictions based on model type. If classification, take corresponding
        probabilities related to the 1st class. If regression, take predictions as is.

        :param df: input dataframe
        :return: predictions
        :rtype: list
        """
        if self.model_type == 'classification':
            preds = self.predict_engine(df)[:, 1]
        else:
            preds = self.predict_engine(df)

        return preds

    def _fmt_raw_df(self,
                    col,
                    groupby_var,
                    cur_group):
        """
        format raw_df by renaming various columns and appending to instance
        raw_df

        :param col: str current col being operated on
        :param groupby_var: str current groupby variable
        :param cur_group: current slice of group data
        :return: Formatted dataframe
        :rtype: pd.DataFrame
        """
        logger.info("""Formatting specifications - col: {} - groupbvy_var: {} - cur_group shape: {}""".format(col, groupby_var, cur_group.shape))

        # take copy
        group_copy = cur_group.copy(deep=True)
        # reformat current slice of data for raw_df
        raw_df = group_copy.rename(columns={col: 'col_value',
                                            groupby_var: 'groupby_level'}).reset_index(drop=True)

        raw_df['groupByVar'] = groupby_var
        raw_df['col_name'] = col
        if 'fixed_bins' in raw_df.columns:
            del raw_df['fixed_bins']

        # self.raw_df = self.raw_df.append(raw_df)
        self.raw_df = pd.concat([self.raw_df, raw_df])

    def _fmt_agg_df(self,
                    col,
                    agg_errors):
        """
        format agg_df by renaming various columns and appending to
        instance agg_df

        :param col: str current col being operated on
        :param agg_errors: aggregated error data for group
        :return: Formatted dataframe
        :rtype: pd.DataFrame
        """

        logger.info("""Formatting specifications - col: {} - agg_errors shape: {}""".format(col, agg_errors.shape))

        group_copy = agg_errors.copy(deep=True)
        debug_df = group_copy.rename(columns={col: 'col_value'})
        debug_df['col_name'] = col
        # self.agg_df = self.agg_df.append(debug_df)
        self.agg_df = pd.concat([self.agg_df, debug_df])

    def run(self,
            output_type='html',
            output_path=''):
        """
        main run engine. Iterate over columns specified in keepfeaturelist,
        and perform anlaysis
        :param output_type: str output type:
                html - save html to output_path
                raw_data - return raw analysis dataframe
                agg_data - return aggregate analysis dataframe
        :param output_path: - fpath to save output
        :return: pd.DataFrame or saved html output
        :rtype: pd.DataFrame or .html
        """
        # ensure supported output types
        if output_type not in wb_utils.Settings.supported_out_types:
            error_out = """Output type {} not supported.
                                \nCurrently support {} output""".format(output_type,
                                                                        wb_utils.Settings.supported_out_types)

            logger.error(error_out)

            raise ValueError(error_out)

        # run the prediction function first to assign the errors to the dataframe
        self._cat_df = fmt_sklearn_preds(self.predict_engine,
                                         self.modelobj,
                                         self._model_df,
                                         self._cat_df,
                                         self.ydepend,
                                         self.model_type)
        # create placeholder for outputs
        placeholder = []
        # create placeholder for all insights
        insights_list = []
        logging.info("""Running main program. Iterating over 
                    columns and applying functions depednent on datatype""")

        not_in_cols = ['errors', 'predictedYSmooth', self.ydepend]

        # filter columns to iterate through
        to_iter_cols = self._cat_df.columns[~self._cat_df.columns.isin(not_in_cols)]

        for idx, col in enumerate(to_iter_cols):

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
                    insights_list.append(acc)

                logger.info("""Run processed - Col: {} - groupby_var: {}""".format(col, groupby_var))

            # map all of the same columns errors to the first element and
            # append to placeholder
            # dont append if placeholder is empty due to col being the same as groupby
            if len(colhold) > 0:
                placeholder.append(formatting.FmtJson.flatten_json(colhold))
            # TODO redirect stdout so progress bar can output to single line
            wb_utils.sysprint('Percent Complete: {per:2.0f}%'.format(per=(idx/len(to_iter_cols))*100))

        wb_utils.sysprint('Percent Complete: 100%')
        logging.info('Converting accuracy outputs to json format')
        # finally convert insights_df into json object
        # convert insights list to dataframe
        insights_df = pd.concat(insights_list)
        insights_json = formatting.FmtJson.to_json(insights_df.round(self.round_num),
                                                   html_type='accuracy',
                                                   vartype='Accuracy',
                                                   err_type=self.error_type,
                                                   ydepend=self.ydepend)
        # append to outputs
        placeholder.append(insights_json)
        # append percentiles
        placeholder.append(self.Percentiles.percentiles)
        # append groupby percentiles
        placeholder.append(self.Percentiles.group_percentiles_out)
        # assign placeholder final outputs to class instance
        self.outputs = placeholder
        # save outputs if specified
        if output_type == 'html':
            self._save(fpath=output_path)
        elif output_type == 'raw_data':
            return self.get_raw_df()
        elif output_type == 'agg_data':
            return self.get_agg_df()

    def get_raw_df(self):
        """
        return unaggregated analysis dataframe

        :return: unaggregated analysis dataframe
        :rtype: pd.DataFrame
        """
        if not hasattr(self, 'outputs'):
            raise RuntimeError(""".run() must be called before returning analysis data""")

        return self.raw_df.reset_index()

    def get_agg_df(self):
        """
        return aggregated analysis dataframe

        :return: aggregated analysis dataframe
        :rtype: pd.DataFrame
        """
        if not hasattr(self, 'outputs'):
            raise RuntimeError(""".run() must be called before returning analysis data""")

        return self.agg_df.reset_index()

    def _save(self, fpath=''):
        """
        save html output to disc
        
        :param fpath: file path to save html file to
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

    def __str__(self):
        """
        Print readable version of class and params
        """
        header = '{name}:\n'.format(name=self.__class__.__name__)
        footer = '\n '.join(['{k}: {v}'.format(k=key, v=self.__dict__.get(key)) for key in self.__dict__ if not isinstance(self.__dict__.get(key),
                                                                                                                         pd.DataFrame)])
        return header + footer
