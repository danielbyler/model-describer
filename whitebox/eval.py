#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype

try:
    import utils.utils as wb_utils
    from utils.categorical_conversions import pandas_switch_modal_dummy
    import utils.formatting as formatting
    from base import WhiteBoxBase
except ImportError:
    import whitebox.utils.utils as wb_utils
    from whitebox.utils.categorical_conversions import pandas_switch_modal_dummy
    from whitebox.base import WhiteBoxBase
    from whitebox.utils import formatting


class WhiteBoxError(WhiteBoxBase):

    """
    Error model analysis.

    In the continuous case with over 100 datapoints for a particular slice of data,
    calculate percentiles of group of shrink data to 100 datapoints for scalability.
    Calculate average errors within this region of the data for positive and negative errors.

    In the categorical case, calculate average positive/negative error within region of data

    Parameters

    ----------

    modelobj : sklearn model object
        Used to create predictions for synthetic data

    model_df : pandas DataFrame
        Original raw dataset used to train and calibrate modelobj. This can
        and should in most cases include dummy variables for categorical data columns.

    ydepend : str
        Y dependent variable used to build modelobj

    cat_df : pandas DataFrame
        Raw, unadjusted dataframe with categorical columns converted to pandas
        data type Categorical. These categorical designations are picked up throughout
        and are important for proper functioning of WhiteBoxSensitvitiy

    keepfeaturelist : list
        Optional user defined list to subset
        the outputs to only columns defined in keepfeaturelist

    groupbyvars : list
        grouping variables to analyze impact of model for various groups of data. I.e.
        if working on dataset with red and white wine, we can disambiguate how sensitive
        the model is to changes in data for each type of wine

    aggregate_func : function
        function to perform aggregate function to groups of data pertaining to error
        analysis. I.e. take the median model error for groups data.

    verbose : int
        Logging level

    See also

    ------------

    WhiteBoxSensitivity : analyze how the model errors are doing for various groups of data
    WhiteBoxBase : base class inherited from WhiteBoxSensitivity to perform key functionality
        and grouping logic
    """

    def __init__(
                    self,
                    modelobj,
                    model_df,
                    ydepend,
                    cat_df=None,
                    keepfeaturelist=None,
                    groupbyvars=None,
                    aggregate_func=np.nanmedian,
                    error_type='MSE',
                    autoformat_types=False,
                    verbose=0):

        """
        :param modelobj: sklearn model object
        :param model_df: Pandas Dataframe used to build/train model
        :param ydepend: Y dependent variable
        :param cat_df: Pandas Dataframe of raw data - with categorical datatypes
        :param keepfeaturelist: Subset and rename columns
        :param groupbyvars: grouping variables
        :param aggregate_func: numpy aggregate function like np.mean
        :param dominate_class: in the case of binary classification, class of interest
            to measure probabilities from
        :param autoformat: experimental autoformatting of dataframe
        :param verbose: Logging level
        """
        super(WhiteBoxError, self).__init__(
                                            modelobj,
                                            model_df,
                                            ydepend,
                                            cat_df=cat_df,
                                            keepfeaturelist=keepfeaturelist,
                                            groupbyvars=groupbyvars,
                                            aggregate_func=aggregate_func,
                                            error_type=error_type,
                                            autoformat_types=autoformat_types,
                                            verbose=verbose)

        self.debug_df = pd.DataFrame()

    def _create_group_errors(self,
                             group_copy):
        """
        split errors into positive and negative errors, concatenate poisitive and negative
        errosr on index. Concatenate with original group columns for final dataset
        :param group_copy: deep copy of region of data driven by groupby level
        :return: errors dataframe
        """
        # split out positive vs negative errors
        errors = group_copy['errors'].reset_index(drop=True).copy(deep=True)
        # check if classification
        if self.model_type == 'classification':
            # get user defined aggregate (central values) of the errors
            agg_errors = self.aggregate_func(errors)
            # subtract the aggregate value for the group from the errors
            errors = errors.apply(lambda x: agg_errors - x)
        # create separate columns for pos or neg errors - average in zeros to both
        errors = pd.concat([errors[errors >= 0], errors[errors <= 0]], axis=1)
        # rename error columns
        errors.columns = ['errPos', 'errNeg']
        # merge back with orignial data
        toreturn = pd.concat([group_copy.loc[:, group_copy.columns != 'errors'], errors], axis=1)
        # return
        return toreturn

    def _transform_function(
                            self,
                            group,
                            groupby_var='Type',
                            col=None,
                            vartype='Continuous'):
        """
        transform slice of data by separating our pos/neg errors, and aggregating
        based on user defined aggregation function. Format final error output
        :param group: slice of data being operated on
        :param col: current col name needed for continuous transform
        :param vartype: str --> categorical or continuous
        :return: compressed data representation in dataframe object
        """
        assert 'errors' in group.columns, 'errors needs to be present in dataframe slice'
        assert vartype in ['Continuous', 'Categorical'], 'variable type needs to be continuous or categorical'

        # copy so we don't change the org. data
        group_copy = group.reset_index(drop=True).copy(deep=True)
        # create group errors dataframe
        toreturn = self._create_group_errors(group_copy)

        # fmt and append to instance raw_df
        self.fmt_raw_df(col=col,
                        groupby_var=groupby_var,
                        cur_group=toreturn)

        # create switch for aggregate types based on continuous or categorical values
        if vartype == 'Categorical':
            col_val = toreturn[col].mode()
        else:
            col_val = toreturn[col].max()

        # aggregate errors
        agg_errors = pd.DataFrame({col: col_val,
                                   'groupByValue': toreturn[groupby_var].mode(),
                                   'groupByVarName': groupby_var,
                                   'predictedYSmooth': self.aggregate_func(toreturn['predictedYSmooth']),
                                   'errPos': self.aggregate_func(toreturn['errPos']),
                                   'errNeg': self.aggregate_func(toreturn['errNeg'])}, index=[0])

        # fmt and append to instance agg_df attribute
        self.fmt_agg_df(col=col,
                        agg_errors=agg_errors)

        return agg_errors

    def _var_check(
                    self,
                    col=None,
                    groupby_var=None):
        """
        handle continuous and categorical variable types
        :param col: specific column being operated on within dataset -- str
        :param groupby_var: specific groupby variable being operated on within dataset -- str
        :return: errors dataframe for particular column and groupby variable
        """
        # subset col indices
        col_indices = [col, 'errors', 'predictedYSmooth', groupby_var]

        error_holder = pd.DataFrame()

        # iterate over groups
        for group_level in self._cat_df[groupby_var].unique():
            # subset data to current group
            cur_group = self._cat_df[self._cat_df[groupby_var] == group_level][col_indices]\
                                                                                .reset_index(drop=True).copy(deep=True)

            # check if categorical
            if is_object_dtype(self._cat_df.loc[:, col]):
                # set variable type
                vartype = 'Categorical'
                # apply transform function
                group_errors = self._transform_function(cur_group,
                                                        col=col,
                                                        vartype=vartype,
                                                        groupby_var=groupby_var)

            elif is_numeric_dtype(self._cat_df.loc[:, col]):
                vartype = 'Continuous'

                print("VARCHECK --- COL: {} --- GROUPBY: {}".format(col, groupby_var))
                # apply transform function
                group_errors = self._continuous_slice(cur_group,
                                                      col=col,
                                                      groupby_var=groupby_var)

            else:
                raise ValueError("""unsupported dtype: {}""".format(self._cat_df.loc[:, col].dtype))

            error_holder = error_holder.append(group_errors)

        # reset & drop index - replace NaN with 'null' for d3 out
        error_holder.reset_index(drop=True, inplace=True)
        error_holder.fillna('null', inplace=True)
        # convert to json structure
        json_out = formatting.FmtJson.to_json(
                                    error_holder,
                                    vartype=vartype,
                                    html_type='error',
                                    incremental_val=None)
        # return json_out
        return json_out


class WhiteBoxSensitivity(WhiteBoxBase):

    """
    Sensitivity model analysis.

    In the continuous case, increment values by user defined number of standard
    deviations, rerun model predictions, and calculate the average differences
    in original predictions with raw data and new predictions with synthetic data
    for various groups or slices of the original dataset.

    In the categorical case, convert non modal values to the mode and calculate
    model sensitivity based on this adjustment.

    Model sensitivity is the difference between original predictions for unadjusted data
    and synthetic data predictions.

    Parameters

    ----------

    modelobj : sklearn model object
        Used to create predictions for synthetic data

    model_df : pandas DataFrame
        Original raw dataset used to train and calibrate modelobj. This can
        and should in most cases include dummy variables for categorical data columns.

    ydepend : str
        Y dependent variable used to build modelobj

    cat_df : pandas DataFrame
        Raw, unadjusted dataframe with categorical columns converted to pandas
        data type Categorical. These categorical designations are picked up throughout
        and are important for proper functioning of WhiteBoxSensitvitiy

    featuredict : dict
        Optional user defined dictionary to clean up column name and subset
        the outputs to only columns defined in featuredict

    groupbyvars : list
        grouping variables to analyze impact of model for various groups of data. I.e.
        if working on dataset with red and white wine, we can disambiguate how sensitive
        the model is to changes in data for each type of wine

    aggregate_func : function
        function to perform aggregate function to groups of data pertaining to sensitivity
        analysis. I.e. take the median model sensitivity for groups of data.

    verbose : int
        Logging level

    std_num : int
        Number of standard deviations to push data for syntehtic variable creation and senstivity analysis. Appropriate
        values include -3, -2, -1, 1, 2, 3

    See also

    ------------

    WhiteBoxError : analyze how the model errors are doing for various groups of data
    WhiteBoxBase : base class inherited from WhiteBoxSensitivity to perform key functionality
        and grouping logic
    """

    def __init__(self,
                 modelobj,
                 model_df,
                 ydepend,
                 cat_df=None,
                 keepfeaturelist=None,
                 groupbyvars=None,
                 aggregate_func=np.nanmedian,
                 error_type='MEAN',
                 std_num=0.5,
                 autoformat_types=False,
                 verbose=0,
                 ):
        """
        :param modelobj: sklearn model object
        :param model_df: Pandas Dataframe used to build/train model
        :param ydepend: Y dependent variable
        :param cat_df: Pandas Dataframe of raw data - with categorical datatypes
        :param featuredict: Subset and rename columns
        :param groupbyvars: grouping variables
        :param aggregate_func: function to aggregate sensitivity results by group
        :param autoformat: experimental auto formatting of dataframe
        :param verbose: Logging level
        :param std_num: Standard deviation adjustment
        """

        if std_num > 3 or std_num < -3:
            raise ValueError("""Standard deviation adjustment must be between -3 and 3
                                \nCurrent value: {}""".format(std_num))

        self.std_num = std_num

        super(WhiteBoxSensitivity, self).__init__(
                                                    modelobj,
                                                    model_df,
                                                    ydepend,
                                                    cat_df=cat_df,
                                                    keepfeaturelist=keepfeaturelist,
                                                    groupbyvars=groupbyvars,
                                                    aggregate_func=aggregate_func,
                                                    error_type=error_type,
                                                    autoformat_types=autoformat_types,
                                                    verbose=verbose)

    def _transform_function(
                            self,
                            group,
                            groupby_var='Type',
                            col=None,
                            vartype='Continuous'):
        """
        transform slice of data by separating our pos/neg errors, aggregating up to the mean of the slice
        and returning transformed dataset
        :param group: slice of data being operated on
        :param col: current col name needed for continuous transform
        :param vartype: str --> categorical or continuous
        :return: compressed data representation in dataframe object
        """
        assert 'errors' in group.columns, 'errors needs to be present in dataframe slice'
        assert vartype in ['Continuous', 'Categorical'], 'variable type needs to be continuous or categorical'

        # append raw_df to instance attribute raw_df
        self.fmt_raw_df(col=col,
                        groupby_var=groupby_var,
                        cur_group=group)

        # create switch for aggregate types based on continuous or categorical values
        if vartype == 'Categorical':
            col_val = group[col].mode()
        else:
            col_val = group[col].max()

        # aggregate errors
        agg_errors = pd.DataFrame({col: col_val,
                                   'groupByValue': group[groupby_var].mode(),
                                   'groupByVarName': groupby_var,
                                   'predictedYSmooth': self.aggregate_func(group['diff'])}, index=[0])

        # fmt and append agg_df to instance attribute
        # agg_df
        self.fmt_agg_df(col=col,
                        agg_errors=agg_errors)

        return agg_errors

    def _handle_categorical_preds(self,
                                  col,
                                  groupby,
                                  copydf,
                                  col_indices):
        """
        To measure sensitivity in the categorical case, the mode value of the categorical
        column is identified, and all other levels within the category are switch to the
        mode value. New predictions are created with this synthetic dataset and the
        difference between the original predictions with the real data vs the synthetic
        predictions are calculated and returned.
        :param col: current column being operated on
        :param groupby: current groupby being operated on
        :param copydf: deep copy of cat_df
        :param col_indices: column indices to include
        :return: incremental bump value, sensitivity dataframe
        """
        logging.info("""Column determined as categorical datatype, transforming data for categorical column
                                                \nColumn: {}
                                                \nGroup: {}""".format(col, groupby))

        # switch modal column for predictions
        modal_val, modaldf = pandas_switch_modal_dummy(col,
                                                       self._cat_df,
                                                       copydf)

        # make predictions with the switches to the dataset
        if self.model_type == 'classification':
            copydf['new_predictions'] = self.predict_engine(modaldf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                'predictedYSmooth'])])[:, 1]
        elif self.model_type == 'regression':
            copydf['new_predictions'] = self.predict_engine(modaldf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                'predictedYSmooth'])])
        # calculate difference between actual predictions and new_predictions
        self._cat_df['diff'] = modaldf['new_predictions'] - modaldf['predictedYSmooth']
        # create mask of data to select rows that are not equal to the mode of the category.
        # This will prevent blank displays in HTML
        mode_mask = self._cat_df[col] != modal_val
        # slice over the groupby variable and the categories within the current column
        sensitivity = self._cat_df[mode_mask][col_indices].groupby([groupby, col]).apply(self._transform_function,
                                                                                         col=col,
                                                                                         groupby_var=groupby,
                                                                                         vartype='Categorical')
        # return sensitivity
        return modal_val, sensitivity

    def _handle_continuous_preds(self,
                                 col,
                                 groupby,
                                 copydf,
                                 col_indices):
        """
        In the continuous case, the standard deviation is determined by the values of
        the continuous column. This is multipled by the user defined std_num and applied
        to the values in the continuous column. New predictions are generated on this synthetic
        dataset, and the difference between the original predictions and the new predictions are
        captured and assigned.
        :param col: current column being operated on
        :param groupby: current groupby being operated on
        :param copydf: deep copy of cat_df
        :param col_indices: column indices to include
        :return: incremental bump value, sensitivity dataframe
        """
        logging.info("""Column determined as continuous datatype, transforming data for continuous column
                                                            \nColumn: {}
                                                            \nGroup: {}""".format(col, groupby))
        incremental_val = copydf[col].std() * self.std_num
        # tweak the currently column by the incremental_val
        copydf[col] = copydf[col] + incremental_val
        # make predictions with the switches to the dataset
        if self.model_type == 'classification':
            # binary classification - pull prediction probabilities for the class designated as 1
            copydf['new_predictions'] = self.predict_engine(copydf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                'predictedYSmooth'])])[:,1]
        elif self.model_type == 'regression':
            copydf['new_predictions'] = self.predict_engine(copydf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                'predictedYSmooth'])])

        # calculate difference between actual predictions and new_predictions
        self._cat_df['diff'] = copydf['new_predictions'] - copydf['predictedYSmooth']
        # groupby and apply
        sensitivity = self._cat_df[col_indices].groupby(groupby).apply(self._continuous_slice,
                                                                       col=col,
                                                                       groupby_var=groupby)

        # return sensitivity
        return incremental_val, sensitivity

    def _var_check(
                    self,
                    col=None,
                    groupby_var=None):
        """
        handle continuous and categorical variable types
        :param col: specific column being operated on within dataset -- str
        :param groupby_var: specific groupby variable being operated on within dataset -- str
        :return: errors dataframe for particular column and groupby variable
        """
        # subset col indices
        col_indices = [col, 'errors', 'predictedYSmooth', groupby_var, 'diff']
        # make a copy of the data to manipulate and change values
        copydf = self._model_df.copy(deep=True)
        # check if categorical
        if is_object_dtype(self._cat_df.loc[:, col]):
            # set variable type
            vartype = 'Categorical'
            incremental_val, sensitivity = self._handle_categorical_preds(col,
                                                                          groupby_var,
                                                                          copydf,
                                                                          col_indices)
        elif is_numeric_dtype(self._cat_df.loc[:, col]):
            # set variable type
            vartype = 'Continuous'
            incremental_val, sensitivity = self._handle_continuous_preds(col,
                                                                         groupby_var,
                                                                         copydf,
                                                                         col_indices)

        else:
            raise ValueError("""Unsupported dtypes: {}""".format(self._cat_df.loc[:, col].dtype))

        sensitivity = sensitivity.reset_index(drop=True).fillna('null')
        logging.info("""Converting output to json type using to_json utility function""")
        # convert to json structure
        json_out = formatting.FmtJson.to_json(sensitivity,
                                              vartype=vartype,
                                              html_type='sensitivity',
                                              incremental_val=incremental_val)
        # return json_out
        return json_out
