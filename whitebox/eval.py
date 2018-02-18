#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
from pandas import DataFrame, concat
from pandas.api.types import is_object_dtype

try:
    from utils import to_json
    from base import WhiteBoxBase
except ImportError:
    from whitebox.utils import to_json
    from whitebox.base import WhiteBoxBase


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

    featuredict : dict
        Optional user defined dictionary to clean up column name and subset
        the outputs to only columns defined in featuredict

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
                    featuredict=None,
                    groupbyvars=None,
                    aggregate_func=np.mean,
                    error_type='MSE',
                    autoformat=False,
                    verbose=0):

        """
        :param modelobj: sklearn model object
        :param model_df: Pandas Dataframe used to build/train model
        :param ydepend: Y dependent variable
        :param cat_df: Pandas Dataframe of raw data - with categorical datatypes
        :param featuredict: Subset and rename columns
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
                                            featuredict=featuredict,
                                            groupbyvars=groupbyvars,
                                            aggregate_func=aggregate_func,
                                            error_type=error_type,
                                            autoformat=autoformat,
                                            verbose=verbose)

        self.allerrors = DataFrame()

    def _create_group_errors(self,
                             group_copy):
        """
        split errors into positive and negative errors, concatenate poisitive and negative
        errosr on index. Concatenate with original group columns for final dataset
        :param group_copy: deep copy of region of data driven by groupby level
        :return: errors dataframe
        """
        # split out positive vs negative errors
        errors = group_copy['errors']
        # check if classification
        if self.model_type == 'classification':
            # get user defined aggregate (central values) value of the errors
            agg_errors = self.aggregate_func(errors)
            # subtract the aggregate value for the group from the errors
            errors = errors.apply(lambda x: agg_errors - x)
        # need non zero mask when concatenating non error columns back
        non_zero_errors_mask = errors != 0
        # create separate columns for pos or neg errors
        errors = concat([errors[errors > 0], errors[errors < 0]], axis=1)
        # rename error columns
        errors.columns = ['errPos', 'errNeg']
        # merge back with orignial data
        toreturn = concat([group_copy.loc[non_zero_errors_mask, group_copy.columns != 'errors'], errors], axis=1)
        # return
        return toreturn

    def _transform_function(
                            self,
                            group,
                            groupby='Type',
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
        # copy so we don't change the og data
        group_copy = group.copy(deep=True)

        # create group errors dataframe
        toreturn = self._create_group_errors(group_copy)

        # return the mean
        if vartype == 'Categorical':
            logging.info(""""Returning mean values for group of categorical variable in transform_function
                             \nGroup: {}
                             \nGroup Shape: {}
                             \nCol: {}
                             \nvartype: {}""".format(groupby,
                                                     group.shape,
                                                     col,
                                                     vartype))
            # return the mean of the columns
            return self.aggregate_func(toreturn)
        else:
            logging.info(""""Returning continuous aggregate values in transform_function
                                         \nGroup: {}
                                         \nGroup Shape: {}
                                         \nCol: {}
                                         \nvartype: {}""".format(groupby,
                                                                 group.shape,
                                                                 col,
                                                                 vartype))
            errors = DataFrame({
                                    col: toreturn[col].max(),
                                    groupby: toreturn[groupby].mode(),
                                    'predictedYSmooth': self.aggregate_func(toreturn['predictedYSmooth']),
                                    'errPos': self.aggregate_func(toreturn['errPos']),
                                    'errNeg': self.aggregate_func(toreturn['errNeg'])})


            self.allerrors = self.allerrors.append(errors)
            return errors

    def _handle_categorical(self,
                            col,
                            groupby,
                            col_indices):
        """
        pass categorical variable column to  _transform_func to capture
        positive and negative errosr within a particular region of the data
        driven by groupby level.
        :param col: current column being operated on
        :param groupby: current groupby being operated on
        :param col_indices: column indices to include
        :return: errors dataframe
        """
        logging.info("""Column determined as categorical datatype, transforming data for categorical column
                                    \nColumn: {}
                                    \nGroup: {}""".format(col, groupby))
        # slice over the groupby variable and the categories within the current column
        errors = self.cat_df[col_indices].groupby([groupby, col]).apply(self._transform_function,
                                                                        col=col,
                                                                        groupby=groupby,
                                                                        vartype='Categorical')
        # final categorical transformations
        errors.reset_index(inplace=True)
        # rename
        errors.rename(columns={groupby: 'groupByValue'}, inplace=True)
        # assign groupby variable to the errors dataframe
        errors['groupByVarName'] = groupby
        # return
        return errors

    def _handle_continuous(self,
                           col,
                           groupby,
                           col_indices):
        """
        pass continous variable column to _continuous_slice to create percentile buckets,
        measures positive and negative errosr, and create final errors dataframe
        :param col: current column being operated on
        :param groupby: current groupby being operated on
        :param col_indices: column indices to include
        :return: errors dataframe
        """
        logging.info("""Column determined as continuous datatype, transforming data for continuous column
                                                \nColumn: {}
                                                \nGroup: {}""".format(col, groupby))
        # groupby the groupby variable on subset of columns and apply _continuous_slice
        errors = self.cat_df[col_indices].groupby(groupby).apply(self._continuous_slice,
                                                                 col=col,
                                                                 vartype='Continuous',
                                                                 groupby=groupby)

        return errors

    def _var_check(
                    self,
                    col=None,
                    groupby=None):
        """
        handle continuous and categorical variable types
        :param col: specific column being operated on within dataset -- str
        :param groupby: specific groupby variable being operated on within dataset -- str
        :return: errors dataframe for particular column and groupby variable
        """
        # subset col indices
        col_indices = [col, 'errors', 'predictedYSmooth', groupby]
        # check if categorical
        if is_object_dtype(self.cat_df.loc[:, col]):
            # set variable type
            vartype = 'Categorical'
            errors = self._handle_categorical(col,
                                              groupby,
                                              col_indices)

        else:
            vartype = 'Continuous'
            errors = self._handle_continuous(col,
                                             groupby,
                                             col_indices)

        errors = errors.fillna('null')
        # convert to json structure
        json_out = to_json(
                                    errors,
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
                 featuredict=None,
                 groupbyvars=None,
                 aggregate_func=np.median,
                 error_type='MSE',
                 std_num=0.5,
                 autoformat=False,
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
                                                    featuredict=featuredict,
                                                    groupbyvars=groupbyvars,
                                                    aggregate_func=aggregate_func,
                                                    error_type=error_type,
                                                    autoformat=autoformat,
                                                    verbose=verbose)

    def _transform_function(
                            self,
                            group,
                            groupby='Type',
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
        if vartype == 'Continuous':
            logging.info(""""Returning aggregate values for group of continuous variable in transform_function of WhiteBoxSensitivity.
                            \nColumn: {}
                            \nGroup: {}
                            \nGroup shape: {}""".format(col, groupby, group.shape))
            # return the max value for the Continuous case
            errors = DataFrame({
                                    col: group[col].max(),
                                    groupby: group[groupby].mode(),
                                    'predictedYSmooth': self.aggregate_func(group['diff'])})
        else:
            logging.info(""""Returning aggregate values for group of categorical variable in transform_function of WhiteBoxSensitvity.
                                        \nColumn: {}
                                        \nGroup: {}
                                        \nGroup shape: {}""".format(col, groupby, group.shape))
            # return the mode for the categorical case
            errors = DataFrame({col: group[col].mode(),
                                groupby: group[groupby].mode(),
                                'predictedYSmooth': self.aggregate_func(group['diff'])})

        return errors

    def _handle_categorical(self,
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
        # TODO need to handle case where user adds columns to cat_df but is not in model_df - can't select these cols
        # TODO split this into two additional methods - categorical and continuous case
        logging.info("""Column determined as categorical datatype, transforming data for categorical column
                                                \nColumn: {}
                                                \nGroup: {}""".format(col, groupby))
        rev_col = self._reverse_featuredict[col]
        # map categories with main column name to properly subset
        all_type_cols = ['{}_{}'.format(rev_col, cat) for cat in self.cat_df.loc[:, col].unique()]
        # find the mode from the original cat_df for this column
        incremental_val = str(self.cat_df[col].mode().values[0])
        # find the columns within all_type_cols related to the mode_val
        mode_col = list(filter(lambda x: incremental_val in x, all_type_cols))
        # convert mode cols to all 1's
        copydf.loc[:, mode_col] = 1
        # convert all other non mode cols to zeros
        non_mode_col = list(filter(lambda x: incremental_val not in x, all_type_cols))
        copydf.loc[:, non_mode_col] = 0
        # make predictions with the switches to the dataset
        if self.model_type == 'classification':
            copydf['new_predictions'] = self.predict_engine(copydf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                'predictedYSmooth'])])[
                                        :, 1]
        if self.model_type == 'regression':
            copydf['new_predictions'] = self.predict_engine(copydf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                'predictedYSmooth'])])
        # calculate difference between actual predictions and new_predictions
        self.cat_df['diff'] = copydf['new_predictions'] - copydf['predictedYSmooth']
        # create mask of data to select rows that are not equal to the mode of the category.
        # This will prevent blank displays in HTML
        mode_mask = self.cat_df[col] != incremental_val
        # slice over the groupby variable and the categories within the current column
        sensitivity = self.cat_df[mode_mask][col_indices].groupby([groupby, col]).apply(self._transform_function,
                                                                                        col=col,
                                                                                        groupby=groupby,
                                                                                        vartype='Categorical')
        # rename groupby
        sensitivity.rename(columns={groupby: 'groupByValue'}, inplace=True)
        # assign groupby variable to the errors dataframe
        sensitivity['groupByVarName'] = groupby
        # return sensitivity
        return incremental_val, sensitivity

    def _handle_continuous(self,
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
        rev_col = self._reverse_featuredict[col]
        incremental_val = copydf[rev_col].std() * self.std_num
        # tweak the currently column by the incremental_val
        copydf[rev_col] = copydf[rev_col] + incremental_val
        # make predictions with the switches to the dataset
        if self.model_type == 'classification':
            copydf['new_predictions'] = self.predict_engine(copydf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                'predictedYSmooth'])])[
                                        :, 1]
        if self.model_type == 'regression':
            copydf['new_predictions'] = self.predict_engine(copydf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                'predictedYSmooth'])])
        # calculate difference between actual predictions and new_predictions
        self.cat_df['diff'] = copydf['new_predictions'] - copydf['predictedYSmooth']
        # groupby and apply
        sensitivity = self.cat_df[col_indices].groupby(groupby).apply(self._continuous_slice,
                                                                      col=col,
                                                                      vartype='Continuous',
                                                                      groupby=groupby)

        # return sensitivity
        return incremental_val, sensitivity

    def _var_check(
                    self,
                    col=None,
                    groupby=None):
        """
        handle continuous and categorical variable types
        :param col: specific column being operated on within dataset -- str
        :param groupby: specific groupby variable being operated on within dataset -- str
        :return: errors dataframe for particular column and groupby variable
        """
        # subset col indices
        col_indices = [col, 'errors', 'predictedYSmooth', groupby, 'diff']
        # make a copy of the data to manipulate and change values
        copydf = self.model_df.copy(deep=True)
        # check if categorical
        if is_object_dtype(self.cat_df.loc[:, col]):
            # set variable type
            vartype = 'Categorical'
            incremental_val, sensitivity = self._handle_categorical(col,
                                                                    groupby,
                                                                    copydf,
                                                                    col_indices)
        else:
            # set variable type
            vartype = 'Continuous'
            incremental_val, sensitivity = self._handle_continuous(col,
                                                                   groupby,
                                                                   copydf,
                                                                   col_indices)


        sensitivity = sensitivity.fillna('null')
        logging.info("""Converting output to json type using to_json utility function""")
        # convert to json structure
        json_out = to_json(sensitivity, vartype=vartype, html_type = 'sensitivity',
                                 incremental_val=incremental_val)
        # return json_out
        return json_out
