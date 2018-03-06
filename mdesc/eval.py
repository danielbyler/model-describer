#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import gc

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype

try:
    import utils.utils as wb_utils
    from utils.categorical_conversions import pandas_switch_modal_dummy
    import utils.formatting as formatting
    from base import MdescBase
except ImportError:
    import mdesc.utils.utils as wb_utils
    from mdesc.utils.categorical_conversions import pandas_switch_modal_dummy
    from mdesc.base import MdescBase
    from mdesc.utils import formatting

logger = wb_utils.util_logger(__name__)

class ErrorViz(MdescBase):

    """
    Error model analysis.

    In the continuous case with over 100 datapoints for a particular slice of data,
    calculate percentiles of group to shrink data to 100 datapoints for scalability.
    Calculate average positive and negative errors within this region of the data.

    In the categorical case, calculate average positive/negative error within specific
    level of category

    Example:
    >>> from mdesc.eval import ErrorViz
    ...
    >>> EV = ErrorViz(modelobj=modelObjc,
    ...                model_df=mod_df,
    ...                ydepend=ydepend,
    ...                cat_df=wine_sub,
    ...                groupbyvars=['Type', 'alcohol'],
    ...                keepfeaturelist=None,
    ...                verbose=None,
    ...                round_num=2,
    ...                autoformat_types=True)
    >>> # run MLReveal error calibration and save output to local html file
    >>> EV.run(output_type='html', output_path='path/to/save.html')

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
        and are important for proper functioning of SensitivityViz

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

    SensitivityViz : analyze how the model errors are doing for various groups of data
    MdescBase : base class inherited from SensitivityViz to perform key functionality
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
                    round_num=2,
                    verbose=None):

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
        :param round_num: round numeric values for output
        :param autoformat: experimental autoformatting of dataframe
        :param verbose: Logging level
        """
        logger.setLevel(wb_utils.Settings.verbose2log[verbose])
        super(ErrorViz, self).__init__(
                                            modelobj,
                                            model_df,
                                            ydepend,
                                            cat_df=cat_df,
                                            keepfeaturelist=keepfeaturelist,
                                            groupbyvars=groupbyvars,
                                            aggregate_func=aggregate_func,
                                            error_type=error_type,
                                            autoformat_types=autoformat_types,
                                            round_num=round_num,
                                            verbose=verbose)

        self.debug_df = pd.DataFrame()

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

        logger.info("""Processing -- groupby_var: {} -- col: {} -- vartype: {} -- group shape: {}""".format(groupby_var,
                                                                                                            col,
                                                                                                            vartype,
                                                                                                            group.shape))
        # pull out errors
        error_arr = group['errors'].values
        # subtract errors from group median
        if self.model_type == 'classification':
            # get user defined aggregate (central values) of the errors
            agg_errors = self.aggregate_func(error_arr)
            # subtract the aggregate value for the group from the errors
            error_arr = agg_errors - error_arr

        # create switch for aggregate types based on continuous or categorical values
        if vartype == 'Categorical':
            col_val = group[col].mode()
        else:
            col_val = group[col].max()

        # aggregate errors
        agg_errors = pd.DataFrame({col: col_val,
                                   'groupByValue': group[groupby_var].mode(),
                                   'groupByVarName': groupby_var,
                                   'predictedYSmooth': self.aggregate_func(group['predictedYSmooth']),
                                   'errPos': self.aggregate_func(error_arr[error_arr >= 0]),
                                   'errNeg': self.aggregate_func(error_arr[error_arr <= 0])}, index=[0])

        # fmt and append to instance agg_df attribute
        self._fmt_agg_df(col=col,
                         agg_errors=agg_errors)

        return agg_errors.round(self.round_num)

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

        error_list = []

        # iterate over groups
        for group_level in self._cat_df[groupby_var].unique():
            # subset data to current group
            cur_group = (self._cat_df[self._cat_df[groupby_var] == group_level][col_indices]
                         .reset_index(drop=True)
                         .copy(deep=True))



            # check if categorical
            if is_object_dtype(self._cat_df.loc[:, col]):
                # set variable type
                vartype = 'Categorical'
                logger.info("""Categorical variable detected - group_level: {}""".format(group_level))
                # apply transform function

                group_errors = cur_group.groupby(col).apply(self._transform_function,
                                                            col=col,
                                                            vartype=vartype,
                                                            groupby_var=groupby_var)


            elif is_numeric_dtype(self._cat_df.loc[:, col]):
                vartype = 'Continuous'
                logger.info("""Categorical variable detected - group_level: {}""".format(group_level))
                # apply transform function
                group_errors = self._continuous_slice(cur_group,
                                                      col=col,
                                                      groupby_var=groupby_var)

            else:
                logger.error("""unsupported dtype detected: {}""".format(self._cat_df.loc[:, col].dtype))
                raise ValueError("""unsupported dtype: {}""".format(self._cat_df.loc[:, col].dtype))

            error_list.append(group_errors)

        # reset & drop index - replace NaN with 'null' for d3 out
        error_df = pd.concat(error_list)
        error_df = (error_df.reset_index(drop=True)
                            .fillna('null')
                            .round(self.round_num))
        # convert to json structure
        json_out = formatting.FmtJson.to_json(
                                    error_df,
                                    vartype=vartype,
                                    html_type='error',
                                    incremental_val=None)
        # return json_out
        return json_out


class SensitivityViz(MdescBase):

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
        and are important for proper functioning of SensitivityViz

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

    >>> from mdesc.eval import SensitivityViz
    ...
    >>> SV = SensitivityViz(modelobj=modelObjc,
    ...               model_df=mod_df,
    ...               ydepend=ydepend,
    ...               cat_df=wine_sub,
    ...               groupbyvars=['Type', 'alcohol'],
    ...               keepfeaturelist=None,
    ...               verbose=None,
    ...               round_num=2,
    ...               autoformat_types=True)
    >>> # run MLReveal sensity calibration and save final output to html
    >>> SV.run(output_type='html', output_path='path/to/save.html')

    See also

    ------------

    ErrorViz : analyze how the model errors are doing for various groups of data
    MdescBase : base class inherited from SensitivityViz to perform key functionality
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
                 round_num=2,
                 verbose=None,
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
        :param round_num: round numeric values for output
        :param verbose: Logging level
        :param std_num: Standard deviation adjustment
        """
        logger.setLevel(wb_utils.Settings.verbose2log[verbose])
        logger.info('Initilizing {} parameters'.format(self.__class__.__name__))

        if std_num > 3 or std_num < -3:
            raise ValueError("""Standard deviation adjustment must be between -3 and 3
                                \nCurrent value: {}""".format(std_num))

        self.std_num = std_num

        super(SensitivityViz, self).__init__(
                                                    modelobj,
                                                    model_df,
                                                    ydepend,
                                                    cat_df=cat_df,
                                                    keepfeaturelist=keepfeaturelist,
                                                    groupbyvars=groupbyvars,
                                                    aggregate_func=aggregate_func,
                                                    error_type=error_type,
                                                    autoformat_types=autoformat_types,
                                                    round_num=round_num,
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
        logger.info("""Processing -- groupby_var: {} -- col: {} -- vartype: {} -- group shape: {}""".format(groupby_var,
                                                                                                            col,
                                                                                                            vartype,
                                                                                                            group.shape))

        # append raw_df to instance attribute raw_df
        self._fmt_raw_df(col=col,
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
        self._fmt_agg_df(col=col,
                         agg_errors=agg_errors)

        return agg_errors

    def _predict_synthetic(self,
                           col,
                           groupby,
                           copydf,
                           col_indices,
                           vartype='Continuous'):
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
        logger.info(
            """Column determined as {} datatype, transforming data for continuous column -- Column: {} -- Group: {}""".format(
                vartype, col, groupby))

        # create copy of cat_df
        cat_df = self._cat_df.copy(deep=True)

        if vartype == 'Continuous':
            incremental_val = copydf[col].std() * self.std_num
            # tweak the currently column by the incremental_val
            copydf[col] = copydf[col] + incremental_val

        else:
            # switch modal column for predictions and subset
            # rows that are not already the mode value
            incremental_val, copydf, cat_df = pandas_switch_modal_dummy(col,
                                                                        cat_df,
                                                                        copydf)

        # make predictions with the switches to the dataset
        new_preds = self._create_preds(copydf)

        # calculate difference between actual predictions and new_predictions
        cat_df['diff'] = new_preds - cat_df['predictedYSmooth']

        if vartype == 'Continuous':
            # groupby and apply
            sensitivity = cat_df[col_indices].groupby(groupby).apply(self._continuous_slice,
                                                                     col=col,
                                                                     groupby_var=groupby)
        # categorical
        else:
            sensitivity = cat_df[col_indices].groupby([col, groupby]).apply(self._transform_function,
                                                                            col=col,
                                                                            groupby_var=groupby,
                                                                            vartype=vartype)

        # cleanup
        del cat_df
        gc.collect()

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
            logger.info("""Categorical variable detected""")

        elif is_numeric_dtype(self._cat_df.loc[:, col]):
            # set variable type
            vartype = 'Continuous'
            logger.info("""Continuous variable detected""")

        else:
            raise ValueError("""Unsupported dtypes: {}""".format(self._cat_df.loc[:, col].dtype))

        incremental_val, sensitivity = self._predict_synthetic(col,
                                                               groupby_var,
                                                               copydf,
                                                               col_indices,
                                                               vartype=vartype)

        sensitivity = (sensitivity.reset_index(drop=True)
                       .fillna('null'))


        logging.info("""Converting output to json type using to_json utility function""")
        # convert to json structure
        json_out = formatting.FmtJson.to_json(sensitivity,
                                              vartype=vartype,
                                              html_type='sensitivity',
                                              incremental_val=incremental_val)

        logger.info("""Converted output to json format""")
        # return json_out
        return json_out
