#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import abc
import warnings
from functools import partial
import numpy as np
from sklearn.exceptions import NotFittedError
import pandas as pd
from pandas.api.types import is_categorical_dtype
from whitebox import utils


__author__ = "Jason Lewris, Daniel Byler, Venkat Gangavarapu, Shruti Panda, Shanti Jha"
__credits__ = ["Brian Ray"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jason Lewris"
__email__ = "jlewris@deloitte.com"
__status__ = "Beta"

class WhiteBoxBase(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,
                 modelobj,
                 model_df,
                 ydepend,
                 cat_df = None,
                 featuredict = None,
                 groupbyvars = None,
                 aggregate_func=np.mean,
                 verbose=None):
        """
        initialize base class
        :param modelobj: fitted sklearn model object
        :param model_df: data used for training sklearn object
        :param ydepend: dependent variable
        :param cat_df: formatted dataset with categorical dtypes specified, and non-dummy categories
        :param featuredict: prettty printing and subsetting analysis
        :param groupbyvars: grouping variables
        :param verbose: set verbose level -- 0 = debug, 1 = warning, 2 = error
        """

        # basic parameter checks
        if not hasattr(modelobj, 'predict'):
            raise ValueError("""modelObj does not have predict method.
                                WhiteBoxError only works with model objects with predict method""")
        # need to ensure modelobj has previously been fitted, otherwise raise NotFittedError
        try:
            # try predicting on model dataframe
            modelobj.predict(model_df.loc[:, model_df.columns != ydepend])
        except NotFittedError as e:
            # raise exception and not fitted error
            raise Exception('{}\nPlease fit model: {}'.format(e, modelobj.__class__))

        if not isinstance(model_df, pd.core.frame.DataFrame):
            raise TypeError("""model_df variable not pandas dataframe.
                                WhiteBoxError only works with dataframe objects""")

        if featuredict is not None and not isinstance(featuredict, dict):
            raise TypeError("""When used Featuredict needs to be of type dictionary. Keys are original column
                               names and values are formatted column names:
                               \n{sloppy.name.here: Sloppy Name}""")
        # toggle featuredict depending on whether user specifys cat_df and leaves featuredict blank
        if featuredict is None and cat_df is not None:
            self.featuredict = {col : col for col in cat_df.columns}
        elif featuredict is None:
            self.featuredict = {col : col for col in model_df.columns}
        else:
            self.featuredict = featuredict

        if isinstance(cat_df, pd.core.frame.DataFrame):
            # check tthat the number of rows from cat_df matches that of model_df
            if model_df.shape[0] != cat_df.shape[0]:
                raise StandardError("""Misaligned rows. \norig_df shape: {}
                                        \ndummy_df shape: {}""".format(model_df.shape[0],
                                                                                        cat_df.shape[0]))
            # assign cat_df to class instance and subset to featuredict keys
            self.cat_df = cat_df[list(self.featuredict.keys())].copy(deep = True)

        else:
            # check that cat_df is not none and if it's not a pandas dataframe, throw warning
            if not isinstance(cat_df, type(None)):
                warnings.warn("""cat_df is not None and not a pd.core.frame.DataFrame. Default becomes model_df
                              and may not be intended user behavior""", UserWarning)
            # assign cat_df to class instance and subset based on featuredict keys
            self.cat_df = model_df[list(self.featuredict.keys())].copy(deep = True)
        # check that ydepend variable is of string type
        if not isinstance(ydepend, str):
            raise TypeError("""ydepend not string, dependent variable must be single column name""")
        # check verbose log level if not None
        if verbose:

            if verbose not in [0, 1, 2]:
                raise ValueError("""Verbose flag must be set to level 0, 1 or 2.
                                \nLevel 0: Debug
                                \nLevel 1: Warning
                                \nLevel 2: Info""")

            # create log dict
            log_dict = {0: logging.DEBUG,
                        1: logging.WARNING,
                        2: logging.INFO}

            logging.basicConfig(format="""%(asctime)s:[%(filename)s:%(lineno)s - %(funcName)20s()]
                                          %(levelname)s:\n%(message)s""", level=log_dict[verbose])
            logging.info("Logger started....")


        try:
            agg_results = aggregate_func(list(range(100)))
            if hasattr(agg_results, '__len__'):
                raise ValueError("""aggregate_func must return scalar""")
        except Exception as e:
            raise TypeError("""aggregate_func must work on arrays of data and yield scalar
                                \nError: {}""".format(e))

        # assign parameters to class instance
        self.modelobj = modelobj
        self.model_df = model_df.copy(deep = True)
        self.ydepend = ydepend
        self.aggregate_func = aggregate_func
        # subset down cat_df to only those features in featuredict
        self.cat_df = self.cat_df[list(self.featuredict.keys())]
        # instantiate self.outputs
        self.outputs = False
        # check groupby vars
        if not groupbyvars:
            raise ValueError("""groupbyvars must be a list of grouping variables and cannot be None""")
        self.groupbyvars = list(groupbyvars)

    def predict(self):
        """
        create predictions based on trained model object, dataframe, and dependent variables
        :return: dataframe with prediction column
        """
        logging.info("""Creating predictions using modelobj.
                        \nModelobj class name: {}""".format(self.modelobj.__class__.__name__))
        # create predictions
        preds = self.modelobj.predict(self.model_df.loc[:, self.model_df.columns != self.ydepend])#self.model_df.loc[:, self.model_df.columns != self.ydepend])
        # calculate error
        diff = preds - self.cat_df.loc[:, self.ydepend]
        # assign errors
        self.cat_df['errors'] = diff
        # assign predictions
        logging.info('Assigning predictions to instance dataframe')
        self.cat_df['predictedYSmooth'] = preds
        self.model_df['predictedYSmooth'] = preds
        # return
        return self.cat_df

    @abc.abstractmethod
    def transform_function(self, group):
        # method to operate on slices of data within groups
        pass

    @abc.abstractmethod
    def var_check(self, col=None,
                  groupby=None):
        # method to check which type of variable the column is and perform operations specific
        # to that variable type
        pass

    def run(self):
        """
        main run engine. Iterate over columns specified in featuredict, and perform anlaysis
        :return: None - does put outputs in place
        """
        # testing run function
        # run the prediction function first to assign the errors to the dataframe
        self.predict()
        # create placeholder for outputs
        placeholder = []
        # create placeholder for all insights
        insights_df = pd.DataFrame()
        logging.info('Running main program. Iterating over columns and applying functions depednent on datatype')
        for col in self.cat_df.columns[~self.cat_df.columns.isin(['errors', 'predictedYSmooth', self.ydepend])]:

            # column placeholder
            colhold = []

            for groupby in self.groupbyvars:
                logging.info("""Currently working on column: {}
                                \nGroupby: {}\n""".format(col, groupby))
                # check if we are a col that is the groupbyvar3
                if col != groupby:
                    json_out = self.var_check(col=col,
                                            groupby=groupby)
                    # append to placeholder
                    colhold.append(json_out)

                else:
                    logging.info("""Creating accuracy metric for groupby variable: {}""".format(groupby))
                    # create error metrics for slices of groupby data
                    acc = self.create_accuracy(groupby=groupby)
                    # append to insights dataframe placeholder
                    insights_df = insights_df.append(acc)

            # map all of the same columns errors to the first element and
            # append to placeholder
            # dont append if placeholder is empty due to col being the same as groupby
            if len(colhold) > 0:
                placeholder.append(utils.flatten_json(colhold))

        logging.info('Converting accuracy outputs to json format')
        # finally convert insights_df into json object
        insights_json = utils.to_json(insights_df, vartype='Accuracy')
        # append to outputs
        placeholder.append(insights_json)
        # assign placeholder final outputs to class instance
        self.outputs = placeholder

    def create_accuracy(self,
                     groupby=None):
        """
        create error metrics for each slice of the groupby variable. i.e. if groupby is type of wine,
        create error metric for all white wine, then all red wine.
        :param groupby: groupby variable -- str -- i.e. 'Type'
        :return: accuracy dataframe for groupby variable
        """
        # use this as an opportunity to capture error metrics for the groupby variable
        # create a partial func by pre-filling in the parameters for create_insights
        insights = partial(utils.create_insights, group_var=groupby,
                           error_type='MSE')

        acc = self.cat_df.groupby(groupby).apply(insights)
        # drop the grouping indexing
        acc.reset_index(drop=True, inplace=True)
        # append to insights_df
        return acc

    def continuous_slice(self, group, col=None,
                         groupby=None,
                         vartype='Continuous'):
        """
        continuous_slice operates on portions of the data that correspond to a particular group of data from the groupby
        variable. For instance, if working on the wine quality dataset with Type representing your groupby variable, then
        continuous_slice would operate on 'White' wine data
        :param group: slice of data with columns, etc.
        :param col: current continuous variable being operated on
        :param vartype: continuous
        :return: transformed data with col data max, errPos mean, errNeg mean, and prediction means for this group
        """
        # check right vartype
        assert vartype in ['Continuous', 'Categorical'], 'Vartype must be Categorical or Continuous'
        # create percentiles for specific grouping of variables of interest
        group_vecs = utils.getVectors(group)
        # test
        # if more than 100 values in the group, use percentile bins
        if group.shape[0] > 100:
            logging.info("""Creating percentile bins for current continuous grouping""")
            group['fixed_bins'] = np.digitize(group.loc[:, col],
                                              sorted(list(set(group_vecs.loc[:, col]))), right=True)
        else:
            logging.warning("""Slice of data less than 100 continuous observations, using raw data as opposed to percentile groups.
                                            \nGroup size: {}""".format(group.shape))
            group['fixed_bins'] = group.loc[:, col]

        # create partial function for error transform (pos/neg split and reshape)
        trans_partial = partial(self.transform_function,
                                col=col,
                                groupby=groupby,
                                vartype=vartype)

        logging.info("""Applying transform function to continuous bins""")
        # group by bins
        errors = group.groupby('fixed_bins').apply(trans_partial)
        # final data prep for continuous errors case
        # and finalize errors dataframe processing
        errors.reset_index(drop=True, inplace=True)
        # rename columns
        errors.rename(columns={groupby: 'groupByValue'}, inplace=True)
        errors['groupByVarName'] = groupby
        return errors

    def save(self, fpath = ''):
        """
        save html output to disc
        :param fpath: file path to save html file to
        :return: None
        """
        if not self.outputs:
            RuntimeError('Must run WhiteBoxError.run() on data to store outputs')
            logging.warning("""Must run WhiteBoxError.run() before calling save method""")

        # change html output based on used class
        called_class = self.__class__.__name__
        # create html type dict
        html_type = {'WhiteBoxSensitivity': 'html_sensitivity',
                     'WhiteBoxError': 'html_error'}
        logging.info("""creating html output for type: {}""".format(html_type[called_class]))
        # create HTML output
        html_out = utils.createMLErrorHTML(str(self.outputs), self.ydepend,
                                            htmltype = html_type[called_class])
        # save html_out to disk
        with open(fpath, 'w') as outfile:
            logging.info("""Writing html file out to disk""")
            outfile.write(html_out)

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

    def __init__(self,
                 modelobj,
                 model_df,
                 ydepend,
                 cat_df=None,
                 featuredict=None,
                 groupbyvars=None,
                 aggregate_func=np.mean,
                 verbose=0):

        """
        :param modelobj: sklearn model object
        :param model_df: Pandas Dataframe used to build/train model
        :param ydepend: Y dependent variable
        :param cat_df: Pandas Dataframe of raw data - with categorical datatypes
        :param featuredict: Subset and rename columns
        :param groupbyvars: grouping variables
        :param aggregate_func: numpy aggregate function like np.mean
        :param verbose: Logging level
        """
        super(WhiteBoxError, self).__init__(modelobj,
                                      model_df,
                                      ydepend,
                                      cat_df=cat_df,
                                      featuredict=featuredict,
                                      groupbyvars=groupbyvars,
                                      aggregate_func=aggregate_func,
                                      verbose=verbose)

    def transform_function(self,
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
        # split out positive vs negative errors
        errors = group_copy['errors']
        # create separate columns for pos or neg errors
        errors = pd.concat([errors[errors > 0], errors[errors < 0]], axis=1)
        # rename error columns
        errors.columns = ['errPos', 'errNeg']
        # merge back with original data
        toreturn = pd.concat([group_copy.loc[:, group_copy.columns != 'errors'], errors], axis=1)
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
            errors = pd.DataFrame({col: toreturn[col].max(),
                                 groupby: toreturn[groupby].mode(),
                                 'predictedYSmooth': toreturn['predictedYSmooth'].mean(),
                                 'errPos': self.aggregate_func(toreturn['errPos']),
                                 'errNeg': self.aggregate_func(toreturn['errNeg'])})

            return errors

    def var_check(self,
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
        if is_categorical_dtype(self.cat_df.loc[:, col]):
            logging.info("""Column determined as categorical datatype, transforming data for categorical column
                            \nColumn: {}
                            \nGroup: {}""".format(col, groupby))
            # set variable type
            vartype = 'Categorical'
            # create a partial function from transform_function to fill in column and variable type
            categorical_partial = partial(self.transform_function,
                                          col=col,
                                          groupby=groupby,
                                          vartype=vartype)
            # slice over the groupby variable and the categories within the current column
            errors = self.cat_df[col_indices].groupby([groupby, col]).apply(categorical_partial)
            # final categorical transformations
            errors.reset_index(inplace=True)
            # rename
            errors.rename(columns={groupby: 'groupByValue'}, inplace=True)
            # rename columns based on user featuredict input
            errors.rename(columns=self.featuredict, inplace=True)
            # assign groupby variable to the errors dataframe
            errors['groupByVarName'] = groupby

        else:
            logging.info("""Column determined as continuous datatype, transforming data for continuous column
                                        \nColumn: {}
                                        \nGroup: {}""".format(col, groupby))
            # set variable type
            vartype = 'Continuous'
            # create partial function to fill in col and vartype of continuous_slice
            cont_slice_partial = partial(self.continuous_slice,
                                         col=col,
                                         vartype=vartype,
                                         groupby=groupby)
            # groupby the groupby variable on subset of columns and apply cont_slice_partial
            errors = self.cat_df[col_indices].groupby(groupby).apply(cont_slice_partial)
            # rename columns based on user featuredict input
            errors.rename(columns=self.featuredict, inplace=True)

        # json out

        errors = errors.replace(np.nan, 'null')
        # convert to json structure
        json_out = utils.to_json(errors, vartype=vartype, html_type='error',
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
                 verbose=0,
                 std_num = 1):
        """
        :param modelobj: sklearn model object
        :param model_df: Pandas Dataframe used to build/train model
        :param ydepend: Y dependent variable
        :param cat_df: Pandas Dataframe of raw data - with categorical datatypes
        :param featuredict: Subset and rename columns
        :param groupbyvars: grouping variables
        :param aggregate_func: function to aggregate sensitivity results by group
        :param verbose: Logging level
        :param std_num: Standard deviation adjustment
        """

        if std_num not in [-3, -2, -1, 1, 2, 3]:
            raise ValueError("""Standard deviation adjustment needs to be -3, -2, -1, 1, 2, or 3
                                \nCurrently value: {}""".format(std_num))

        self.std_num = std_num

        super(WhiteBoxSensitivity, self).__init__(modelobj,
                                      model_df,
                                      ydepend,
                                      cat_df=cat_df,
                                      featuredict=featuredict,
                                      groupbyvars=groupbyvars,
                                      aggregate_func=aggregate_func,
                                      verbose=verbose)

    def transform_function(self,
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
        print(group.head())
        if vartype == 'Continuous':
            logging.info(""""Returning aggregate values for group of continuous variable in transform_function of WhiteBoxSensitivity.
                            \nColumn: {}
                            \nGroup: {}
                            \nGroup shape: {}""".format(col, groupby, group.shape))
            # return the max value for the Continuous case
            errors = pd.DataFrame({col: group[col].max(),
                                 groupby: group[groupby].mode(),
                                 'predictedYSmooth': self.aggregate_func(group['diff'])})
        else:
            logging.info(""""Returning aggregate values for group of categorical variable in transform_function of WhiteBoxSensitvity.
                                        \nColumn: {}
                                        \nGroup: {}
                                        \nGroup shape: {}""".format(col, groupby, group.shape))
            # return the mode for the categorical case
            errors = pd.DataFrame({col: group[col].mode(),
                                   groupby: group[groupby].mode(),
                                   'predictedYSmooth': self.aggregate_func(group['diff'])})

        return errors

    def var_check(self,
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
        if is_categorical_dtype(self.cat_df.loc[:, col]):
            logging.info("""Column determined as categorical datatype, transforming data for categorical column
                                        \nColumn: {}
                                        \nGroup: {}""".format(col, groupby))
            # set variable type
            vartype = 'Categorical'
            # pull out the categorical dummy columns that match the current column
            all_type_cols = copydf.filter(regex='{}_*'.format(col)).columns
            # find the mode from the original cat_df for this column
            incremental_val = self.cat_df[col].mode().values[0]
            # find the columns within all_type_cols related to the mode_val
            mode_col = list(filter(lambda x: incremental_val in x, all_type_cols))
            # convert mode cols to all 1's
            copydf.loc[:, mode_col] = 1
            # convert all other non mode cols to zeros
            non_mode_col = list(filter(lambda x: incremental_val not in x, all_type_cols))
            copydf.loc[:, non_mode_col] = 0
            # make predictions with the switches to the dataset
            copydf['new_predictions'] = self.modelobj.predict(copydf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                  'predictedYSmooth'])])
            # calculate difference between actual predictions and new_predictions
            self.cat_df['diff'] = copydf['new_predictions'] - copydf['predictedYSmooth']
            # create a partial function from transform_function to fill in column and variable type
            categorical_partial = partial(self.transform_function,
                                          col=col,
                                          groupby=groupby,
                                          vartype=vartype)
            # create mask of data to select rows that are not equal to the mode of the category. This will prevent
            # blank displays in the HTML
            mode_mask = self.cat_df[col] != incremental_val
            # slice over the groupby variable and the categories within the current column
            sensitivity = self.cat_df[mode_mask][col_indices].groupby([groupby, col]).apply(categorical_partial)
            # final categorical transformations
            #sensitivity.reset_index(inplace=True)
            # rename
            sensitivity.rename(columns={groupby: 'groupByValue'}, inplace=True)
            # rename columns based on user featuredict input
            sensitivity.rename(columns=self.featuredict, inplace=True)
            # assign groupby variable to the errors dataframe
            sensitivity['groupByVarName'] = groupby

        else:
            logging.info("""Column determined as continuous datatype, transforming data for continuous column
                                                    \nColumn: {}
                                                    \nGroup: {}""".format(col, groupby))
            # set variable type
            vartype = 'Continuous'
            incremental_val = copydf[col].std() * self.std_num
            # tweak the currently column by the incremental_val
            copydf[col] = copydf[col] + incremental_val
            # make predictions with the switches to the dataset
            copydf['new_predictions'] = self.modelobj.predict(copydf.loc[:, ~copydf.columns.isin([self.ydepend,
                                                                                                  'predictedYSmooth'])])
            # calculate difference between actual predictions and new_predictions
            self.cat_df['diff'] = copydf['new_predictions'] - copydf['predictedYSmooth']
            # create partial function to fill in col and vartype of continuous_slice
            cont_slice_partial = partial(self.continuous_slice,
                                         col=col,
                                         vartype=vartype,
                                         groupby=groupby)
            # groupby the groupby variable on subset of columns and apply cont_slice_partial
            sensitivity = self.cat_df[col_indices].groupby(groupby).apply(cont_slice_partial)
            # rename columns based on user featuredict input
            sensitivity.rename(columns=self.featuredict, inplace=True)

        # json out
        sensitivity = sensitivity.replace(np.nan, 'null')
        logging.info("""Converting output to json type using to_json utility function""")
        # convert to json structure
        json_out = utils.to_json(sensitivity, vartype=vartype, html_type = 'sensitivity',
                                 incremental_val=incremental_val)
        # return json_out
        return json_out