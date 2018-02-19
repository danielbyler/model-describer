#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from abc import abstractmethod, ABCMeta
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted, check_consistent_length

#TODO fix imports
try:
    import utils as wb_utils
except:
    import whitebox.utils as wb_utils


class WhiteBoxBase(object):

    __metaclass__ = ABCMeta

    @abstractmethod
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
                    verbose=None):
        """
        initialize base class
        :param modelobj: fitted sklearn model object
        :param model_df: data used for training sklearn object
        :param ydepend: dependent variable
        :param cat_df: formatted dataset with categorical dtypes specified,
                       and non-dummy categories
        :param featuredict: prettty printing and subsetting analysis
        :param groupbyvars: grouping variables
        :param error_type: Aggregate error metric i.e. MSE, MAE, RMSE
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
        self._cat_df = cat_df
        # check modelobj
        self._modelobj = modelobj
        # check verbosity
        self._check_verbose(verbose)
        # check for classification or regression
        self._is_regression_classification()
        # check featuredict
        self._check_featuredict(featuredict)
        # create reverse featuredict
        self._reverse_featuredict = {val: key for key, val in self.featuredict.items()}
        # check aggregate func
        self._check_agg_func(aggregate_func)
        # check error type supported
        self.error_type = error_type
        # assign dependent variable
        self.ydepend = ydepend
        # instantiate self.outputs
        self.outputs = False
        # if user specified featuredict, use column mappings otherwise use original groupby
        self.groupbyvars = groupbyvars
        # determine the calling class (WhiteBoxError or WhiteBoxSensitivity)
        self.called_class = self.__class__.__name__
        # format ydepend, catdf, groupbyvars
        self._formatter(autoformat)
        # create percentiles
        self._run_percentiles()

    @property
    def cat_df(self):
        return self._cat_df

    @cat_df.setter
    def cat_df(self, value):
        """
        ensure validity of assigned cat_df - must have same length of model_df
        and if None, is replaced by model_df
        :param value: user defined cat_df
        :return: NA
        """
        # if cat_df not assigned, use model_df
        if value is None:
            warnings.warn(wb_utils.ErrorWarningMsgs.warning_msgs['cat_df'])
            self._cat_df = self._model_df
        else:
            # check both model_df and cat_df have the same length
            check_consistent_length(value, self._model_df)
            # check index's are equal
            if not all(value.index == self._model_df.index):
                raise ValueError("""Indices of cat_df and model_df are not aligned. Ensure Index's are 
                                    \nexactly the same before WhiteBox use.""")
            # create copy of users dataframe as to not adjust their actual dataframe object
            # they are working on
            value = value.copy(deep=True)
            # reset users index in case of multi index or otherwise
            self._cat_df = value.reset_index(drop=True)

    @property
    def modelobj(self):
        return self._modelobj

    @modelobj.setter
    def modelobj(self, value):
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

        self._modelobj = value

    def _check_featuredict(self, featuredict):
        """
        check user defined featuredict - if blank assign all dataframe columns
        :param featuredict: user defined featuredict mapping original col names to cleaned col names
        :return: NA
        """
        # featuredict blank
        if not featuredict:
            self.featuredict = {col: col for col in self._cat_df.columns}
        else:
            if not all([key in self._cat_df.columns for key in featuredict.keys()]):
                # find missing keys
                missing = list(set(featuredict.keys()).difference(set(self._cat_df.columns)))
                raise ValueError(wb_utils.ErrorWarningMsgs.error_msgs['featuredict'].format(missing))
            self.featuredict = featuredict

    def _check_verbose(self, verbose):
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

    def _is_regression_classification(self):
        """
        determined whether users modelobj is regression or classification based on
        presence of predict_proba
        :return: NA
        """
        # determine if in classification problem or regression problem
        if hasattr(self._modelobj, 'predict_proba'):
            # if classification setting, secure the predicted class probabilities
            self.predict_engine = getattr(self._modelobj, 'predict_proba')
            self.model_type = 'classification'
        else:
            # use the regular predict function
            self.predict_engine = getattr(self._modelobj, 'predict')
            self.model_type = 'regression'

    def _check_agg_func(self, aggregate_func):
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

        self.aggregate_func = aggregate_func

    def _formatter(self, autoformat):
        # convert ydepend based on featuredict
        self.ydepend = self.featuredict[self.ydepend] if self.ydepend in list(self.featuredict.keys()) else self.ydepend
        # convert groupby's based on featuredict
        self.groupbyvars = [self.featuredict[group] if group in list(self.featuredict.keys()) else group for group in list(self.groupbyvars)]
        # subset down cat_df to only those features in featuredict
        self._cat_df = self._cat_df.loc[:, list(self.featuredict.keys())].rename(columns=self.featuredict)
        # rename model_df columns based on featuredict
        self._model_df = self._model_df.rename(columns=self.featuredict)

        if autoformat:
            warnings.warn(wb_utils.ErrorWarningMsgs.warning_msgs['auto_format'])
            # convert categorical dtypes to strings
            for cat in self._cat_df.select_dtypes(include=['category']).columns:
                self._cat_df.loc[:, cat] = self._cat_df.loc[:, cat].astype(str)


    def _run_percentiles(self):
        """
        create population percentiles, and group percentiles
        :return: NA
        """
        # create instance wide percentiles for all numeric columns
        self.percentile_vecs = wb_utils.getvectors(self._cat_df)
        # create percentile bars for final out
        self._percentiles_out()
        # create groupby percentiles
        self._group_percentiles_out = wb_utils.create_group_percentiles(self._cat_df,
                                                               self.groupbyvars)

    def _predict(self):
        """
        create predictions based on trained model object, dataframe, and dependent variables
        :return: dataframe with prediction column
        """
        logging.info("""Creating predictions using modelobj.
                        \nModelobj class name: {}""".format(self._modelobj.__class__.__name__))
        # create predictions
        preds = self.predict_engine(
                                    self._model_df.loc[:, self._model_df.columns != self.ydepend])

        if self.model_type == 'regression':
            # calculate error
            diff = preds - self._cat_df.loc[:, self.ydepend]
        elif self.model_type == 'classification':
            # select the prediction probabilities for the class labeled 1
            preds = preds[:, 1].tolist()
            # create a lookup of class labels to numbers
            class_lookup = {class_: num for num, class_ in enumerate(self._modelobj.classes_)}
            # convert the ydepend column to numeric
            actual = self._cat_df.loc[:, self.ydepend].apply(lambda x: class_lookup[x]).values.tolist()
            # calculate the difference between actual and predicted probabilities
            diff = [wb_utils.prob_acc(true_class=actual[idx], pred_prob=pred) for idx, pred in enumerate(preds)]
        else:
            raise RuntimeError(""""unsupported model type
                                    \nInput Model Type: {}""".format(self.model_type))

        # assign errors
        self._cat_df['errors'] = diff
        # assign predictions
        logging.info('Assigning predictions to instance dataframe')
        self._cat_df['predictedYSmooth'] = preds
        self._model_df['predictedYSmooth'] = preds
        # return
        return self._cat_df

    @abstractmethod
    def _transform_function(self, group):
        # method to operate on slices of data within groups
        pass

    @abstractmethod
    def _var_check(
                    self,
                    col=None,
                    groupby=None):
        """
        _var_check tests for categorical or continuous variable and performs operations
        dependent upon the var type
        :param col: current column being operated on
        :param groupby: current groupby level
        :return: NA
        """
        pass

    def _create_accuracy(
                            self,
                            groupby=None):
        """
        create error metrics for each slice of the groupby variable.
        i.e. if groupby is type of wine,
        create error metric for all white wine, then all red wine.
        :param groupby: groupby variable -- str -- i.e. 'Type'
        :return: accuracy dataframe for groupby variable
        """
        # use this as an opportunity to capture error metrics for the groupby variable
        if self.model_type == 'classification':
            error_type = 'RAW'
        if self.model_type == 'regression':
            error_type = self.error_type

        acc = self._cat_df.groupby(groupby).apply(wb_utils.create_insights,
                                                 group_var=groupby,
                                                 error_type=error_type)
        # drop the grouping indexing
        acc.reset_index(drop=True, inplace=True)
        # append to insights_df
        return acc

    def _percentiles_out(self):
        """
        Create designated percentiles for user interface percentile bars
            percentiles calculated include: 10, 25, 50, 75 and 90th percentiles
        :return: Save percnetiles to instance for retrieval in final output
        """
        # subset percentiles to only numeric variables
        numeric_vars = self.percentile_vecs.select_dtypes(include=[np.number])
        # send the percentiles to to_json to create percentile bars in UI
        percentiles = numeric_vars.reset_index().rename(columns={"index": 'percentile'})
        # capture 10, 25, 50, 75, 90 percentiles
        final_percentiles = percentiles[percentiles.percentile.str
                                        .contains('10%|25%|50%|75%|90%')].copy(deep=True)
        # melt to long format
        percentiles_melted = pd.melt(final_percentiles, id_vars='percentile')
        # convert to_json
        self.percentiles = wb_utils.to_json(dataframe=percentiles_melted,
                                   vartype='Percentile',
                                   html_type='percentile',
                                   incremental_val=None)
    #TODO add weighted schematics function
    def _continuous_slice(
                        self,
                        group,
                        col=None,
                        groupby=None,
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
        # if more than 100 values in the group, use percentile bins
        if group.shape[0] > 100:
            logging.info("""Creating percentile bins for current 
                            continuous grouping""")
            group['fixed_bins'] = np.digitize(group.loc[:, col],
                                              sorted(list(set(self.percentile_vecs.loc[:, col]))),
                                              right=True)
        else:
            logging.warning("""Slice of data less than 100 continuous observations, 
                                using raw data as opposed to percentile groups.
                                \nGroup size: {}""".format(group.shape))
            group['fixed_bins'] = group.loc[:, col]

        logging.info("""Applying transform function to continuous bins""")
        # group by bins
        errors = group.groupby('fixed_bins').apply(self._transform_function,
                                                   col=col,
                                                   groupby=groupby,
                                                   vartype=vartype)
        # final data prep for continuous errors case
        # and finalize errors dataframe processing
        errors.reset_index(drop=True, inplace=True)
        # rename columns
        errors.rename(columns={groupby: 'groupByValue'}, inplace=True)
        errors['groupByVarName'] = groupby
        return errors

    def run(self):
        """
        main run engine. Iterate over columns specified in featuredict,
        and perform anlaysis
        :return: None - does put outputs in place
        """
        # testing run function
        # run the prediction function first to assign the errors to the dataframe
        self._predict()
        # create placeholder for outputs
        placeholder = []
        # create placeholder for all insights
        insights_df = pd.DataFrame()
        logging.info("""Running main program. Iterating over 
                    columns and applying functions depednent on datatype""")
        for col in self._cat_df.columns[
                                        ~self._cat_df.columns.isin(['errors',
                                                                   'predictedYSmooth',
                                                                   self.ydepend])]:

            # column placeholder
            colhold = []

            for groupby in self.groupbyvars:
                logging.info("""Currently working on column: {}
                                \nGroupby: {}\n""".format(col, groupby))
                # check if we are a col that is the groupbyvar3
                if col != groupby:
                    json_out = self._var_check(
                                                col=col,
                                                groupby=groupby)
                    # append to placeholder
                    colhold.append(json_out)

                else:
                    logging.info(
                                """Creating accuracy metric for 
                                groupby variable: {}""".format(groupby))
                    # create error metrics for slices of groupby data
                    acc = self._create_accuracy(groupby=groupby)
                    # append to insights dataframe placeholder
                    insights_df = insights_df.append(acc)

            # map all of the same columns errors to the first element and
            # append to placeholder
            # dont append if placeholder is empty due to col being the same as groupby
            if len(colhold) > 0:
                placeholder.append(wb_utils.flatten_json(colhold))

        logging.info('Converting accuracy outputs to json format')
        # finally convert insights_df into json object
        insights_json = wb_utils.to_json(insights_df, vartype='Accuracy')
        # append to outputs
        placeholder.append(insights_json)
        # append percentiles
        placeholder.append(self.percentiles)
        # append groupby percnetiles
        placeholder.append(self._group_percentiles_out)
        # assign placeholder final outputs to class instance
        self.outputs = placeholder

    def save(self, fpath=''):
        """
        save html output to disc
        :param fpath: file path to save html file to
        :return: None
        """
        if not self.outputs:
            RuntimeError(wb_utils.ErrorWarningMsgs.error_msgs['wb_run_error'].format(self.called_class))
            logging.warning(wb_utils.ErrorWarningMsgs.error_msgs['wb_run_error'].format(self.called_class))

        logging.info("""creating html output for type: {}""".format(wb_utils.Settings.html_type[self.called_class]))

        # tweak self.ydepend if classification case (add dominate class)
        if self.model_type == 'classification':
            ydepend_out = '{}: {}'.format(self.ydepend, self._modelobj.classes_[1])
        else:
            # regression case
            ydepend_out = self.ydepend

        # create HTML output
        html_out = wb_utils.createmlerror_html(
                                            str(self.outputs),
                                            ydepend_out,
                                            htmltype=wb_utils.Settings.html_type[self.called_class])
        # save html_out to disk
        with open(fpath, 'w') as outfile:
            logging.info("""Writing html file out to disk""")
            outfile.write(html_out)
