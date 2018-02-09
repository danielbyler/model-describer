#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from abc import abstractmethod, ABCMeta
from functools import partial
import logging

import numpy as np
from pandas import core, DataFrame, melt
from sklearn.exceptions import NotFittedError

try:
    from utils import (prob_acc,
                       create_insights,
                       getvectors, flatten_json,
                       to_json, createmlerror_html,
                       Settings)
except:
    from whitebox.utils import (prob_acc,
                                create_insights,
                                getvectors, flatten_json,
                                to_json, createmlerror_html,
                                Settings)


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
        :param verbose: set verbose level -- 0 = debug, 1 = warning, 2 = error
        """

        # basic parameter checks
        if not hasattr(modelobj, 'predict'):
            raise ValueError("""modelObj does not have predict method. 
                                WhiteBoxError only works with model 
                                objects with predict method""")
        # need to ensure modelobj has previously been fitted, otherwise
        # raise NotFittedError
        try:
            # try predicting on model dataframe
            modelobj.predict(model_df.loc[:, model_df.columns != ydepend])
        except NotFittedError as e:
            # raise exception and not fitted error
            raise Exception('{}\nPlease fit model: {}'.format(e, modelobj.__class__))

        if not isinstance(model_df, core.frame.DataFrame):
            raise TypeError("""model_df variable not pandas dataframe.
                                WhiteBoxError only works with dataframe objects""")

        if featuredict is not None and not isinstance(featuredict, dict):
            raise TypeError("""When used Featuredict needs to be of type dictionary. 
                                Keys are original column
                                names and values are formatted column names:
                                \n{sloppy.name.here: Sloppy Name}""")
        # toggle featuredict depending on whether user specifys cat_df and leaves
        # featuredict blank
        if featuredict is None and cat_df is not None:
            self.featuredict = {col: col for col in cat_df.columns}
        elif featuredict is None:
            self.featuredict = {col: col for col in model_df.columns}
        else:
            self.featuredict = featuredict

        if isinstance(cat_df, core.frame.DataFrame):
            # check tthat the number of rows from cat_df matches that of model_df
            if model_df.shape[0] != cat_df.shape[0]:
                raise StandardError("""Misaligned rows. \norig_df shape: {}
                                        \ndummy_df shape: {}
                                        """.format(
                                                    model_df.shape[0],
                                                    cat_df.shape[0]))
            # assign cat_df to class instance and subset to featuredict keys
            self.cat_df = cat_df[list(self.featuredict.keys())].copy(deep=True)

        else:
            # check that cat_df is not none and if it's not a pandas dataframe, throw warning
            if not isinstance(cat_df, type(None)):
                warnings.warn(
                                """cat_df is not None and not a 
                                pd.core.frame.DataFrame. 
                                Default becomes model_df
                                and may not be intended user behavior""",
                                UserWarning)
            # assign cat_df to class instance and subset based on featuredict keys
            self.cat_df = model_df[list(self.featuredict.keys())].copy(deep=True)
        # check that ydepend variable is of string type
        if not isinstance(ydepend, str):
            raise TypeError("""ydepend not string, dependent variable 
                                must be single column name""")
        # check verbose log level if not None
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

        try:
            agg_results = aggregate_func(list(range(100)))
            if hasattr(agg_results, '__len__'):
                raise ValueError("""aggregate_func must return scalar""")
        except Exception as e:
            raise TypeError(
                            """aggregate_func must work on 
                            arrays of data and yield scalar
                            \nError: {}""".format(e))

        if error_type not in Settings.supported_agg_errors:
            raise ValueError("""Supported aggregate error metrics are MSE, MAE, RMSE
                            \nInput value: {} not supported""".format(error_type))

        # assign parameters to class instance
        self.error_type = error_type
        self.modelobj = modelobj
        self.model_df = model_df.copy(deep=True)
        self.ydepend = ydepend
        self.aggregate_func = aggregate_func
        # subset down cat_df to only those features in featuredict
        self.cat_df = self.cat_df[list(self.featuredict.keys())]
        # instantiate self.outputs
        self.outputs = False
        # check groupby vars
        if not groupbyvars:
            raise ValueError(
                            """groupbyvars must be a list of grouping 
                            variables and cannot be None""")
        self.groupbyvars = list(groupbyvars)

        # determine if in classification problem or regression problem
        if hasattr(self.modelobj, 'predict_proba'):
            # if classification setting, secure the predicted class probabilities
            self.predict_engine = getattr(self.modelobj, 'predict_proba')
            self.model_type = 'classification'
        else:
            # use the regular predict function
            self.predict_engine = getattr(self.modelobj, 'predict')
            self.model_type = 'regression'

        # create instance wide percentiles for all numeric columns
        self.percentile_vecs = getvectors(self.cat_df)
        # create percentile out
        self._percentiles_out(self.percentile_vecs.select_dtypes(include=[np.number]))

    def _predict(self):
        """
        create predictions based on trained model object, dataframe, and dependent variables
        :return: dataframe with prediction column
        """
        logging.info("""Creating predictions using modelobj.
                        \nModelobj class name: {}""".format(self.modelobj.__class__.__name__))
        # create predictions
        preds = self.predict_engine(
                                    self.model_df.loc[:, self.model_df.columns != self.ydepend])

        if self.model_type == 'regression':
            # calculate error
            diff = preds - self.cat_df.loc[:, self.ydepend]
        elif self.model_type == 'classification':
            # select the prediction probabilities for the class labeled 1
            preds = preds[:, 1].tolist()
            # create a lookup of class labels to numbers
            class_lookup = {class_: num for num, class_ in enumerate(self.modelobj.classes_)}
            # convert the ydepend column to numeric
            actual = self.cat_df.loc[:, self.ydepend].apply(lambda x: class_lookup[x])
            # calculate the difference between actual and predicted probabilities
            diff = [prob_acc(true_class=actual[idx], pred_prob=pred) for idx, pred in enumerate(preds)]
        else:
            raise RuntimeError(""""unsupported model type
                                    \nInput Model Type: {}""".format(self.model_type))

        # assign errors
        self.cat_df['errors'] = diff
        # assign predictions
        logging.info('Assigning predictions to instance dataframe')
        self.cat_df['predictedYSmooth'] = preds
        self.model_df['predictedYSmooth'] = preds
        # return
        return self.cat_df

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
        # create a partial func by pre-filling in the parameters for create_insights
        if self.model_type == 'classification':
            error_type = 'RAW'
        if self.model_type == 'regression':
            error_type = self.error_type
        insights = partial(create_insights, group_var=groupby,
                           error_type=error_type)

        acc = self.cat_df.groupby(groupby).apply(insights)
        # drop the grouping indexing
        acc.reset_index(drop=True, inplace=True)
        # append to insights_df
        return acc

    def _percentiles_out(self,
                         groupvecs):
        # send the percentiles to to_json to create percentile bars in UI
        percentiles = groupvecs.reset_index().rename(columns={"index": 'percentile'})
        # capture 10, 25, 50, 75, 90 percentiles
        final_percentiles = percentiles[percentiles.percentile.str
                                        .contains('10%|25%|50%|75%|90%')].copy(deep=True)
        # rename to featuredict values
        final_percentiles.rename(columns=self.featuredict, inplace=True)
        # melt to long format
        percentiles_melted = melt(final_percentiles, id_vars='percentile')
        # convert to_json
        self.percentiles = to_json(dataframe=percentiles_melted,
                                   vartype='Percentile',
                                   html_type='percentile',
                                   incremental_val=None)

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

        # create partial function for error transform (pos/neg split and reshape)
        trans_partial = partial(self._transform_function,
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
        insights_df = DataFrame()
        logging.info("""Running main program. Iterating over 
                    columns and applying functions depednent on datatype""")
        for col in self.cat_df.columns[
                                        ~self.cat_df.columns.isin(['errors',
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
                placeholder.append(flatten_json(colhold))

        logging.info('Converting accuracy outputs to json format')
        # finally convert insights_df into json object
        insights_json = to_json(insights_df, vartype='Accuracy')
        # append to outputs
        placeholder.append(insights_json)
        # append percentiles
        placeholder.append(self.percentiles)
        # assign placeholder final outputs to class instance
        self.outputs = placeholder

    def save(self, fpath=''):
        """
        save html output to disc
        :param fpath: file path to save html file to
        :return: None
        """
        if not self.outputs:
            RuntimeError("""Must run WhiteBoxError.run() 
                        on data to store outputs""")
            logging.warning("""Must run WhiteBoxError.run() 
                            before calling save method""")

        # change html output based on used class
        called_class = self.__class__.__name__

        logging.info("""creating html output for type: {}""".format(Settings.html_type[called_class]))

        # tweak self.ydepend if classification case (add dominate class)
        if self.model_type == 'classification':
            ydepend_out = '{}: {}'.format(self.ydepend, self.modelobj.classes_[1])
        else:
            # regression case
            ydepend_out = self.ydepend

        # create HTML output
        html_out = createmlerror_html(
                                            str(self.outputs),
                                            ydepend_out,
                                            htmltype=Settings.html_type[called_class])
        # save html_out to disk
        with open(fpath, 'w') as outfile:
            logging.info("""Writing html file out to disk""")
            outfile.write(html_out)
