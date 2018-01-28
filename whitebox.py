import pandas as pd
from utils import to_json, getVectors, create_insights, createMLErrorHTML, flatten_json
import abc
import numpy as np
from itertools import product
from sklearn.exceptions import NotFittedError
import warnings
from functools import partial

class WhiteBoxBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self,
                 modelobj,
                 model_df,
                 ydepend,
                 cat_df = None,
                 featuredict = None):

        # basic parameter checks
        if not hasattr(modelobj, 'predict'):
            raise ValueError('modelObj does not have predict method.' \
                                             ' WhiteBoxError only works with model objects with predict method')
        # need to ensure modelobj has previously been fitted, otherwise raise NotFittedError
        try:
            # try predicting on model dataframe
            modelobj.predict(model_df.loc[:, model_df.columns != ydepend])
        except NotFittedError as e:
            # raise exception and not fitted error
            raise Exception('{}\nPlease fit model: {}'.format(e, modelobj.__class__))

        if not isinstance(model_df, pd.core.frame.DataFrame):
            raise TypeError('model_df variable not pandas dataframe. ' \
                                                             'WhiteBoxError only works with dataframe objects')

        if featuredict is not None and not isinstance(featuredict, dict):
            raise TypeError('When used Featuredict needs to be of type dictionary. Keys are original column'\
                            'names and values are formatted column names:'\
                            '\n{sloppy.name.here: Sloppy Name}')
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
                raise StandardError('Misaligned rows. \norig_df shape: {} ' \
                                                          '\ndummy_df shape: {}'.format(model_df.shape[0],
                                                                                        cat_df.shape[0]))
            # assign cat_df to class instance and subset to featuredict keys
            self.cat_df = cat_df[self.featuredict.keys()].copy(deep = True)

        else:
        	# check that cat_df is not none and if it's not a pandas dataframe, throw warning
            if not isinstance(cat_df, type(None)):
                warnings.warn('cat_df is not None and not a pd.core.frame.DataFrame. Default becomes model_df'\
                              'and may not be intended user behavior', UserWarning)
            # assign cat_df to class instance and subset based on featuredict keys
            self.cat_df = model_df[self.featuredict.keys()].copy(deep = True)
        # check that ydepend variable is of string type
        if not isinstance(ydepend, str):
            raise TypeError('ydepend not string, dependent variable must be single column name')
        # assing parameters to class instance
        self.modelobj = modelobj
        self.model_df = model_df.copy(deep = True)
        self.ydepend = ydepend

        # subset down cat_df to only those features in featuredict
        self.cat_df = self.cat_df[self.featuredict.keys()]

    def predict(self):
        """
        create predictions based on trained model object, dataframe, and dependent variables
        :param modelObjc: fitted model object
        :param dataframe: dataframe with X data
        :param yDepend: dependent variables (predictor)
        :return: dataframe with prediction column
        """
        # convert categories to numeric using underlying category datatype
        # predict_df = convert_categorical_independent(self.model_df)
        # create predictions
        preds = self.modelobj.predict(self.model_df.loc[:, self.model_df.columns != self.ydepend])#self.model_df.loc[:, self.model_df.columns != self.ydepend])
        # calculate error
        diff = preds - self.cat_df.loc[:, self.ydepend]
        # assign errors
        self.cat_df['errors'] = diff
        # assign predictions
        self.cat_df['predictedYSmooth'] = preds
        # return
        return self.cat_df

    @abc.abstractmethod
    def transform_function(self, group):
        # method to operate on slices of data within groups
        pass

    @abc.abstractmethod
    def run(self):
        """
        method to run over users data, manipulate various groups to explain prediction quality, errors,
        etc. during specific regions of the data, store those manipulations and output them
        :return: undetermined
        """
        pass

    @abc.abstractmethod
    def save(self, fpath = ''):
        """
        method to save HTML output to disk
        :return: html output
        """
        pass

class WhiteBoxError(WhiteBoxBase):

    def __init__(self,
                 modelobj,
                 model_df,
                 ydepend,
                 cat_df = None,
                 featuredict = None,
                 groupbyvars = None):

        if groupbyvars is None:
            raise TypeError('groupbyvars cannot be none, must be a list of grouping variables')

        self.groupbyvars = groupbyvars

        super(WhiteBoxError, self).__init__(modelobj,
                                      model_df,
                                      ydepend,
                                      cat_df = cat_df,
                                      featuredict = featuredict)

    @staticmethod
    def transform_function(group,
                           groupby = 'Type',
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
            # return the mean of the columns
            return toreturn.mean()
        else:
            errors = pd.DataFrame({col: toreturn[col].max(),
                                 groupby: toreturn[groupby].mode(),
                                 'predictedYSmooth': toreturn['predictedYSmooth'].mean(),
                                 'errPos': toreturn['errPos'].mean(),
                                 'errNeg': toreturn['errNeg'].mean()})

            return errors

    @staticmethod
    def continuous_slice(group, col=None,
                         groupby = None,
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
        # create percentiles for specific grouping of variables
        group_vecs = getVectors(group)

        # if more than 100 values in the group, use percentile bins
        if group.shape[0] > 100:
            group['fixed_bins'] = np.digitize(group.loc[:, col],
                                              sorted(list(set(group_vecs.loc[:, col]))), right=True)
        else:
            group['fixed_bins'] = group.loc[:, col]

        # create partial function for error transform (pos/neg split and reshape)
        trans_partial = partial(WhiteBoxError.transform_function,
                                col=col,
                                groupby = groupby,
                                vartype=vartype)

        # group by bins
        errors = group.groupby('fixed_bins').apply(trans_partial)
        # final data prep for continuous errors case
        # and finalize errors dataframe processing
        errors.reset_index(drop=True, inplace=True)
        # drop rows that are all NaN
        #errors.dropna(how='all', axis=0, inplace = True)
        # errors.reset_index(drop = True, inplace = True)
        errors.rename(columns={groupby: 'groupByValue'}, inplace=True)
        errors['groupByVarName'] = groupby
        errors['highlight'] = 'N'
        '''
        # merge the data back with the groups perdcentile buckets
        final_out = pd.merge(pd.DataFrame(group_vecs[col]), errors,
                             left_on=col, right_on=col, how='left')

        print(final_out)

        final_out.fillna(method='ffill', inplace=True)

        # fill na values to null
        final_out.dropna(axis = 0, how = 'any', inplace = True)
        final_out.replace(np.nan, 'null', inplace=True)
        '''
        return errors

    def run(self):
        # run the prediction function first to assign the errors to the dataframe
        self.predict()
        # create placeholder for outputs
        placeholder = []
        # create placeholder for all insights
        insights_df = pd.DataFrame()

        #for col, groupby in product(self.cat_df.columns[~self.cat_df.columns.isin(['errors', 'predictedYSmooth',
        #                                                                           self.ydepend])], self.groupbyvars):

        for col in self.cat_df.columns[~self.cat_df.columns.isin(['errors', 'predictedYSmooth', self.ydepend])]:

            # column placeholder
            colhold = []

            for groupby in self.groupbyvars:

                # check if we are a col that is the groupbyvar3
                if col != groupby:
                    print('Currently on col: {}\nGroupby: {}'.format(col, groupby))
                    # subset col indices
                    col_indices = [col, 'errors', 'predictedYSmooth', groupby]
                    # check if categorical
                    if isinstance(self.cat_df.loc[:, col].dtype, pd.types.dtypes.CategoricalDtype):
                        # set variable type
                        print('FINALLY ON A CATEGORICAL VARIABLE')
                        vartype = 'Categorical'
                        # create a partial function from transform_function to fill in column and variable type
                        categorical_partial = partial(WhiteBoxError.transform_function,
                                                      col = col,
                                                      groupby = groupby,
                                                      vartype = vartype)
                        # slice over the groupby variable and the categories within the current column
                        errors = self.cat_df[col_indices].groupby([groupby, col]).apply(categorical_partial)
                        # final categorical transformations
                        errors.reset_index(inplace = True)
                        errors.rename(columns = {groupby: 'groupByValue'}, inplace = True)
                        errors['groupByVarName'] = groupby

                    else:
                        # set variable type
                        vartype = 'Continuous'
                        # create partial function to fill in col and vartype of continuous_slice
                        cont_slice_partial = partial(WhiteBoxError.continuous_slice,
                                                     col = col,
                                                     vartype = vartype,
                                                     groupby = groupby)
                        # groupby the groupby variable on subset of columns and apply cont_slice_partial
                        errors = self.cat_df[col_indices].groupby(groupby).apply(cont_slice_partial)


                    # json out
                    errors = errors.replace(np.nan, 'null')
                    print(errors)
                    json_out = to_json(errors, vartype = vartype)
                    # append to placeholder
                    colhold.append(json_out)

                else:
                    #todo this is happening multiple times when it should only occur once
                    # use this as an opportunity to capture error metrics for the groupby variable
                    # create a partial func by pre-filling in the parameters for create_insights
                    insights = partial(create_insights, group_var = groupby,
                                       error_type='MSE')

                    acc = self.cat_df.groupby(groupby).apply(insights)
                    # drop the grouping indexing
                    acc.reset_index(drop = True, inplace = True)
                    # append to insights_df
                    insights_df = insights_df.append(acc)

            # map all of the same columns errors to the first element and
            # append to placeholder
            placeholder.append(flatten_json(colhold))


        # finally convert insights_df into json object
        insights_json = to_json(insights_df, vartype = 'Accuracy')
        # append to outputs
        placeholder.append(insights_json)
        # flatten nested outputs

        # assign outputs to class
        self.outputs = placeholder #flatten_outputs(placeholder)

    def save(self, fpath = ''):
        if not self.outputs:
            RuntimeError('Must run WhiteBoxError.run() on data to store outputs')

        # create HTML output
        html_out = createMLErrorHTML(str(self.outputs), self.ydepend)
        # save html_out to disk
        with open(fpath, 'w') as outfile:
            outfile.write(html_out)