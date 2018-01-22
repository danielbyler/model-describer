import pandas as pd
from utils.utils import to_json, getVectors, create_insights, createMLErrorHTML, convert_categorical_independent
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
                 cat_df = None):
        # assertion statements
        assert hasattr(modelobj, 'predict'), 'modelObj does not have predict method.' \
                                             ' WhiteBoxError only works with model objects with predict method'

        # need to ensure modelobj has previously been fitted, otherwise raise NotFittedError
        try:
            # try predicting on model dataframe
            modelobj.predict(model_df.loc[:, model_df.columns != ydepend])
        except NotFittedError as e:
            # raise exception and not fitted error
            raise Exception('{}\nPlease fit model: {}'.format(e, modelobj.__class__))

        assert isinstance(model_df, pd.core.frame.DataFrame), 'orig_df variable not pandas dataframe. ' \
                                                             'WhiteBoxError only works with dataframe objects'

        if isinstance(cat_df, pd.core.frame.DataFrame):
            assert model_df.shape[0] == cat_df.shape[0], 'Misaligned rows. \norig_df shape: {} ' \
                                                          '\ndummy_df shape: {}'.format(model_df.shape[0],
                                                                                        cat_df.shape[0])
            self.cat_df = cat_df.copy(deep = True)

        else:
            if not isinstance(cat_df, type(None)):
                warnings.warn('cat_df is not None and not a pd.core.frame.DataFrame. Default becomes model_df'\
                              'and may not be intended user behavior', UserWarning)

            self.cat_df = model_df.copy(deep = True)

        assert isinstance(ydepend, str), 'ydepend not string, dependent variable must be single column name'

        self.modelobj = modelobj
        self.model_df = model_df.copy(deep = True)
        self.ydepend = ydepend

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
        print(self.model_df.head())
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
        pass

class WhiteBoxError(WhiteBoxBase):

    def __init__(self,
                 modelobj,
                 model_df,
                 ydepend,
                 cat_df = None,
                 featuredict = None,
                 groupbyvars = None):

        if featuredict:
            assert isinstance(featuredict, dict), 'featuredict must be dictionary object mapping original column names to ' \
                                                  'new col names {org_column : pretty_name}. Only featuredict keys are used ' \
                                                  'in display if utilized'

        assert groupbyvars, 'groupbyvars cannot be none'

        self.featuredict = featuredict
        self.groupbyvars = groupbyvars

        super(WhiteBoxError, self).__init__(modelobj,
                                      model_df,
                                      ydepend,
                                      cat_df = cat_df)

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
        # create bins
        group['fixed_bins'] = np.digitize(group.loc[:, col],
                                          sorted(list(set(group_vecs.loc[:, col]))), right=True)

        # create partial function for error transform (pos/neg split and reshape)
        trans_partial = partial(WhiteBoxError.transform_function, col=col,
                                vartype=vartype)

        # group by bins
        errors = group.groupby('fixed_bins').apply(trans_partial)
        # final data prep for continuous errors case
        # and finalize errors dataframe processing
        errors.reset_index(drop=True, inplace=True)
        # drop rows that are all NaN
        errors.dropna(how='all', axis=0)
        # errors.reset_index(drop = True, inplace = True)
        errors.rename(columns={groupby: 'groupByValue'}, inplace=True)
        errors['groupByVarName'] = groupby
        errors['highlight'] = 'N'

        # fill na values to null
        errors.replace(np.nan, 'null', inplace=True)
        # fill forward any missing values
        errors.fillna(method='ffill')
        # merge the data back with the groups perdcentile buckets
        final_out = pd.merge(pd.DataFrame(group_vecs[col]), errors,
                             left_on=col, right_on=col, how='left')

        final_out.fillna(method='ffill', inplace=True)

        return final_out

    def run(self):
        # run the prediction function first to assign the errors to the dataframe
        self.predict()
        # create placeholder for outputs
        placeholder = []
        # create placeholder for all insights
        insights_df = pd.DataFrame()

        for col, groupby in product(self.cat_df.columns[~self.cat_df.columns.isin(['errors', 'predictedYSmooth',
                                                                                   self.ydepend])], self.groupbyvars):
            # check if we are a col that is the groupbyvar3
            if col != groupby:
                # print('Currently on col: {}\nGroupby: {}'.format(col, groupby))
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
                                                 col = col, vartype = vartype,
                                                 groupby = groupby)
                    # groupby the groupby variable on subset of columns and apply cont_slice_partial
                    errors = self.cat_df[col_indices].groupby(groupby).apply(cont_slice_partial)


                # json out
                print(vartype)
                json_out = to_json(errors, vartype = vartype)
                # append to placeholder
                placeholder.append(json_out)

            else:
                # use this as an opportunity to capture error metrics for the groupby variable
                # create a partial func by pre-filling in the parameters for create_insights
                insights = partial(create_insights, group_var = groupby,
                                   error_type='MSE')

                acc = self.cat_df.groupby(groupby).apply(insights)
                # drop the grouping indexing
                acc.reset_index(drop = True, inplace = True)
                # append to insights_df
                insights_df = insights_df.append(acc)

        # finally convert insights_df into json object
        insights_json = to_json(insights_df, vartype = 'Accuracy')
        # append to outputs
        placeholder.append(insights_json)
        # assign outputs to class
        self.outputs = placeholder





