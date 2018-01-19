import pandas as pd
from utils.utils import to_json, getVectors, create_insights, createMLErrorHTML, convert_categorical_independent
import abc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from itertools import product
from sklearn.utils.validation import check_is_fitted
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
            self.cat_df = cat_df

        else:
            if not isinstance(cat_df, type(None)):
                warnings.warn('cat_df is not None and not a pd.core.frame.DataFrame. Default becomes model_df'\
                              'and may not be intended user behavior', UserWarning)

            self.cat_df = model_df

        assert isinstance(ydepend, str), 'ydepend not string, dependent variable must be single column name'

        self.modelobj = modelobj
        self.model_df = model_df
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

    def transform_function(self, group):
        """
        transform slice of data by separating our pos/neg errors, aggregating up to the mean of the slice
        and returning transformed dataset
        :param group: slice of data being operated on
        :return: compressed data representation in dataframe object
        """
        assert 'errors' in group.columns, 'errors needs to be present in dataframe slice'
        # split out positive vs negative errors
        errors = group['errors']
        # create separate columns for pos or neg errors
        errors = pd.concat([errors[errors > 0], errors[errors < 0]], axis=1)
        # rename error columns
        errors.columns = ['errPos', 'errNeg']
        # merge back with original data
        toreturn = pd.concat([group.loc[:, group.columns != 'errors'], errors], axis=1)
        # return the mean
        return toreturn.mean()

    def run(self):
        # run the prediction function first to assign the errors to the dataframe
        self.predict()
        # get the 100 percentiles of the data
        vecs = getVectors(self.cat_df)

        # create placeholder for outputs
        placeholder = []

        for col, groupby in product(self.cat_df.columns[~self.cat_df.columns.isin(['errors', 'predictedYSmooth',
                                                                                   self.ydepend])], self.groupbyvars):
            # check if we are a col that is the groupbyvar3
            if col != groupby:
                print(col)
                # subset col indices
                col_indices = [col, 'errors', 'predictedYSmooth', groupby]
                # check if categorical
                if isinstance(self.cat_df.loc[:, col].dtype, pd.types.dtypes.CategoricalDtype):
                    # slice over the groupby variable and the categories within the current column
                    errors = self.cat_df[col_indices].groupby([groupby, col]).apply(self.transform_function)
                    # append to all errors
                    #placeholder.append(errors)
                else:
                    # create col bin name
                    col_bin_name = '{}_bins'.format(col)
                    # expand col indices to include bins
                    col_indices2 = col_indices[:]
                    # append the new column bins
                    col_indices2.append(col_bin_name)
                    # create bins for those percentiles across all data
                    self.cat_df[col_bin_name] = np.digitize(self.cat_df.loc[:, col],
                                                        sorted(list(set(vecs.loc[:, col]))), right=True)
                    # iterate over the grouping variable and the bins to capture the mean positive and negative errors within this slice
                    errors = self.cat_df[col_indices2].groupby([groupby, col_bin_name]).apply(self.transform_function)
                    # remove col_bin_name
                    del errors[col_bin_name]
                    del self.cat_df[col_bin_name]
                #errors = errors.reset_index(drop = True).rename(columns={groupby: 'groupByValue'})
                errors.reset_index(drop = True, inplace = True)
                #errors.rename(columns = {groupby = 'groupByValue'}, inplace = True)
                errors.rename(columns = {groupby : 'groupByValue'}, inplace = True)
                errors['groupByVarName'] = groupby
                # append to placeholder
                placeholder.append(errors)

            else:
                # use this as an opportunity to capture error metrics for the groupby variable
                # create a partial func by pre-filling in the parameters for create_insights
                insights = partial(create_insights, group_var = groupby,
                                   error_type='MSE')

                acc = self.cat_df.groupby(groupby).apply(insights)
                # drop the grouping indexing
                acc.reset_index(drop = True, inplace = True)
                # append placeholder
                print(acc)
                placeholder.append(acc)

        # assign outputs to class
        self.outputs = placeholder
'''
from sklearn import datasets
iris_data = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                            columns = ['sepall', 'sepalw', 'petall', 'petalw', 'target'])

# set up randomforestregressor
modelobj = RandomForestRegressor()

modelobj.fit(df.loc[:, df.columns != 'target'],
            df['target'])



# test whether outputs are assigned to instance after run
WB = WhiteBoxError(modelobj = modelobj,
              model_df = df,
              ydepend = 'target',
              groupbyvars = ['sepalw'])

WB.run()

filter(lambda x: 'MSE' in x.columns, WB.outputs)

WB.outputs
df.loc[:, df.columns != 'target']

modelobj.n_features_

modelobj.predict(df.loc[:, df.columns != 'target'])

WB.run()

df.columns



wine = pd.read_csv('./data/winequality.csv')

modelObjc = RandomForestRegressor()
#=====================================
yDepend = 'fixed.acidity'
groupbyVars = ['Type']

wine_sub = wine[['fixed.acidity', 'volatile.acidity', 'citric.acid',
             'residual.sugar', 'Type', 'quality', 'AlcoholContent']].copy(deep = True)

string_categories = wine_sub.select_dtypes(include = ['O'])
# iterate over string categories
for cat in string_categories:
    wine_sub[cat] = wine_sub[cat].astype('category')

# create dummies example using all categorical columns
dummies = pd.concat([pd.get_dummies(wine_sub.loc[:, col], prefix = col) for col in wine_sub.select_dtypes(include = ['category']).columns], axis = 1)
finaldf = pd.concat([wine_sub.select_dtypes(include = [np.number]), dummies], axis = 1)


xTrainData = wine_sub.loc[:, wine_sub.columns != yDepend].copy(deep = True)
xTrainData = convert_categorical_independent(xTrainData)
yTrainData = wine_sub[yDepend].copy(deep = True)

modelObjc.fit(xTrainData, yTrainData)

modelObjc.n_features_
xTrainData.columns

wine_sub.shape
yDepend
WB = WhiteBoxError(modelobj = modelObjc,
                   model_df = xTrainData,
                   ydepend= yDepend,
                   cat_df = wine_sub,
                   groupbyvars = ['Type'])
WB.run()

type(WB.outputs)
'''