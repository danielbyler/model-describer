import pandas as pd
from utils.utils import to_json, getVectors, create_insights, createMLErrorHTML
import abc


class WhiteBoxBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def handle_variables(self, dataframe, predictions, diff,
                         group, group_name, groupbylist = [],
                         vartype = 'cat'):
        """handle data for categorical and continuous variables"""
        return



    def __repr__(self):
        """when printed, print class name and parameters"""
        class_name = self.__class__.__name__
        #TODO add parameters in print out
        return '{}'.format(class_name)

class WhiteBoxError(object):

    def __init__(self, trainData, dependVar, groupbylist, output_path = None):
        #super(WhiteBoxError).__init__()
        # assign data to class object
        self.trainData = trainData
        # assign groupby list
        self.groupbylist = groupbylist
        # output path
        self.output_path = output_path
        # datastring
        self.datastring = []
        # dependent variable
        self.dependVar = dependVar

    def handle_variables(self, predictions, diff,
                         vartype='cat'):
        # predictions are actual predictions
        # diff is the difference between predictions and actual values
        # group is the pandas group object
        # group name is the value of the specific value within a group we are examining i.e. Red wine
        # groupedby is the name of the variable or column itself i.e. Type of Wine
        assert vartype in ['cat', 'cont'], 'must be cat or cont variable'
        # create placeholder list
        #masterlist = []
        # create insights holder
        #insights = []
        # iterate over the columne
        for col in self.trainData.columns:
            # create placeholder dataframe
            tmpdf = pd.DataFrame()
            # create placeholder for insights df
            insightsdf = pd.DataFrame()
            # iterate over each var in the gropubylist
            for groupvar in self.groupbylist:
                # if vartype is categorical, add the col to the groupby list
                if vartype == 'cat':
                    groupvar = [groupvar, col]

                for name, group in self.trainData.groupby(groupvar):
                    # get the indices of predictions for this group
                    preds_group = predictions[group.index]
                    # of these predictions, separate out positive, negative, and zero preds
                    err_df = pd.DataFrame({'errPos': diff[group.index],
                                           'errNeg': diff[group.index]})
                    # if categorical variable, use the mean of the slice
                    if vartype == 'cat':
                        # convert neg vals in positive column to None
                        err_df['errPos'] = err_df['errPos'].apply(lambda x: x if x > 0 else None)
                        # convert positive vals in neg column to None
                        err_df['errNeg'] = err_df['errNeg'].apply(lambda x: x if x < 0 else None)
                        # compress dataframe to single observation by taking mean of each type of error
                        err_df = pd.DataFrame({'errPos': err_df['errPos'].mean(),
                                               'errNeg': err_df['errNeg'].mean()}, index=[0])
                        # assign values of column
                        err_df[col] = name[1] if isinstance(name,
                                                            list) else name  # dataframe.loc[group.index, col].values.tolist()
                        # assign predictions
                        err_df['predictedYSmooth'] = preds_group.mean()
                    else:
                        err_df[col] = self.trainData.loc[group.index, col].values.tolist()
                        # assign predictions
                        err_df['predictedYSmooth'] = preds_group
                        # convert neg vals in positive column to None
                        err_df['errPos'] = err_df['errPos'].apply(lambda x: x if x > 0 else float('NaN'))
                        # convert positive vals in neg column to None
                        err_df['errNeg'] = err_df['errNeg'].apply(lambda x: x if x < 0 else float('NaN'))
                        # convert to vectors
                        err_df = getVectors(err_df)
                        print(err_df)
                        # capture mse for additional insights
                        tmpinsights = create_insights(group, diff, name, groupvar,
                                                      error_type = 'MSE')
                        # append to insights df
                        # insightsdf = insightsdf.append(tmpinsights)
                        # append to datastring
                        # self.datastring.append(insightsdf)
                        insightsdf = insightsdf.append(tmpinsights)
                    # assign group by variable name
                    err_df['groupByVarName'] = groupvar[0] if isinstance(groupvar, list) else groupvar
                    # assign groupby value
                    err_df['groupByValue'] = name[0] if isinstance(name, list) else name
                    # replace nan with 'nan' for D3 purposes
                    err_df.fillna("null", inplace = True)

                    # return err_df
                    tmpdf = tmpdf.append(err_df)
            json = to_json(tmpdf)
            self.datastring.append(json)
            #masterlist.append(tmpdf)
            #insights.append(insightsdf)
            insights_json = to_json(insightsdf, vartype = 'Accuracy')
            self.datastring.append(insights_json)

        #return masterlist, insights


# load sample data
from sklearn import datasets
import numpy as np
iris_data = datasets.load_iris()
iris = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                            columns=iris_data['feature_names'] + ['target'])

from sklearn.ensemble import RandomForestRegressor
modelObjc = RandomForestRegressor()



iris.columns = ['sepall', 'sepalw', 'petall', 'petalw', 'target']

dependVar = 'target'
xTrainData = iris.loc[:, iris.columns != dependVar]
yTrainData = iris[dependVar]

vecs = getVectors(xTrainData)
vecs
WBError = WhiteBoxError(xTrainData, yTrainData, ['sepall'], output_path = None)

modelObjc.fit(xTrainData, yTrainData)

preds = modelObjc.predict(xTrainData)

diff = preds - yTrainData

WBError.handle_variables(preds, diff,
                         vartype = 'cont')

WBError.datastring[4]

output = createMLErrorHTML(str(WBError.datastring), dependVar)

with open('./output/test1.html', 'w') as outfile:
    outfile.write(output)