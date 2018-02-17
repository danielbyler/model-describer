from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from whitebox import utils
from whitebox.eval import WhiteBoxError
import requests
import io
import numpy as np



#====================
# wine quality dataset example
# featuredict - cat and continuous variables
wine = utils.create_wine_data(None)


# init randomforestregressor
modelObjc = RandomForestRegressor()

###
#
# Specify model parameters
#
###
yDepend = 'quality'
# create second categorical variable by binning
wine['volatile.acidity.bin'] = wine['volatile acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
# specify groupby variables
groupbyVars = ['Type'] #, 'volatile.acidity.bin']
# subset dataframe down
wine_sub = wine.copy(deep = True)

# create train dataset for fitting model
xTrainData = wine_sub.loc[:, wine_sub.columns != yDepend].copy(deep = True)
# convert all the categorical columns into their category codes
xTrainData = utils.convert_categorical_independent(xTrainData)
yTrainData = wine_sub.loc[:, yDepend]



modelObjc.fit(xTrainData, yTrainData)

from sklearn.utils.validation import check_is_fitted, check_consistent_length


check_consistent_length(xTrainData, yTrainData.reset_index().ix[0:100])

check_is_fitted(modelObjc, 'base_estimator_')

dir(modelObjc)
sklearn.utils.validation.check_is_fitted(estimator, attributes, msg=None, all_or_any=<built-in function all>)


# specify featuredict as a subset of columns we want to focus on
featuredict = {'fixed acidity': 'FIXED ACIDITY_test',
               'Type': 'TYPE_test',
               'quality': 'SUPERQUALITY_test',
               'volatile.acidity.bin': 'VOLATILE ACIDITY BINS_test',
               'alcohol': 'AC_test',
               'sulphates': 'SULPHATES_test'}



WB = WhiteBoxError(modelobj=modelObjc,
                   model_df=xTrainData,
                   ydepend=yDepend,
                   cat_df=wine_sub,
                   groupbyvars=['Type'],
                   featuredict=featuredict,
                   verbose=None)

WB.run()
WB.cat_df.columns
import os
os.getcwd()
WB.save(fpath='PERCENTILESTEST.html')


class Test(object):

    def __init__(self, x):

        self._x = x

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):

        if value > 50:
            raise ValueError("x cannot be over 50")

        else:
            self._x = value


class Celsius:
    def __init__(self, temperature = 0):
        self._temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    @property
    def temperature(self):
        print("Getting value")
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value < -273:
            raise ValueError("Temperature below -273 is not possible")
        print("Setting value")
        self._temperature = value

z = Test(49)



