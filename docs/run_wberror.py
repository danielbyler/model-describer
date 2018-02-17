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
red_raw = requests.get(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv').content
red = pd.read_csv(io.StringIO(red_raw.decode('utf-8-sig')),
                  sep=';')
red['Type'] = 'Red'

white_raw = requests.get(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv').content
white = pd.read_csv(io.StringIO(white_raw.decode('utf-8-sig')),
                    sep=';')
white['Type'] = 'White'
# read in wine quality dataset
wine = pd.concat([white, red])


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


# specify featuredict as a subset of columns we want to focus on
featuredict = {'fixed acidity': 'FIXED ACIDITY',
               'Type': 'TYPE',
               'quality': 'SUPERQUALITY',
               'volatile.acidity.bin': 'VOLATILE ACIDITY BINS',
               'alcohol': 'AC',
               'sulphates': 'SULPHATES'}



WB = WhiteBoxError(modelobj=modelObjc,
                   model_df=xTrainData,
                   ydepend=yDepend,
                   cat_df=wine_sub,
                   groupbyvars=['Type'],
                   featuredict=featuredict,
                   verbose=None)


wine.columns.map(lambda x: x.strip())
WB.run()
import os
os.getcwd()
WB.save(fpath='PERCENTILESTEST.html')