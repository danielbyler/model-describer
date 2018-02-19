# sensitivity plot creation and testing

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from whitebox import utils
import numpy as np
import io
import requests
from whitebox.eval import WhiteBoxSensitivity


#====================
# wine quality dataset example
# featuredict - cat and continuous variables

# featuredict - cat and continuous variables
red_raw = requests.get(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv').content
red = pd.read_csv(io.StringIO(red_raw.decode('utf-8')),
                  sep=';')
red['Type'] = 'Red'

white_raw = requests.get(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv').content
white = pd.read_csv(io.StringIO(white_raw.decode('utf-8')),
                    sep=';')
white['Type'] = 'White'
# read in wine quality dataset
wine = pd.concat([white, red])

wine['alcohol'] = pd.cut(wine['alcohol'], bins=3, labels=['high', 'medium', 'low'])
# init randomforestregressor
modelObjc = RandomForestRegressor()

###
#
# Specify model parameters
#

yDepend = 'quality'
# create second categorical variable by binning
wine['volatile.acidity.bin'] = wine['volatile acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
# specify groupby variables
groupbyVars = ['Type', 'volatile.acidity.bin']
# subset dataframe down
wine_sub = wine.copy(deep = True)

# specify featuredict as a subset of columns we want to focus on
featuredict = {'fixed acidity': 'FIXED ACIDITY',
               'Type': 'TYPE',
               'quality': 'SUPERQUALITY',
               'volatile.acidity.bin': 'VOLATILE ACIDITY BINS',
               'alcohol': 'AC',
               'sulphates': 'SULPHATES'}


# create dummies example using all categorical columns
dummies = pd.concat([pd.get_dummies(wine_sub.loc[:, col], prefix = col) for col in wine_sub.select_dtypes(include = ['O', 'category']).columns], axis = 1)
finaldf = pd.concat([wine_sub.select_dtypes(include = [np.number]), dummies], axis = 1)

# fit the model using the dummy dataframe
modelObjc.fit(finaldf.loc[:, finaldf.columns != yDepend], finaldf.loc[:, yDepend])


wine_sub['alcohol'] = wine_sub['alcohol'].astype(str)

import warnings

type(warnings.warn('test'))

# instantiate whitebox sensitivity
WB = WhiteBoxSensitivity(modelobj=modelObjc,
                   model_df=finaldf,
                   ydepend=yDepend,
                   cat_df=wine_sub,
                   groupbyvars=['volatile.acidity.bin'], #TODO case where groupbyvars not in modeldf
                   featuredict=None,
                    autoformat=True)


WB.run()
# save the final outputs to disk
WB.save(fpath = 'SENSITIVITYPERCENTILES.html')


group = wine_sub.groupby('Type').get_group('Red')
group.index


