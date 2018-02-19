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




ydepend, groupby, df = utils.create_synthetic(nrows=5000,
                                              ncols=10,
                                              ncat=2,
                                              max_levels=5,
                                              mod_type='regression')


final_df = pd.concat([pd.get_dummies(df.select_dtypes(include=['O'])),
                      df.select_dtypes(include=[np.number])], axis=1)

modelObjc.fit(final_df.loc[:, final_df.columns != ydepend],
              final_df.loc[:, ydepend])

WB = WhiteBoxError(modelobj=modelObjc,
                   model_df=final_df,
                   ydepend=ydepend,
                   cat_df=df,
                   groupbyvars=groupby,
                   featuredict=None,
                   verbose=None)

df.groupby('col1').get_group('level_1')

WB.run()

WB.save('REGRESSIONTEST.html')

ydepend, groupby, df = utils.create_synthetic(nrows=10000,
                                              ncols=10,
                                              ncat=3,
                                              max_levels=10,
                                              mod_type='classification')



