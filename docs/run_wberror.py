from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from mdesc.utils import utils
from mdesc.utils import percentiles
from mdesc.eval import ErrorViz
import numpy as np

#====================
# wine quality dataset example
# featuredict - cat and continuous variables
wine = utils.create_wine_data(None)

wine.head()


# init randomforestregressor
modelObjc = RandomForestRegressor(random_state=2)

###
#
# Specify model parameters
#
###
ydepend = 'quality'
# create second categorical variable by binning
wine['volatile.acidity.bin'] = wine['volatile acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
# specify groupby variables
groupbyVars = ['Type'] #, 'volatile.acidity.bin']
# subset dataframe down
wine_sub = wine.copy(deep = True)

mod_df = pd.get_dummies(wine_sub.loc[:, wine_sub.columns != ydepend])




modelObjc.fit(mod_df,
              wine_sub.loc[:, ydepend])

keepfeaturelist = ['fixed acidity',
                   'Type',
                   'quality',
                   'volatile.acidity.bin',
                   'alcohol',
                   'sulphates']

WB = ErrorViz(modelobj=modelObjc,
              model_df=mod_df,
              ydepend=ydepend,
              cat_df=wine_sub,
              groupbyvars=['Type', 'alcohol'],
              keepfeaturelist=None,
              verbose=None,
              round_num=2,
              autoformat_types=True)

WB.run(output_type='html',
       output_path='REGRESSIONTEST2.html')