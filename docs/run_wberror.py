from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from whitebox.utils import utils
from whitebox.utils import percentiles
from whitebox.eval import WhiteBoxError
import numpy as np

#====================
# wine quality dataset example
# featuredict - cat and continuous variables
wine = utils.create_wine_data(None)

wine.head()


# init randomforestregressor
modelObjc = RandomForestRegressor()

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

mod_df = pd.concat([pd.get_dummies(wine_sub.loc[:, wine_sub.columns != ydepend].select_dtypes(include=['category', 'O'])),
           wine_sub.select_dtypes(include=[np.number])], axis=1)


modelObjc.fit(mod_df.loc[:, mod_df.columns != ydepend],
              mod_df.loc[:, ydepend])

keepfeaturelist = ['fixed acidity',
                   'Type',
                   'quality',
                   'volatile.acidity.bin',
                   'alcohol',
                   'sulphates']

wine_sub['alcohol'] = wine_sub['alcohol'].astype('object')

wine_sub.head()

WB = WhiteBoxError(modelobj=modelObjc,
                   model_df=mod_df,
                   ydepend=ydepend,
                   cat_df=wine_sub,
                   groupbyvars=['Type', 'alcohol'],
                   keepfeaturelist=None,
                   verbose=None,
                   autoformat_types=True)

from timeit import Timer

T = Timer(lambda: WB.run(output_type='html',
                         output_path='REGRESSIONTEST2.html'))

T.timeit(number=1)

WB.run(output_type='html',
       output_path='REGRESSIONTEST2.html')