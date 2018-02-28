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
                   autoformat_types=True,
                   round_num=4)

WB.Percentiles.group_percentiles_out
WB.Percentiles.percentiles

df = pd.DataFrame({"col1": np.random.rand(100),
                   'col2': np.random.rand(100)})

df['val2'] = 0.000000000000000000000000000000000000000000000005

df.round(2)
df.head()
df['test'] = 'a'
df.round(2)

WB.run(output_type='html',
       output_path='REGRESSIONTEST2.html')

WB.outputs

WB.agg_df.tail(100)