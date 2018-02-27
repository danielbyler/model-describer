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

WB = WhiteBoxError(modelobj=modelObjc,
                   model_df=mod_df,
                   ydepend=ydepend,
                   cat_df=wine_sub,
                   groupbyvars=['Type'],
                   keepfeaturelist=None,
                   verbose=None,
                   autoformat_types=True)


WB.run(output_type='html',
       output_path='REGRESSIONTEST2.html')

dir(WB)
WB.__doc__
help(WB)
import timeit
from timeit import Timer

t = Timer(lambda: WB.run(output_type='html',
       output_path='REGRESSIONTEST2.html'))

ti = t.timeit(number=5)

WB.groupbyvars

WB.run(output_type='html',
       output_path='REGRESSIONTEST2.html')

WB.raw_df.head()

WB.raw_df.head(100)
WB.agg_df.tail()

import re
import ast
string_out = str(WB.outputs)

for k, v in featuredict.items():
    string_out = re.sub("(?<='){}(?=')".format(k), "{}".format(v), string_out)

ast.literal_eval(string_out)
test
WB.debug_df[WB.debug_df['predictedYSmooth'].isnull()]

any(pd.isnull(WB.debug_df))

WB.debug_df[WB.debug_df.duplicated()].shape

WB.groupbyvars

WB._cat_df.dtypes


WB.debug_df[WB.debug_df.duplicated()]['col_name'].unique()

WB.debug_df.shape

WB._cat_df.columns

WB.outputs

wine_sub['Type'].unique()

subset = WB._cat_df.groupby('Type').get_group('Red')

density_subset = subset.loc[:, ['density', 'Type', 'predictedYSmooth', 'errors']]

# create density percentiles
density_percentiles = percentiles.create_percentile_vecs(density_subset.loc[:, 'density'])

list(density_percentiles)


density_subset['fixed_bins'] = np.digitize(density_subset.loc[:, 'density'],
                                           density_percentiles,
                                           right=True)

density_subset['fixed_bins'].nunique()

z = density_subset.groupby('fixed_bins')['density'].count().reset_index()
z[z['density'] == 1]['density'].max()


density_subset.head()
wine_sub['fixed_bins'] = np.digitize(wine_sub.loc[:, 'sulphates'],
            sorted(list(set(WB.Percentiles.percentile_vecs['sulphates']))),
            right=True)

WB.Percentiles.percentile_vecs['sulphates']



"""
 {'Data': [{'errNeg': 'null',
    'errPos': 0.20000000000000018,
    'groupByValue': 'Red',
    'groupByVarName': 'Type',
    'predictedYSmooth': 4.2,
    'sulphates': 0.33},
   {'errNeg': 'null',
    'errPos': 0.3500000000000001,
    'groupByValue': 'Red',
    'groupByVarName': 'Type',
    'predictedYSmooth': 4.85,
    'sulphates': 0.37},
   {'errNeg': 'null',
    'errPos': 'null',
    'groupByValue': 'null',
    'groupByVarName': 'Type',
    'predictedYSmooth': 'null',
    'sulphates': 'null'},
   {'errNeg': -0.09999999999999964,
    'errPos': 0.19999999999999973,
    'groupByValue': 'Red',
    'groupByVarName': 'Type',
    'predictedYSmooth': 5.05,
    'sulphates': 0.4},
"""


from scipy import stats
stats.mode([0, 1, 1, 0, 1]).mode[0]

ydepend, groupby, df = utils.create_synthetic(nrows=5000,
                                              ncols=10,
                                              ncat=2,
                                              max_levels=5,
                                              mod_type='regression')


final_df = pd.concat([pd.get_dummies(df.select_dtypes(include=['O'])),
                      df.select_dtypes(include=[np.number])], axis=1)

modelObjc.fit(final_df.loc[:, final_df.columns != ydepend],
              final_df.loc[:, ydepend])


print('hello')

WB = WhiteBoxError(modelobj=modelObjc,
                   model_df=final_df,
                   ydepend=ydepend,
                   cat_df=df,
                   groupbyvars=groupby,
                   featuredict=None,
                   verbose=None)

WB.run(output_type='html',
       output_path='REGRESSIONTESTCONTROLLED.html')


WB.debug_df[WB.debug_df['groupByValue'].isnull()]

WB.debug_df.head()
WB.outputs

WB.outputs
import warnings
warn = warnings.WarningMessage('test')




