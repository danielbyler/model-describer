from whitebox import WhiteBoxError
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from utils.utils import convert_categorical_independent, createMLErrorHTML, create_insights, getVectors
import numpy as np
from utils.utils import to_json
from functools import partial

#====================
# wine quality dataset example

wine = pd.read_csv('./data/winequality.csv')

modelObjc = RandomForestRegressor()
#=====================================
yDepend = 'fixed.acidity'
groupbyVars = ['Type']

wine_sub = wine[['fixed.acidity', 'volatile.acidity', 'citric.acid',
             'residual.sugar', 'Type', 'quality', 'AlcoholContent']].copy(deep = True)


string_categories = wine_sub.select_dtypes(include = ['O'])
string_categories
# iterate over string categories
for cat in string_categories:
    wine_sub[cat] = pd.Categorical(wine_sub[cat])

wine_sub['errors'] = np.random.rand(wine_sub.shape[0], 1)


xTrainData = wine_sub.loc[:, wine_sub.columns != yDepend].copy(deep = True)
xTrainData = convert_categorical_independent(xTrainData)
yTrainData = wine_sub.loc[:, yDepend]

wine_sub.select_dtypes(include = ['category'])

modelObjc.fit(xTrainData, yTrainData)

WB = WhiteBoxError(modelobj = modelObjc,
                   model_df = xTrainData,
                   ydepend= yDepend,
                   cat_df = wine_sub,
                   groupbyvars = ['Type'])
WB.run()

WB.outputs[0]
# filter outputs to examine Categorical output
cats = filter(lambda x: x if x['Type'] == 'Categorical' else None, WB.outputs)
# check categorical output
cats[0]

# secure final output from createMLErrorHTML
final_out = createMLErrorHTML(str(WB.outputs), yDepend)


with open('./output/test_jan22.html', 'w') as outfile:
    outfile.write(final_out)


wine_sub.describe()

#--------------------
# test framework
import random
bins = ['bin_{}'.format(bin) for bin in range(0, 100)]

wine_sub['test_bins'] = [random.sample(bins, 1)[0] for _ in xrange(wine_sub.shape[0])]

wine_sub.head(1)
res = wine_sub.groupby(['Type', 'test_bins']).apply(WB.transform_function) #.get_group(('White','Low')))
res
res.reset_index()



#=====================
# sensitivity

# create dummies example using all categorical columns
dummies = pd.concat([pd.get_dummies(wine_sub.loc[:, col], prefix = col) for col in wine_sub.select_dtypes(include = ['category']).columns], axis = 1)
finaldf = pd.concat([wine_sub.select_dtypes(include = [np.number]), dummies], axis = 1)



ydepend = 'residual.sugar'
groupbyvars = ['Type']
yTrainData = finaldf.loc[:, ydepend]
xTrainData = finaldf.loc[:, finaldf.columns != ydepend]

modelObjc.fit(xTrainData, yTrainData)

og_preds = modelObjc.predict(xTrainData)

copy_df = xTrainData.copy(deep = True)
# continuous case
# punch data up 1 standard deviation
copy_df['fixed.acidity'] = copy_df['fixed.acidity'] + copy_df['fixed.acidity'].std()

std_preds = modelObjc.predict(copy_df)

diff = std_preds - og_preds

diff[100:500]


#str(WB.outputs).replace("'", '"').replace('"null"', 'null')

final_out = createMLErrorHTML(str(WB.outputs), yDepend)


with open('./output/test.html', 'w') as outfile:
    outfile.write(final_out)

#=================================
# IRIS Dataset Example
#

from sklearn import datasets
iris_data = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                            columns = ['sepall', 'sepalw', 'petall', 'petalw', 'target'])

df['Type'] = ['white'] * 75 + ['red'] * 75


# set up randomforestregressor
modelobj = RandomForestRegressor()

df['Type'] = pd.Categorical(df['Type'])

model_df = df.copy(deep = True)
model_df['Type'] = model_df['Type'].cat.codes

modelobj.fit(model_df.loc[:, model_df.columns != 'target'],
            model_df['target'])


sub = df.loc[df['Type'] == 'white']
sub['errors'] = 0


# test whether outputs are assigned to instance after run
WB = WhiteBoxError(modelobj = modelobj,
              model_df = model_df,
              ydepend = 'target',
              groupbyvars = ['Type'],
                   cat_df = df)
WB.run()

WB.outputs[2]



json_out = to_json(WB.outputs[0])
json_out
type(WB.outputs)

final_out = createMLErrorHTML(str(WB.outputs), yDepend)

import re

start = re.search("type", final_out).start()
end = re.search('type', final_out).end()

final_out[start:end+500]

WB.outputs[0]

final_out[0:100]

with open('./output/test.html', 'w') as outfile:
    outfile.write(final_out)
