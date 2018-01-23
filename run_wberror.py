from whitebox import WhiteBoxError
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from utils.utils import convert_categorical_independent, createMLErrorHTML, create_insights, getVectors
import numpy as np
from utils.utils import to_json
from functools import partial
from itertools import product

#====================
# wine quality dataset example


wine = pd.read_csv('./data/winequality.csv')

wine.columns.values.tolist()

modelObjc = RandomForestRegressor()
#=====================================
yDepend = 'quality'
wine['volatile.acidity.bin'] = wine['volatile.acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')

groupbyVars = ['Type', 'volatile.acidity.bin']

wine_sub = wine[['fixed.acidity', 'volatile.acidity.bin', 'citric.acid',
             'residual.sugar', 'Type', 'quality', 'AlcoholContent']].copy(deep = True)


string_categories = wine_sub.select_dtypes(include = ['O'])
string_categories
# iterate over string categories
for cat in string_categories:
    wine_sub[cat] = pd.Categorical(wine_sub[cat])


xTrainData = wine_sub.loc[:, wine_sub.columns != yDepend].copy(deep = True)
xTrainData = convert_categorical_independent(xTrainData)
yTrainData = wine_sub.loc[:, yDepend]

modelObjc.fit(xTrainData, yTrainData)

wine_sub.columns
featuredict = {'fixed.acidity': 'fa',
               'Type': 'Type',
               'quality': 'q',
               'volatile.acidity.bin': 'acid_bins'}

wine_sub[featuredict.keys()]

wine_sub.columns

WB = WhiteBoxError(modelobj = modelObjc,
                   model_df = xTrainData,
                   ydepend= yDepend,
                   cat_df = wine_sub,
                   groupbyvars = groupbyVars,
                   featuredict = featuredict)
WB.run()
WB.save(fpath = './output/test_jan25.html')


for out in WB.outputs:
    all_data_keys = out['Data'].keys()

featuredict

'''
len(set(featuredict.keys()).intersection(groupbyVars))

# see various groupby varialbe names in the final outputs
all_groups = []

for group in WB.outputs:
    for data_elem in group['Data']:
        if data_elem.has_key('fixed.acidity'):
            all_groups.append(data_elem['fixed.acidity'])


set(all_groups)


WB.outputs[1]
# filter outputs to examine Categorical output
cats = filter(lambda x: x if x['Type'] == 'Categorical' else None, WB.outputs)

# filter accuracy
acc = filter(lambda x: x if x['Type'] == 'Accuracy' else None, WB.outputs)
acc
# check categorical output
cats[0]

# secure final output from createMLErrorHTML
final_out = createMLErrorHTML(str(WB.outputs), yDepend)

with open('./output/test_jan23.html', 'w') as outfile:
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
ydepend = 'target'
groupbyvars = ['Type']


iris_data = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                            columns = ['sepall', 'sepalw', 'petall', 'petalw', 'target'])

df['Type'] = ['white'] * 75 + ['red'] * 75


# set up randomforestregressor
modelobj = RandomForestRegressor()

df['Type'] = pd.Categorical(df['Type'])

model_df = df.copy(deep = True)
model_df['Type'] = model_df['Type'].cat.codes

modelobj.fit(model_df.loc[:, model_df.columns != ydepend],
             model_df.loc[:, ydepend])

# test whether outputs are assigned to instance after run
WB = WhiteBoxError(modelobj = modelobj,
              model_df = model_df,
              ydepend = ydepend,
              groupbyvars = groupbyvars,
                   cat_df = df)

WB.featuredict
WB.run()

final_out = createMLErrorHTML(str(WB.outputs), ydepend)

with open('./output/IRIS.html', 'w') as outfile:
    outfile.write(final_out)





df_sub = df[df['Type'] == 'white']

col = 'sepalw'
group_vecs = getVectors(df_sub)

group_vecs.loc[:, col]

df_sub['fixed_bins'] = df_sub.loc[:, col]

df_sub['errors'] = np.random.rand(df_sub.shape[0], 1)
df_sub['predictedYSmooth'] = np.random.rand(df_sub.shape[0], 1)

trans_partial = partial(WhiteBoxError.transform_function,
                                col=col,
                                groupby = 'Type',
                                vartype='Continuous')

errors = df_sub.groupby('fixed_bins').apply(trans_partial)

errors

errors.reset_index(drop = True, inplace = True)

errors.dropna(how = 'all', axis = 0, inplace = True)

errors.rename(columns={'Type': 'groupByValue'}, inplace=True)

errors['groupByVarName'] = 'Type'
errors['highlight'] = 'N'

final_out = pd.merge(pd.DataFrame(group_vecs[col]), errors,
                             left_on=col, right_on=col, how='left')

final_out

set(group_vecs.loc[:, col].values.tolist()).intersection(set(df_sub.loc[:, col]))

df['predictedYSmooth'] = modelobj.predict(model_df.loc[:, model_df.columns != 'target'])

xw
'''