from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from whitebox import utils
import numpy as np
from whitebox.wbox_error import WhiteBoxError

#====================
# wine quality dataset example
# featuredict - cat and continuous variables

# read in wine quality dataset
wine = pd.read_csv('./data/winequality.csv')
# init randomforestregressor
modelObjc = RandomForestRegressor()

###
#
# Specify model parameters
#
###
yDepend = 'quality'
# create second categorical variable by binning
wine['volatile.acidity.bin'] = wine['volatile.acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
# specify groupby variables
groupbyVars = ['Type', 'volatile.acidity.bin']
# subset dataframe down
wine_sub = wine.copy(deep = True)
# select all string columns so we can convert to pandas Categorical dtype
string_categories = wine_sub.select_dtypes(include = ['O'])
# iterate over string categories
for cat in string_categories:
    wine_sub[cat] = pd.Categorical(wine_sub[cat])

# create train dataset for fitting model
xTrainData = wine_sub.loc[:, wine_sub.columns != yDepend].copy(deep = True)
# convert all the categorical columns into their category codes
xTrainData = utils.convert_categorical_independent(xTrainData)
yTrainData = wine_sub.loc[:, yDepend]

modelObjc.fit(xTrainData, yTrainData)

# specify featuredict as a subset of columns we want to focus on
featuredict = {'fixed.acidity': 'FIXED ACIDITY',
               'Type': 'TYPE',
               'quality': 'SUPERQUALITY',
               'volatile.acidity.bin': 'VOLATILE ACIDITY BINS',
               'AlcoholContent': 'AC',
               'sulphates': 'SULPHATES'}


WB = WhiteBoxError(modelobj = modelObjc,
                   model_df = xTrainData,
                   ydepend= yDepend,
                   cat_df = wine_sub,
                   groupbyvars = groupbyVars,
                   featuredict = featuredict)

wine_sub.head()

cat = filter(lambda x: x['Type'] == 'Categorical', WB.outputs)
cont = filter(lambda x: x['Type'] == 'Continuous', WB.outputs)
cont
for val in WB.outputs:
    print(val['Type'])

WB.save(fpath = './output/wine_quality_test.html')

#=================================
# IRIS Dataset Example
#

from sklearn import datasets
ydepend = 'target'
groupbyvars = ['Type', 'Subtype']


iris_data = datasets.load_iris()
df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                            columns = ['sepall', 'sepalw', 'petall', 'petalw', 'target'])

df['Type'] = ['white'] * 75 + ['red'] * 75
df['Subtype'] = ['bin1'] * 50 + ['bin2'] * 50 + ['bin3'] * 50


# set up randomforestregressor
modelobj = RandomForestRegressor()

df['Type'] = pd.Categorical(df['Type'])
df['Subtype'] = pd.Categorical(df['Subtype'])

model_df = df.copy(deep = True)
model_df['Type'] = model_df['Type'].cat.codes
model_df['Subtype'] = model_df['Subtype'].cat.codes


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
WB.save(fpath = './output/IRIS.html')

acc = filter(lambda x: x['Type'] == 'Accuracy', WB.outputs)
acc
WB.outputs
WB.cat_df['errors']
WB.cat_df['predictedYSmooth']
WB.cat_df['target']
'''


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
'''