from whitebox.eval import WhiteBoxError
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from whitebox.utils.utils import create_wine_data

df = create_wine_data(None)

# set up y var
# set up some params
ydepend = 'quality'

# turn it into a binary classification problem
df.loc[:, ydepend] = df.loc[:, ydepend].apply(lambda x: 0 if x < 5 else 1)

# convert categorical
model_df = pd.concat([df.select_dtypes(include=[np.number]),
                      pd.get_dummies(df.select_dtypes(include=['O', 'category']), prefix='col')], axis=1)

model_df.head()

# build model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(model_df.loc[:, model_df.columns != ydepend],
        model_df.loc[:, ydepend])

from sklearn.utils.validation import check_is_fitted

WB = WhiteBoxError(clf,
                   model_df=model_df,
                   ydepend=ydepend,
                   cat_df=df,
                   keepfeaturelist=None,
                   groupbyvars=['alcohol'],
                   aggregate_func=np.mean,
                   verbose=None,
                   autoformat_types=True
                   )
df['alcohol'].unique()

WB.run(output_path='CLASSIFICATIONTEST.html',
       output_type='html')

WB.agg_df[WB.agg_df['predictedYSmooth'].isnull()]
WB.agg_df[WB.agg_df.duplicated()].shape

WB.agg_df.shape

WB.save("CLASSIFICATIONTEST.html")

df = pd.DataFrame({'col1': np.random.rand(100),
                   'col2': np.random.rand(100)})

df.loc[50, 'col1'] = 0.5

zero_mask = df['col1'] != 0.5

df.loc[zero_mask]

z1 = pd.concat([df[df['col1'] > 0.5]['col1'], df[df['col1'] < 0.5]['col1']], axis=1,
               verify_integrity=True)

z1.ix[51]



z1.shape
df.shape

pd.concat([z1, df.loc[zero_mask, df.columns != 'col1']], axis=1)


model_df.columns.isin(['quality'])

model_df.loc[:, model_df.columns != 'quality'].columns

WB.ydepend

WB.model_df.loc[:, ~WB.model_df.columns.isin(list(ydepend))]

WB.model_df.ix[:, WB.model_df.columns.isin(list(ydepend))]

WB.model_df.loc[:, 'quality']

WB.save('./output/WINEQUALITY_CLASSIFICATION.html')