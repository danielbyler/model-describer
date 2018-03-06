import pandas as pd
import numpy as np
import os
from mdesc.eval import ErrorViz
from sklearn.ensemble import RandomForestClassifier


os.getcwd()

# read in nls data
df = pd.read_csv('docs/nls.csv')

df['DEGREE'].unique()

# convert degree to has degree or doesn't have degree
# 1 has degree, 0 does not
df['DEGREE'] = df.loc[:, 'DEGREE'].apply(lambda x: 0 if x == 'None' else 1)

df.head()

# define dependent variable
ydepend = 'DEGREE'

# define keep feature list
keepfeaturelist = ['DEGREE', 'HS_GPA', 'AGE', 'FELON', 'PRNTS_INCOME_PERCENTILE',
                   'FMLY_MEM_HOSPITALIZED']


df2 = df.loc[df['YEAR'] == 2013, keepfeaturelist]

df2.shape

df2['YEAR'].unique()
df2['FELON'] = df2['FELON'].astype(str)
df2['FMLY_MEM_HOSPITALIZED'] = df2['FMLY_MEM_HOSPITALIZED'].astype(str)
df2['YEAR'] = df2['YEAR'].astype(str)

del df2['YEAR']

df2.fillna('nan', inplace=True)

df2.head()
df2['PRNTS_INCOME_PERCENTILE'].values[100]

model_df = pd.get_dummies(df2.loc[:, df2.columns != ydepend])


clf = RandomForestClassifier(max_depth=2, random_state=2)


clf.fit(model_df, df2.loc[:, ydepend])

EV = ErrorViz(clf,
              model_df=model_df,
              ydepend=ydepend,
              cat_df=df2,
              keepfeaturelist=keepfeaturelist,
              groupbyvars=['FELON'],
              aggregate_func=np.mean,
              verbose=None,
              autoformat_types=True
              )

EV.run(output_path='nls_error_classification.html',
       output_type='html')

EV.outputs