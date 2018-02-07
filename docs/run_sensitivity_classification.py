from whitebox.whitebox import WhiteBoxSensitivity
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

df = pd.read_csv('docs/datasets/winequality.csv')

# set up y var
# set up some params
ydepend = 'Type'

# convert categorical
df['AlcoholContent'] = pd.Categorical(df['AlcoholContent'])
df['quality'] = pd.Categorical(df['quality'])

df.select_dtypes(include = ['category'])
model_df = df.copy(deep = True)

# create dummies example using all categorical columns
dummies = pd.concat([pd.get_dummies(model_df.loc[:, col], prefix = col) for col in model_df.select_dtypes(include = ['category']).columns], axis = 1)
finaldf = pd.concat([model_df.select_dtypes(include = [np.number]), dummies], axis = 1)


clf = RandomForestClassifier()
# fit the model using the dummy dataframe
clf.fit(finaldf.loc[:, finaldf.columns != ydepend], df.loc[:, ydepend])


WB = WhiteBoxSensitivity(clf,
                   model_df=finaldf,
                   ydepend=ydepend,
                   cat_df=df,
                   featuredict=None,
                   groupbyvars=['AlcoholContent'],
                   aggregate_func=np.mean,
                   verbose=None,
                    std_num=2
                   )



WB.run()

WB.save('./output/WINEQUALITY_SENSITIVITY_CLASSIFICATION.html')