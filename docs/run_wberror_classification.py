from whitebox.whitebox import WhiteBoxError
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

df = pd.read_csv('docs/datasets/winequality.csv')

# set up y var
# set up some params
ydepend = 'Type'

# convert categorical
df['AlcoholContent'] = pd.Categorical(df['AlcoholContent'])
model_df = df.copy(deep = True)

model_df['AlcoholContent'] = model_df['AlcoholContent'].cat.codes

x = model_df.loc[:, model_df.columns != ydepend]
y = model_df.loc[:, ydepend]

# build model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x, y)


WB = WhiteBoxError(clf,
                   model_df=model_df,
                   ydepend=ydepend,
                   cat_df=df,
                   featuredict=None,
                   groupbyvars=['AlcoholContent'],
                   aggregate_func=np.mean,
                   verbose=None
                   )

WB.run()

WB.save('./output/WINEQUALITY_CLASSIFICATION.html')