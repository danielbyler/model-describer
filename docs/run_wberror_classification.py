from whitebox.eval import WhiteBoxError
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from whitebox.utils import create_wine_data

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

x = model_df.loc[:, model_df.columns != ydepend]
y = model_df.loc[:, ydepend]

# build model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x, y)

from sklearn.utils.validation import check_is_fitted
check_is_fitted(clf, 'base_estimator_')

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