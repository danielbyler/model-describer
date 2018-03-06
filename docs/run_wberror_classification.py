from mdesc.eval import ErrorViz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from mdesc.utils.utils import create_wine_data

df = create_wine_data(None)

# set up y var
# set up some params
ydepend = 'quality'

# turn it into a binary classification problem
df.loc[:, ydepend] = df.loc[:, ydepend].apply(lambda x: 0 if x < 5 else 1)

# convert categorical
model_df = pd.get_dummies(df.loc[:, df.columns != ydepend])

# build model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(model_df,
        df.loc[:, ydepend])

from sklearn.utils.validation import check_is_fitted

EV = ErrorViz(clf,
              model_df=model_df,
              ydepend=ydepend,
              cat_df=df,
              keepfeaturelist=None,
              groupbyvars=['alcohol'],
              aggregate_func=np.mean,
              verbose=None,
              autoformat_types=True
              )

EV.run(output_path='error_viz_classification.html',
       output_type='html')

EV.model_type


store = []
for out in EV.outputs:
    if out['Type'] == 'Accuracy':
        store.append(out)

store
EV.model_type