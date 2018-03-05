from whitebox.eval import WhiteBoxSensitivity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from whitebox.utils.utils import create_wine_data

df = create_wine_data(None)

# set up y var
# set up some params
ydepend = 'quality'

# turn it into a binary classification problem
df.loc[:, ydepend] = df.loc[:, ydepend].apply(lambda x: 0 if x < 5 else 1)

# convert categorical
model_df = pd.get_dummies(df.loc[:, df.columns != ydepend])

model_df.head()

# build model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = LogisticRegression()
clf.fit(model_df,
        df.loc[:, ydepend])

WB = WhiteBoxSensitivity(clf,
                   model_df=model_df,
                   ydepend=ydepend,
                   cat_df=df,
                   keepfeaturelist=None,
                   groupbyvars=['alcohol', 'Type'],
                   aggregate_func=np.mean,
                   verbose=None,
                    std_num=2,
                    autoformat_types=True,
                   )



WB.run(output_type='html',
       output_path='WINEQUALITY_SENSITIVITY_CLASSIFICATION.html')

rdf = WB.raw_df

rdf[(rdf['groupByVar'] == 'Type') & (rdf['col_name'] == 'fixed acidity')]