from whitebox.eval import WhiteBoxSensitivity
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

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
                      pd.get_dummies(df.select_dtypes(include=['O', 'category']))], axis=1)

model_df.head()

# build model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(model_df.loc[:, model_df.columns != ydepend],
        model_df.loc[:, ydepend])

df.dtypes

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



WB.raw_df.head()
WB.raw_df[WB.raw_df['fixed_bins'].notnull()]

WB.agg_df.head(100)

WB.save('./output/WINEQUALITY_SENSITIVITY_CLASSIFICATION.html')