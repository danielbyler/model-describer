# sensitivity plot creation and testing

from sklearn.ensemble import RandomForestRegressor
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
ydepend = 'free sulfur dioxide'


model_df = pd.get_dummies(df.loc[:, df.columns != ydepend])
model_df[ydepend] = df.loc[:, ydepend]

model_df.head()

# build model
clf = RandomForestRegressor()
clf.fit(model_df.loc[:, model_df.columns != ydepend],
        model_df.loc[:, ydepend])

df.columns

WB = WhiteBoxSensitivity(clf,
                   model_df=model_df,
                   ydepend=ydepend,
                   cat_df=df,
                   keepfeaturelist=None,
                   groupbyvars=['alcohol', 'Type'],
                   verbose=None,
                    std_num=2,
                    autoformat_types=True,
                   )

WB.run(output_type='html',
       output_path='SENSITIVITYTEST.html')


