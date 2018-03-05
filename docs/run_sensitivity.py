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

# build model
clf = RandomForestRegressor()
clf.fit(model_df,
        df.loc[:, ydepend])


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

df = pd.DataFrame({'col1': np.random.uniform(10000000, 20000000, 1000)})

df['col1(10000000s)'] = list(map(lambda p: round(p, 2), df['col1']/10000000))
