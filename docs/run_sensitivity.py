# sensitivity plot creation and testing

from sklearn.ensemble import RandomForestRegressor
from mdesc.eval import SensitivityViz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from mdesc.eval import ErrorViz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from mdesc.utils.utils import create_wine_data

df = create_wine_data(None)

# set up y var
# set up some params
ydepend = 'free sulfur dioxide'


model_df = pd.get_dummies(df.loc[:, df.columns != ydepend])

# build model
clf = RandomForestRegressor()
clf.fit(model_df,
        df.loc[:, ydepend])


WB = SensitivityViz(clf,
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
       output_path='sensitivityviz_regression.html')
