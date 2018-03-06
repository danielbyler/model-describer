from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from mdesc.utils import utils
from mdesc.utils import percentiles
from mdesc.eval import ErrorViz
import numpy as np
import requests
import io


def create_wine_data(cat_cols):
    """
    create UCI wine machine learning dataset
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality

    :param cat_cols: columns to convert to categories
    :return UCI wine machine learning dataset
    :rtype pd.DataFrame
    """

    if not cat_cols:
        cat_cols = ['alcohol', 'fixed acidity']

    red_raw = requests.get(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv').content
    red = pd.read_csv(io.StringIO(red_raw.decode('utf-8-sig')),
                      sep=';')
    red['Type'] = 'Red'

    white_raw = requests.get(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv').content
    white = pd.read_csv(io.StringIO(white_raw.decode('utf-8-sig')),
                        sep=';')
    white['Type'] = 'White'

    # read in wine quality dataset
    wine = pd.concat([white, red])

    # create category columns
    # create categories
    for cat in cat_cols:
        wine.loc[:, cat] = pd.cut(wine.loc[:, cat], bins=3, labels=['low', 'medium', 'high'])

    return wine

#====================
# wine quality dataset example
# featuredict - cat and continuous variables
wine = create_wine_data(None)

wine.head()


# init randomforestregressor
modelObjc = RandomForestRegressor(random_state=2)

###
#
# Specify model parameters
#
###
ydepend = 'quality'
# create second categorical variable by binning
wine['volatile.acidity.bin'] = wine['volatile acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
# specify groupby variables
groupbyVars = ['Type'] #, 'volatile.acidity.bin']
# subset dataframe down
wine_sub = wine.copy(deep = True)

mod_df = pd.get_dummies(wine_sub.loc[:, wine_sub.columns != ydepend])




modelObjc.fit(mod_df,
              wine_sub.loc[:, ydepend])

keepfeaturelist = ['fixed acidity',
                   'Type',
                   'quality',
                   'volatile.acidity.bin',
                   'alcohol',
                   'sulphates']


WB = ErrorViz(modelobj=modelObjc,
              model_df=mod_df,
              ydepend=ydepend,
              cat_df=wine_sub,
              groupbyvars=['Type', 'alcohol'],
              keepfeaturelist=None,
              verbose=None,
              round_num=2,
              autoformat_types=True)

WB.run(output_type='html',
       output_path='error_viz_regressions.html')

