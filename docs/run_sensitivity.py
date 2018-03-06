# sensitivity plot creation and testing

from sklearn.ensemble import RandomForestRegressor
from mdesc.eval import SensitivityViz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import requests
import io

from mdesc.eval import ErrorViz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


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

col_indices = ['density', 'errors', 'predictedYSmooth', 'Type', 'diff']

WB._predict_synthetic('density',
                      'Type',
                      model_df,
                      col_indices,
                      vartype='Continuous')

model_df.columns

"""
old tests
    def test_handle_categorical_preds(self):
        """test modal value identified"""

        copydf = self.WB._model_df.copy(deep=True)

        col_indices = ['alcohol', 'errors', 'predictedYSmooth', 'Type', 'diff']

        self.WB._cat_df['errors'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])
        copydf['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['diff'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        modal, res = self.WB._predict_synthetic('alcohol',
                                                'Type',
                                                copydf,
                                                col_indices,
                                                vartype='Categorical')
        self.assertEqual(modal,
                         self.wine['alcohol'].mode().values[0],
                         """unexpected modal value on _handle_categorical_preds""")
    def test_handle_categorical_preds_df_output(self):
        """test dataframe output returned"""

        copydf = self.mod_df.copy(deep=True)

        col_indices = ['alcohol', 'errors', 'predictedYSmooth', 'Type', 'diff']

        self.WB._cat_df['errors'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])
        copydf['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['diff'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        modal, res = self.WB._predict_synthetic('alcohol',
                                                'Type',
                                                copydf,
                                                col_indices,
                                                vartype='Categorical')

        self.assertIsInstance(res,
                              pd.DataFrame,
                              """pd.DataFrame not returned in _predict_synthetic""")
                              
    def test_handle_continuous_incremental_val_output(self):
        """test incremental val output returned is correct"""

        copydf = self.WB._model_df.copy(deep=True)

        col_indices = ['density', 'errors', 'predictedYSmooth', 'Type', 'diff']

        self.WB._cat_df['errors'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])
        copydf['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['diff'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        incremental_val, res = self.WB._predict_synthetic('density',
                                                          'Type',
                                                          copydf,
                                                          col_indices,
                                                          vartype='Continuous')

        self.assertEqual(round(incremental_val, 4),
                         round(copydf['density'].std() * 0.5, 4),
                         """incremental_val is incorrect""")
                         
    def test_handle_continuous_sensitivity_output(self):
        """test sensitvity output is of type dataframe"""

        copydf = self.WB._model_df.copy(deep=True)

        col_indices = ['density', 'errors', 'predictedYSmooth', 'Type', 'diff']

        self.WB._cat_df['errors'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])
        copydf['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['diff'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        incremental_val, res = self.WB._predict_synthetic('density',
                                                          'Type',
                                                          copydf,
                                                          col_indices,
                                                          vartype='Continuous')

        self.assertIsInstance(res,
                              pd.DataFrame,
                              """sensitivity output not of type dataframe""")

"""