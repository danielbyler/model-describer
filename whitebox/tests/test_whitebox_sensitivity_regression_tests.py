#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from whitebox.utils import utils as wb_utils
    from whitebox.eval import WhiteBoxSensitivity

except ImportError:
    import utils as wb_utils
    from eval import WhiteBoxSensitivity


class TestWBBaseMethods(unittest.TestCase):

    def setUp(self):
        # create wine dataset
        wine = pd.read_csv('testdata/wine.csv')

        # init randomforestregressor
        modelObjc = RandomForestRegressor()

        ydepend = 'quality'
        # create second categorical variable by binning
        wine['volatile.acidity.bin'] = wine['volatile acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
        # subset dataframe down
        wine_sub = wine.copy(deep=True)

        mod_df = pd.get_dummies(wine_sub.loc[:, wine_sub.columns != ydepend])

        modelObjc.fit(mod_df,
                      wine_sub.loc[:, ydepend])

        keepfeaturelist = ['fixed acidity',
                           'Type',
                           'quality',
                           'volatile.acidity.bin',
                           'alcohol',
                           'density',
                           'sulphates']

        wine_sub['alcohol'] = wine_sub['alcohol'].astype('object')

        self.WB = WhiteBoxSensitivity(modelobj=modelObjc,
                           model_df=mod_df,
                           ydepend=ydepend,
                           cat_df=wine_sub,
                           groupbyvars=['Type'],
                           keepfeaturelist=keepfeaturelist,
                           verbose=None,
                           autoformat_types=True)

        self.wine = wine

    def test_transform_function_predictedYSmooth(self):
        """test predictedYSmooth present after _transform_function called"""

        group = self.wine.groupby('Type').get_group('White')

        group['diff'] = np.random.uniform(-1, 1, group.shape[0])

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])

        col = 'density'

        res = self.WB._transform_function(group,
                                          col=col,
                                          vartype='Continuous',
                                          groupby_var='Type')

        self.assertIn('predictedYSmooth',
                      res.columns.tolist(),
                      """predictedYSmooth not present in output df after
                      _transform_function called""")

    def test_transform_function_predictedYSmooth_val(self):
        """test median predictedYSmooth is returned after transform_function called"""

        group = self.wine.groupby('Type').get_group('White')

        group['diff'] = np.random.uniform(-1, 1, group.shape[0])

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])

        correct = np.nanmedian(group['diff'])

        col = 'density'

        res = self.WB._transform_function(group,
                                          col=col,
                                          vartype='Continuous',
                                          groupby_var='Type')

        self.assertEqual(res['predictedYSmooth'].values.tolist()[0],
                         correct,
                         """unexpected value for predictedYSmooth - should be median default""")

    def test_handle_categorical_preds(self):
        """test modal value identified"""

        copydf = self.WB._model_df.copy(deep=True)

        col_indices = ['alcohol', 'errors', 'predictedYSmooth', 'Type', 'diff']

        self.WB._cat_df['errors'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])
        copydf['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['diff'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        modal, res = self.WB._handle_categorical_preds('alcohol',
                                                'Type',
                                                copydf,
                                                col_indices)
        self.assertEqual(modal,
                         self.wine['alcohol'].mode().values[0],
                         """unexpected modal value on _handle_categorical_preds""")

    def test_handle_categorical_preds_df_output(self):
        """test dataframe output returned"""

        copydf = self.WB._model_df.copy(deep=True)

        col_indices = ['alcohol', 'errors', 'predictedYSmooth', 'Type', 'diff']

        self.WB._cat_df['errors'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])
        copydf['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['diff'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        modal, res = self.WB._handle_categorical_preds('alcohol',
                                                'Type',
                                                copydf,
                                                col_indices)
        self.assertIsInstance(res,
                              pd.DataFrame,
                              """pd.DataFrame not returned in _handle_categorical_preds""")

    def test_handle_continuous_incremental_val_output(self):
        """test incremental val output returned is correct"""

        copydf = self.WB._model_df.copy(deep=True)

        col_indices = ['density', 'errors', 'predictedYSmooth', 'Type', 'diff']

        self.WB._cat_df['errors'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])
        copydf['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        self.WB._cat_df['diff'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        incremental_val, res = self.WB._handle_continuous_preds('density',
                                                'Type',
                                                copydf,
                                                col_indices)
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

        incremental_val, res = self.WB._handle_continuous_preds('density',
                                                'Type',
                                                copydf,
                                                col_indices)

        self.assertIsInstance(res,
                              pd.DataFrame,
                              """sensitivity output not of type dataframe""")

    def test_var_check_output(self):
        """test return of json like object after var_check run"""

        self.WB._cat_df['errors'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])
        self.WB._cat_df['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        out = self.WB._var_check('fixed acidity',
                      'Type')

        self.assertIsInstance(out,
                              dict,
                              """var_check didn't return json like object after run""")




