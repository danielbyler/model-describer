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
    from mdesc.utils import utils as wb_utils
    from mdesc.eval import ErrorViz

except ImportError:
    import utils as wb_utils
    from base import WhiteBoxBase
    from eval import WhiteBoxError


GLOBAL_ROUND = 2

class TestWBBaseMethods(unittest.TestCase):

    def setUp(self):
        # create wine dataset
        try:
            wine = pd.read_csv('testdata/wine.csv')
        except FileNotFoundError:
            wine = pd.read_csv('mdesc/tests/testdata/wine.csv')

        # init randomforestregressor
        modelObjc = RandomForestRegressor()

        ydepend = 'quality'
        # create second categorical variable by binning
        wine['volatile.acidity.bin'] = wine['volatile acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
        # subset dataframe down
        wine_sub = wine.copy(deep=True)

        mod_df = pd.concat(
            [pd.get_dummies(wine_sub.loc[:, wine_sub.columns != ydepend].select_dtypes(include=['category', 'O'])),
             wine_sub.select_dtypes(include=[np.number])], axis=1)

        modelObjc.fit(mod_df.loc[:, mod_df.columns != ydepend],
                      mod_df.loc[:, ydepend])

        keepfeaturelist = ['fixed acidity',
                           'Type',
                           'quality',
                           'volatile.acidity.bin',
                           'alcohol',
                           'sulphates']

        wine_sub['alcohol'] = wine_sub['alcohol'].astype('object')

        self.WB = ErrorViz(modelobj=modelObjc,
                           model_df=mod_df,
                           ydepend=ydepend,
                           cat_df=wine_sub,
                           groupbyvars=['Type'],
                           keepfeaturelist=keepfeaturelist,
                           verbose=None,
                           autoformat_types=True,
                           round_num=GLOBAL_ROUND)

        self.wine = wine

    def test_transform_function_predictedYSmooth(self):
        """test predictedYSmooth present after _transform_function called"""

        group = self.wine.groupby('Type').get_group('White')

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])

        group['predictedYSmooth'] = np.random.rand(group.shape[0])

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

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])

        group['predictedYSmooth'] = np.random.rand(group.shape[0])

        correct = np.median(group['predictedYSmooth'])

        col = 'density'

        res = self.WB._transform_function(group,
                                          col=col,
                                          vartype='Continuous',
                                          groupby_var='Type')

        self.assertEqual(res['predictedYSmooth'].values.tolist()[0],
                         round(correct, GLOBAL_ROUND),
                         """unexpected value for predictedYSmooth - should be median default""")

    def test_transform_function_errPos_val(self):
        """test median errPos present after _transform_function called"""

        group = self.wine.groupby('Type').get_group('White')

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])

        group['predictedYSmooth'] = np.random.rand(group.shape[0])

        correct = np.median(group.loc[group['errors'] >= 0, 'errors'])

        col = 'density'

        res = self.WB._transform_function(group,
                                          col=col,
                                          vartype='Continuous',
                                          groupby_var='Type')

        self.assertEqual(res['errPos'].values.tolist()[0],
                         round(correct, GLOBAL_ROUND),
                         """unexpected value for errPos - should be median default""")

    def test_transform_function_errNeg_val(self):
        """test median errNeg present after _transform_function called"""

        group = self.wine.groupby('Type').get_group('White')

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])

        group['predictedYSmooth'] = np.random.rand(group.shape[0])

        correct = np.median(group.loc[group['errors'] <= 0, 'errors'])

        col = 'density'

        res = self.WB._transform_function(group,
                                          col=col,
                                          vartype='Continuous',
                                          groupby_var='Type')

        self.assertEqual(res['errNeg'].values.tolist()[0],
                         round(correct, GLOBAL_ROUND),
                         """unexpected value for errNeg - should be median default""")

    def test_transform_function_errPos_classification(self):
        """test correct errPos output from transform_func in classification context"""
        self.WB.model_type = 'classification'
        # get copy of group
        group = self.wine.groupby('Type').get_group('White').copy(deep=True)
        # create random errors
        group.loc[:, 'errors'] = np.random.uniform(-1, 1, group.shape[0])
        # get aggregate value of errors
        agg_errors = np.nanmedian(group['errors'])
        # subtract real errors from aggregate value
        group['errors2'] = agg_errors - group.loc[:, 'errors']
        # create sample predictedYSmooth
        group['predictedYSmooth'] = np.random.rand(group.shape[0])
        # get less than
        errPos = group.loc[group['errors2'] >= 0, 'errors2']

        col = 'density'

        res = self.WB._transform_function(group,
                                     col=col,
                                     vartype='Continuous',
                                     groupby_var='Type')

        correct = np.nanmedian(errPos)

        self.assertEqual(res['errPos'].values.tolist()[0],
                         round(correct, GLOBAL_ROUND),
                         """transform_func classification not returning 
                         correct aggregate value for errPos""")

    def test_transform_function_errNeg_classification(self):
        """test correct errPos output from transform_func in classification context"""
        self.WB.model_type = 'classification'
        # get copy of group
        group = self.wine.groupby('Type').get_group('White').copy(deep=True)
        # create random errors
        group.loc[:, 'errors'] = np.random.uniform(-1, 1, group.shape[0])
        # get aggregate value of errors
        agg_errors = np.nanmedian(group['errors'])
        # subtract real errors from aggregate value
        group['errors2'] = agg_errors - group.loc[:, 'errors']
        # create sample predictedYSmooth
        group['predictedYSmooth'] = np.random.rand(group.shape[0])
        # get less than
        errNeg = group.loc[group['errors2'] <= 0, 'errors2']

        col = 'density'

        res = self.WB._transform_function(group,
                                     col=col,
                                     vartype='Continuous',
                                     groupby_var='Type')

        correct = np.nanmedian(errNeg)

        self.assertEqual(res['errNeg'].values.tolist()[0],
                         round(correct, GLOBAL_ROUND),
                         """transform_func classification not returning 
                         correct aggregate value for errNeg""")

    def test_var_check_output(self):
        """test return of json like object after var_check run"""

        self.WB._cat_df['errors'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])
        self.WB._cat_df['predictedYSmooth'] = np.random.uniform(-1, 1, self.WB._cat_df.shape[0])

        out = self.WB._var_check('fixed acidity',
                      'Type')

        self.assertIsInstance(out,
                              dict,
                              """var_check didn't return json like object after run""")
