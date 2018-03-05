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
    from mdesc.utils import check_utils

except ImportError:
    from utils import check_utils

Check = check_utils.CheckInputs

class TestCheckUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_checks_is_regression_regr_model(self):
        """ test output is regression model """
        mod =RandomForestRegressor()

        engine, modtype = Check.is_regression(mod)

        self.assertEqual(modtype,
                         'regression',
                         """check_is_regression output wrong model type for regression""")
    def test_checks_is_regression_predict(self):
        """ test is_regression engine name is predict for regression """

        mod = RandomForestRegressor()

        engine, modtype = Check.is_regression(mod)

        self.assertEqual(engine.__name__,
                         'predict',
                         """is_regression did not output engine predict on regression""")

    def test_checks_is_regression_classification_model(self):
        """ test output is classification model """
        mod = RandomForestClassifier()

        engine, modtype = Check.is_regression(mod)

        self.assertEqual(modtype,
                         'classification',
                         """check_is_regression output wrong model type for classification""")
    def test_checks_is_regression_predict_proba(self):
        """ test is_regression engine name is predict for classification """

        mod = RandomForestClassifier()

        engine, modtype = Check.is_regression(mod)

        self.assertEqual(engine.__name__,
                         'predict_proba',
                         """is_regression did not output engine predict on classification""")

    def test_check_keepfeaturelist_none(self):
        """ check all cols returned for None input param """
        df = pd.DataFrame({'col1': np.random.rand(100),
                           'col2': np.random.rand(100),
                           'col3': np.random.rand(100)})

        keepfeatures = Check.check_keepfeaturelist(None,
                                                   df)

        self.assertEqual(keepfeatures,
                         df.columns.values.tolist(),
                         """check_keepfeaturelist did not output all cols for None param""")

    def test_check_keepfeaturelist_invalid_feature(self):
        """ assert ValueError raised when non feature used """
        df = pd.DataFrame({'col1': np.random.rand(100),
                           'col2': np.random.rand(100),
                           'col3': np.random.rand(100)})

        with self.assertRaises(ValueError) as context:
            testlist = ['test', 'test2']
            keep = Check.check_keepfeaturelist(testlist,
                                               df)

        self.assertTrue(context,
                        """ValueError not raised when feature not present in dataframe""")

    def test_check_keepfeaturelist_return_list(self):
        """ check_keepfeaturelist returns original featurelist """
        df = pd.DataFrame({'col1': np.random.rand(100),
                           'col2': np.random.rand(100),
                           'col3': np.random.rand(100)})

        keep = ['col1', 'col2']

        res = Check.check_keepfeaturelist(keep,
                                          df)

        self.assertEqual(res,
                         keep,
                         """check_keepfeaturelist does not return origina list when checks pass""")

    def test_check_agg_func(self):
        """ check_agg_func returns func when appropriate """
        res = Check.check_agg_func(np.median)

        self.assertEqual(res.__name__,
                         'median',
                         """check_agg_func doesnt return original function""")

    def test_check_agg_func_cails(self):
        """test check_agg_func fails with inappropriate function"""

        with self.assertRaises(TypeError) as context:
            func = lambda x: x - 10
            Check.check_agg_func(func)

        self.assertTrue(context,
                        """check_agg_func didn't raise TypeError""")

    def test_check_cat_df_none(self):
        """ check_cat_df returns model_df when None"""
        df = pd.DataFrame({'col1': np.random.rand(100),
                           'col2': np.random.rand(100),
                           'col3': np.random.rand(100)})

        res = Check.check_cat_df(None, df)
        self.assertEqual(df.shape,
                         res.shape,
                         """check_cat_df didn't return model_df when None""")


    def test_check_cat_df_consistent_shape(self):
        """check consistent shapes ValueError raised"""
        df = pd.DataFrame({'col1': np.random.rand(100),
                           'col2': np.random.rand(100),
                           'col3': np.random.rand(100)})

        df2 = pd.DataFrame({'col1': np.random.rand(99),
                            'col2': np.random.rand(99),
                            'col3': np.random.rand(99)})

        with self.assertRaises(ValueError) as context:

            Check.check_cat_df(df, df2)

        self.assertTrue(context,
                        """inconsistent shapes error didn't raise""")

    def test_check_cat_df_input_shapes(self):
        """check two separtae dataframes with uneven shapes results in
        error"""

        df = pd.DataFrame({'col1': np.random.rand(100),
                           'col2': np.random.rand(100)})

        df2 = pd.DataFrame({'col1': np.random.rand(100),
                            'col2': np.random.rand(100)})

        # change one index
        df2.index.values[10] = 1100

        with self.assertRaises(ValueError) as context:
            Check.check_cat_df(df, df2)

        self.assertTrue(context,
                        """Inconsistent index error wasn't raised""")

    def test_check_modelobj_no_predict(self):
        """check_modelobj raises error when predict attribute not present"""

        modobj = lambda x: x

        with self.assertRaises(ValueError) as context:
            Check.check_modelobj(modobj)

        self.assertTrue(context,
                        """ValueError not raised when predict attribute
                        not found""")





if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()