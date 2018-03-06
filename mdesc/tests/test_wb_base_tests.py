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
    from base import MdescBase
    from eval import ErrorViz


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
                           'sulphates']

        wine_sub['alcohol'] = wine_sub['alcohol'].astype('object')

        self.WB = ErrorViz(modelobj=modelObjc,
                           model_df=mod_df,
                           ydepend=ydepend,
                           cat_df=wine_sub,
                           groupbyvars=['Type'],
                           keepfeaturelist=keepfeaturelist,
                           verbose=None,
                           autoformat_types=True)

        self.wine = wine

    def test_continuous_slice_greater_100(self):
        """ test continuous slice output for continuous variable with over 100 datapoints """

        cur_col = 'density'

        group = self.wine.groupby('Type').get_group('White')

        groupby_var = 'Type'

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])
        group['predictedYSmooth'] = np.random.rand(group.shape[0])

        res = self.WB._continuous_slice(group,
                                   col=cur_col,
                                   groupby_var=groupby_var)

        self.assertEqual(res.shape[0],
                         100,
                         """continuous_slice returning unexpected shape for 
                         continuous variable greater than 100 and known to have 
                         values for all percentile buckets""")

    def test_continuous_slice_less_100(self):
        """test continuous_slice output for group with less than 100 observations"""

        cur_col = 'density'

        group = self.wine.groupby('Type').get_group('White')

        groupby_var = 'Type'

        group['errors'] = np.random.uniform(-1, 1, group.shape[0])
        group['predictedYSmooth'] = np.random.rand(group.shape[0])

        res = self.WB._continuous_slice(group.iloc[0:20],
                                        col=cur_col,
                                        groupby_var=groupby_var)

        self.assertLessEqual(res.shape[0],
                         20,
                         """continuous_slice returning unexpected shape for 
                         continuous variable less than 100""")

    def test_fmt_raw_df_col_name(self):
        """test fmt_raw_df inserts col_name column"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_raw_df(col='alcohol',
                      groupby_var='Type',
                      cur_group=df)

        self.assertIn('col_name',
                      self.WB.raw_df.columns.tolist(),
                      """col_name not in WB.raw_df after fmt_raw_df run""")

    def test_fmt_raw_df_col_value(self):
        """test fmt_raw_df inserts col_value column"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_raw_df(col='alcohol',
                      groupby_var='Type',
                      cur_group=df)

        self.assertIn('col_value',
                      self.WB.raw_df.columns.tolist(),
                      """col_value not in WB.raw_df after fmt_raw_df run""")

    def test_fmt_raw_df_groupByVar(self):
        """test fmt_raw_df inserts groupByVar column"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_raw_df(col='alcohol',
                      groupby_var='Type',
                      cur_group=df)

        self.assertIn('groupByVar',
                      self.WB.raw_df.columns.tolist(),
                      """groupByVar not in WB.raw_df after fmt_raw_df run""")

    def test_fmt_agg_df_col_value(self):
        """test fmt_agg_df inserts col_value column"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_agg_df(col='alcohol',
                           agg_errors=df)

        self.assertIn('col_value',
                      self.WB.agg_df.columns.tolist(),
                      """col_value not in WB.agg_df after fmt_agg_df run""")

    def test_fmt_agg_df_col_name(self):
        """test fmt_agg_df inserts col_name column"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_agg_df(col='alcohol',
                           agg_errors=df)

        self.assertIn('col_name',
                      self.WB.agg_df.columns.tolist(),
                      """col_name not in WB.agg_df after fmt_agg_df run""")

    def test_get_raw_df(self):
        """test get_raw_df raises error if used before run"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_raw_df(col='alcohol',
                           groupby_var='Type',
                           cur_group=df)

        with self.assertRaises(RuntimeError) as context:
            res = self.WB.get_raw_df()

        self.assertTrue(context,
                        """RunTimeError not raised when get_raw_df run 
                        before WB.run() called""")

    def test_get_agg_df(self):
        """test get_raw_df raises error if used before run"""
        df = pd.DataFrame({'alcohol': np.random.rand(100),
                           'Type': ['White'] * 100,
                           'errPos': np.random.rand(100),
                           'errNeg': np.random.rand(100),
                           'predictedYSmooth': np.random.rand(100)})

        self.WB._fmt_agg_df(col='alcohol',
                           agg_errors=df)

        with self.assertRaises(RuntimeError) as context:
            res = self.WB.get_agg_df()

        self.assertTrue(context,
                        """RunTimeError not raised when get_agg_df run 
                        before WB.run() called""")

    def test_base_run(self):
        """test that run assigns outputs when called"""

        self.WB.run(output_type=None)

        self.assertTrue(hasattr(self.WB, 'outputs'),
                        """WB does not have attribute outputs after 
                        .run() called""")

    def test_base_run_output_type(self):
        """test output type after run called"""

        self.WB.run(output_type=None)

        self.assertIsInstance(self.WB.outputs,
                              list,
                              """WB.outputs not of type list after .run() 
                              called""")

    def test_base_run_output_type_raw_df(self):
        """test output type after run called with output_type='raw_data'"""

        res = self.WB.run(output_type='raw_data')

        self.assertIsInstance(res,
                              pd.DataFrame,
                              """Returned type after .run() called not pd.DataFrame""")

    def test_base_run_output_type_agg_df(self):
        """test output type after run called with output_type='agg_data'"""

        res = self.WB.run(output_type='agg_data')

        self.assertIsInstance(res,
                              pd.DataFrame,
                              """Returned type after .run() called not pd.DataFrame""")




