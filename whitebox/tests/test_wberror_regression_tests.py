#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from whitebox.eval import WhiteBoxError
    import whitebox.utils.utils as utils
except:
    from eval import WhiteBoxError
    import utils.utils


class TestWhiteBoxError(unittest.TestCase):

    def setUp(self):
        """create data framework for regression and classification models"""
        reg_ydepend, reg_groupby, regression = utils.create_synthetic(nrows=300, ncols=5,
                                                                      num_groupby=1,
                                                                      ncat=2,
                                                                      mod_type='regression')

        reg_cats = pd.get_dummies(
            regression.loc[:, regression.columns != reg_ydepend].select_dtypes(include=['O', 'category']),
            prefix='col')

        # merge with numeric
        reg_mod_df = pd.concat([reg_cats, regression.select_dtypes(include=[np.number])], axis=1)

        modobj_reg = RandomForestRegressor()

        modobj_reg.fit(reg_mod_df.loc[:, reg_mod_df.columns != reg_ydepend],
                       reg_mod_df.loc[:, reg_ydepend])

        self.reg_ydepend = reg_ydepend
        self.reg_groupby = reg_groupby
        self.regressiondf = regression
        self.reg_mod_df = reg_mod_df
        self.modobj_reg = modobj_reg

    def tearDown(self):
        """Teardown"""
        pass


    def test_create_group_errors_errpos(self):
        """test errpos in create group errors outputs"""

        testdf = self.regressiondf.copy(deep=True)

        testdf['errors'] = np.random.uniform(-1, 1, size=testdf.shape[0])

        wb = WhiteBoxError(self.modobj_reg,
                           cat_df=testdf,
                           model_df=self.reg_mod_df,
                           groupbyvars=self.reg_groupby,
                           ydepend=self.reg_ydepend,
                           featuredict=None,
                           autoformat=True)

        errors = wb._create_group_errors(testdf.groupby(self.reg_groupby[0]).get_group('level_0'))

        self.assertIn('errPos', errors.columns,
                      msg="""errPos not in errors column after _create_group_errors run on regression case""")

    def test_create_group_errors_errneg(self):
        """test errneg in create group errors outputs"""

        testdf = self.regressiondf.copy(deep=True)

        testdf['errors'] = np.random.uniform(-1, 1, size=testdf.shape[0])

        wb = WhiteBoxError(self.modobj_reg,
                           cat_df=testdf,
                           model_df=self.reg_mod_df,
                           groupbyvars=self.reg_groupby,
                           ydepend=self.reg_ydepend,
                           featuredict=None,
                           autoformat=True)

        errors = wb._create_group_errors(testdf.groupby(self.reg_groupby[0]).get_group('level_0'))

        self.assertIn('errNeg', errors.columns,
                      msg="""errNeg not in errors column after _create_group_errors run on regression case""")

    def test_create_group_errors_nonzero(self):
        """test zeros not in create group errors outputs"""

        testdf = self.regressiondf.copy(deep=True)

        testdf['errors'] = np.random.uniform(-1, 1, size=testdf.shape[0])

        # create synthetic zero
        testdf.loc[10, 'errors'] = 0

        wb = WhiteBoxError(self.modobj_reg,
                           cat_df=testdf,
                           model_df=self.reg_mod_df,
                           groupbyvars=self.reg_groupby,
                           ydepend=self.reg_ydepend,
                           featuredict=None,
                           autoformat=True)

        errors = wb._create_group_errors(testdf.groupby(self.reg_groupby[0]).get_group('level_0'))

        all_errors = errors['errNeg'].values.tolist() + errors['errPos'].values.tolist()

        self.assertNotIn(0, all_errors,
                      msg="""0 found in create_group_errors output""")

    def test_categorical_transform_flat(self):
        """test aggregate flat output from _transform_function for categorical case"""
        testdf = self.regressiondf.copy(deep=True)

        testdf['errors'] = np.random.uniform(-1, 1, size=testdf.shape[0])

        # create synthetic zero
        testdf.loc[10, 'errors'] = 0

        wb = WhiteBoxError(self.modobj_reg,
                           cat_df=testdf,
                           model_df=self.reg_mod_df,
                           groupbyvars=self.reg_groupby,
                           ydepend=self.reg_ydepend,
                           featuredict=None,
                           autoformat=True)

        # get categorical column
        cat_col = testdf.select_dtypes(include=['O']).columns

        # get group
        group = testdf.groupby(self.reg_groupby[0]).get_group('level_0')

        transform_out = wb._transform_function(group,
                                               groupby=self.reg_groupby[0],
                                               col=cat_col,
                                               vartype='Categorical')

        self.assertEqual(transform_out.filter(regex='errPos').shape[0], 1,
                         msg="""transform_out for categorical variable does not return a single row
                                aggregate value. Returned size: {}""".format(transform_out.shape))

    def test_continuous_transform_flat(self):
        """test aggregate flat output from _transform_function for continuous case"""
        testdf = self.regressiondf.copy(deep=True)

        testdf['errors'] = np.random.uniform(-1, 1, size=testdf.shape[0])
        testdf['predictedYSmooth'] = np.random.rand(testdf.shape[0])

        # create synthetic zero
        testdf.loc[10, 'errors'] = 0

        wb = WhiteBoxError(self.modobj_reg,
                           cat_df=testdf,
                           model_df=self.reg_mod_df,
                           groupbyvars=self.reg_groupby,
                           ydepend=self.reg_ydepend,
                           featuredict=None,
                           autoformat=True)


        # get categorical column
        cont_col = testdf.select_dtypes(include=[np.number]).columns[1]

        # get group
        group = testdf.groupby(self.reg_groupby[0]).get_group('level_0')

        transform_out = wb._transform_function(group,
                                               groupby=self.reg_groupby[0],
                                               col=cont_col,
                                               vartype='Continuous')

        self.assertEqual(transform_out['errPos'].shape[0], 1,
                         msg="""transform_out for continuous variable does not return a single row
                                aggregate value. Returned size: {}""".format(transform_out.shape))



import pandas as pd
df = pd.DataFrame({'col1': np.random.rand(100),
                   'col2': np.random.rand(100)})

df.agg({'col1': np.mean,
        'col2': np.std})

from scipy import stats

agg_dict = {'continuous_col': np.max,
            'continuous_groupby': stats.mode,
            'categorical_col': self.aggregate_func,
            'categorical_groupby': self.aggregate_func}

if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()