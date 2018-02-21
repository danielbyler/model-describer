import unittest
from sklearn import datasets
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from whitebox import utils
    from whitebox.eval import WhiteBoxError
except:
    import utils
    from eval import WhiteBoxError


class GlobalWBErrorTestCase(unittest.TestCase):
    """run a full evaluation of all whitebox code for classification and regression cases"""

    def setUp(self):
        """create data framework for regression and classification models"""
        reg_ydepend, reg_groupby, regression = utils.create_synthetic(nrows=300, ncols=5,
                                                              num_groupby=1,
                                                              ncat=2,
                                                              mod_type='regression')

        reg_cats = pd.get_dummies(regression.loc[:, regression.columns != reg_ydepend].select_dtypes(include=['O', 'category']),
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
        """"Teardown"""
        pass

    def test_error_regression_nofdict(self):
        """Test full run of whitebox error with no featuredict"""
        wb = WhiteBoxError(self.modobj_reg,
                           cat_df=self.regressiondf,
                           model_df=self.reg_mod_df,
                           groupbyvars=self.reg_groupby,
                           ydepend=self.reg_ydepend,
                           featuredict=None,
                           autoformat=True)

        with self.assertRaises(Exception):
            try:
                wb.run(output_type=None)
            except:
                pass
            else:
                raise Exception

    def test_error_regression_fdict(self):
        """test full run of whitebox error with featuredict"""

        feature_dict = {'col1': 'TESTCOL',
                        'col2': 'TESTCOL_3'}

        wb = WhiteBoxError(self.modobj_reg,
                           cat_df=self.regressiondf,
                           model_df=self.reg_mod_df,
                           groupbyvars=self.reg_groupby,
                           ydepend=self.reg_ydepend,
                           featuredict=feature_dict,
                           autoformat=True)

        with self.assertRaises(Exception):
            try:
                wb.run(output_type=None)
            except:
                pass
            else:
                raise Exception