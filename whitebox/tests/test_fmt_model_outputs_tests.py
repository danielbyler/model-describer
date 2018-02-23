#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from whitebox.utils import fmt_model_outputs
    from whitebox.utils import utils

except:
    from utils import fmt_model_outputs
    from utils import utils


class TestWhiteBoxError(unittest.TestCase):

    def setUp(self):
        ydepend, groupby, df = utils.create_synthetic(nrows=1000,
                                                      ncols=5,
                                                      num_groupby=1,
                                                      ncat=1,
                                                      mod_type='regression',
                                                      )

        ydepend_class, groupby_class, df_class = utils.create_synthetic(nrows=1000,
                                                                        ncols=5,
                                                                        num_groupby=1,
                                                                        ncat=1,
                                                                        mod_type='classification',
                                                                        )

        df = df.select_dtypes(include=[np.number])
        df_class = df_class.select_dtypes(include=[np.number])

        self.df = df
        self.df_class = df_class

    def test_fmt_sklearn_preds_regression(self):
        """test fmt_sklearn_preds on regression case"""

        modelobj_regr = RandomForestRegressor()

        modelobj_regr.fit(self.df.loc[:, self.df.columns != 'target'],
                          self.df.loc[:, 'target'])

        fmtd_outputs, _ = fmt_model_outputs.fmt_sklearn_preds(getattr(modelobj_regr, 'predict'),
                                            modelobj_regr,
                                            self.df,
                                            self.df,
                                            'target',
                                            'regression')

        self.assertIn('predictedYSmooth',
                      fmtd_outputs.columns.values,
                      """fmt_sklearn_preds on regression case does not return predictions""")

    def test_fmt_sklearn_preds_classification(self):
        """test fmt_sklearn_preds on classification case"""

        modelobj_class = RandomForestClassifier()

        modelobj_class.fit(self.df_class.loc[:, self.df_class.columns != 'target'],
                           self.df_class.loc[:, 'target'])

        fmtd_outputs, _ = fmt_model_outputs.fmt_sklearn_preds(getattr(modelobj_class, 'predict_proba'),
                                            modelobj_class,
                                            self.df_class,
                                            self.df_class,
                                            'target',
                                            'classification')

        self.assertIn('predictedYSmooth',
                      fmtd_outputs,
                      """fmt_sklearn_preds on classificaiton case does not return predictions""")



ydepend, groupby, df = utils.create_synthetic(nrows=1000,
                                                      ncols=5,
                                                      num_groupby=1,
                                                      ncat=1,
                                                      mod_type='regression',
                                                      )

ydepend_class, groupby_class, df_class = utils.create_synthetic(nrows=1000,
                                                                ncols=5,
                                                                num_groupby=1,
                                                                ncat=1,
                                                                mod_type='classification',
                                                                )

df = df.select_dtypes(include=[np.number])

modelobj = GradientBoostingRegressor()
modelobj = RandomForestRegressor()

from sklearn.utils.validation import check_is_fitted

modelobj.fit(df.loc[:, df.columns != 'target'],
             df.loc[:, 'target'])

check_is_fitted(modelobj, 'estimators_')