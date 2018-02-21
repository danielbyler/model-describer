#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
import pandas as pd
import numpy as np

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from whitebox.eval import WhiteBoxSensitivity
except:
    from whitebox.eval import WhiteBoxSensitivity


class TestWhiteBoxSensitivity(unittest.TestCase):

    def setUp(self):
        # load iris data for testing WhiteBox functionality
        self.ydepend = 'target'
        self.groupbyvars = ['Type', 'Subtype']

        iris_data = datasets.load_iris()
        df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                          columns=['sepall', 'sepalw', 'petall', 'petalw', 'target'])

        df['Type'] = ['white'] * 75 + ['red'] * 75
        df['Subtype'] = ['bin1'] * 50 + ['bin2'] * 50 + ['bin3'] * 50

        # set up randomforestregressor
        modelobj = RandomForestRegressor()

        self.cat_df = df

        model_df = df.copy(deep=True)

        # create dummies
        model_df = pd.concat([model_df.select_dtypes(include=[np.number]),
                              pd.get_dummies(model_df.select_dtypes(include=['O', 'category']))], axis=1)

        self.model_df = model_df
        modelobj.fit(self.model_df.loc[:, self.model_df.columns != 'target'],
                     self.model_df['target'])

        self.modelobj = modelobj

    def test_wbox_sensitivity_continuous_slice_output_groupByValue(self):
        # test that groupByValue is inserted into continuous slice results for Sensitivity
        # copy iris data
        iris = self.cat_df.copy(deep=True)
        iris['errors'] = np.random.rand(iris.shape[0], 1)
        iris['predictedYSmooth'] = np.random.rand(iris.shape[0], 1)
        iris['diff'] = np.random.rand(iris.shape[0], 1)

        wb = WhiteBoxSensitivity(
                                    modelobj=self.modelobj,
                                    model_df=self.model_df,
                                    ydepend='target',
                                    groupbyvars=['Type'],
                                    cat_df=self.cat_df,
                                    featuredict=None,
                                    autoformat_types=True)

        results = wb._continuous_slice(
                                        iris.groupby('Type').get_group('white'),
                                        groupby_var='Subtype',
                                        col='sepall',
                                        vartype='Continuous')

        self.assertIn('groupByValue', results.columns,
                      msg="""groupByValue not found in continuous slice results.
                      \nColumns: {}""".format(results.columns))

        self.assertIn('groupByVarName', results.columns,
                      msg="""groupByVarName not found in continuous slice results.
                              \nColumns: {}""".format(results.columns))

        self.assertIn('predictedYSmooth', results.columns,
                      msg="""predictedYSmooth not found in continuous slice results.
                                              \nColumns: {}""".format(results.columns))

        self.assertIn('sepall', results.columns,
                      msg="""sulphates not found in continuous slice results.
                                                      \nColumns: {}""".format(results.columns))

    def test_wbox_class_name(self):
        # test that WhiteBoxError class name is WhiteBoxError in the __class__.__name__
        wb = WhiteBoxSensitivity(
                                    modelobj=self.modelobj,
                                    model_df=self.model_df,
                                    ydepend='target',
                                    groupbyvars=['Type'],
                                    cat_df=self.cat_df,
                                    featuredict=None,
                                    autoformat_types=True)

        self.assertEqual(wb.__class__.__name__,
                         'WhiteBoxSensitivity',
                         msg="""Class name expected to be WhiteBoxError.
                         \nCurrent class name is: {}""".format(wb.__class__.__name__))

    def test_whitebox_var_check(self):
        # test case for var_check method of WhiteBoxError - checking outputs
        iris = self.cat_df.copy(deep=True)
        iris['errors'] = np.random.rand(iris.shape[0], 1)
        iris['predictedYSmooth'] = np.random.rand(iris.shape[0], 1)
        iris['diff'] = np.random.rand(iris.shape[0], 1)

        modeldf = self.model_df.copy(deep=True)
        modeldf['predictedYSmooth'] = np.random.rand(modeldf.shape[0], 1)

        wb = WhiteBoxSensitivity(
                                    modelobj=self.modelobj,
                                    model_df=self.model_df,
                                    ydepend='target',
                                    groupbyvars=['Type'],
                                    cat_df=self.cat_df,
                                    featuredict=None,
                                    autoformat_types=True)

        wb.run(output_type=None)

        var_check = wb._var_check(col='sepall',
                     groupby_var='Type')

        self.assertIn('Type', var_check.keys(),
                      msg="""Type not in json output from var_check for continuous variable
                        \noutput keys: {}""".format(var_check.keys()))

        self.assertEqual(var_check['Type'], 'Continuous',
                         msg="""var check Type not Continuous for Continuous case.
                         \nVar check Type: {}""".format(var_check['Type']))

        self.assertIn('Data', var_check.keys(),
                      msg="""Data key not in var check output.
                      \nKeys: {}""".format(var_check.keys()))

        self.assertIn('Change', var_check.keys(),
                      msg="""Data key not in var check output.
                              \nKeys: {}""".format(var_check.keys()))

        self.assertIsInstance(var_check['Data'], list,
                              msg="""var check data output not of type list.
                              \nReturned Type: {}""".format(type(var_check['Data'])))

    def test_whitebox_var_check_categorical(self):
        # test case for var_check method of WhiteBoxError - checking outputs
        iris = self.cat_df.copy(deep=True)
        iris['errors'] = np.random.rand(iris.shape[0], 1)
        iris['predictedYSmooth'] = np.random.rand(iris.shape[0], 1)
        iris['diff'] = np.random.rand(iris.shape[0], 1)

        wb = WhiteBoxSensitivity(
                                    modelobj=self.modelobj,
                                    model_df=self.model_df,
                                    ydepend='target',
                                    groupbyvars=['Type'],
                                    cat_df=self.cat_df,
                                    featuredict=None,
                                    autoformat_types=True)

        wb.run(output_type=None)

        var_check = wb._var_check(
                                    col='Subtype',
                                    groupby_var='Type')

        self.assertIn('Type', var_check.keys(),
                      msg="""Type not in json output from var_check for categorical variable
                        \noutput keys: {}""".format(var_check.keys()))

        self.assertEqual(var_check['Type'], 'Categorical',
                         msg="""var check Type not Categorical for Categorical case.
                         \nVar check Type: {}""".format(var_check['Type']))

        self.assertIn('Data', var_check.keys(),
                      msg="""Data key not in var check output.
                      \nKeys: {}""".format(var_check.keys()))

        self.assertIn('Change', var_check.keys(),
                      msg="""Data key not in var check output.
                              \nKeys: {}""".format(var_check.keys()))

        self.assertIsInstance(var_check['Data'], list,
                              msg="""var check data output not of type list.
                              \nReturned Type: {}""".format(type(var_check['Data'])))

    def test_called_class(self):
        # test case for var_check method of WhiteBoxError - checking outputs
        iris = self.cat_df.copy(deep=True)
        iris['errors'] = np.random.rand(iris.shape[0], 1)
        iris['predictedYSmooth'] = np.random.rand(iris.shape[0], 1)
        iris['diff'] = np.random.rand(iris.shape[0], 1)

        wb = WhiteBoxSensitivity(
            modelobj=self.modelobj,
            model_df=self.model_df,
            ydepend='target',
            groupbyvars=['Type'],
            cat_df=self.cat_df,
            featuredict=None,
            autoformat_types=True)

        self.assertEqual(wb.called_class, 'WhiteBoxSensitivity',
                         msg="""WhiteBoxBase unable to detect correct super class
                                \nAssigned class: {}""".format(wb.called_class))


if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()

