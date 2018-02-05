#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
import pandas as pd
import numpy as np
from whitebox.whitebox import WhiteBoxSensitivity


__author__ = "Jason Lewris, Daniel Byler, Venkat Gangavarapu, Shruti Panda, Shanti Jha"
__credits__ = ["Brian Ray"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jason Lewris"
__email__ = "jlewris@deloitte.com"
__status__ = "Beta"


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

        df['Type'] = pd.Categorical(df['Type'])
        df['Subtype'] = pd.Categorical(df['Subtype'])

        self.cat_df = df

        model_df = df.copy(deep=True)
        model_df['Type'] = model_df['Type'].cat.codes
        model_df['Subtype'] = model_df['Subtype'].cat.codes

        self.model_df = model_df
        modelobj.fit(self.model_df.loc[:, self.model_df.columns != 'target'],
                     self.model_df['target'])

        self.modelobj = modelobj

    def test_wbox_sensitivity_continuous_slice_outputs(self):
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
                                    featuredict=None)

        results = wb._continuous_slice(
                                        iris.groupby('Type').get_group('white'),
                                        groupby='Subtype',
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
                                    featuredict=None)

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
                                    featuredict=None)

        wb.run()

        var_check = wb._var_check(col='sepall',
                     groupby='Type')

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
                                    featuredict=None)

        wb.run()

        var_check = wb._var_check(
                                    col='Subtype',
                                    groupby='Type')

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


if __name__ == '__main__':
    unittest.main()

