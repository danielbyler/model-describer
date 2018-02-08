#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from whitebox.whitebox import WhiteBoxError
except:
    from whitebox import WhiteBoxError

__author__ = "Jason Lewris, Daniel Byler, Venkat Gangavarapu, Shruti Panda, Shanti Jha"
__credits__ = ["Brian Ray"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jason Lewris"
__email__ = "jlewris@deloitte.com"
__status__ = "Beta"


class TestWhiteBoxError(unittest.TestCase):

    def setUp(self):
        # load iris data for testing WhiteBox functionality
        iris_data = datasets.load_iris()
        self.iris = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                                 columns=['sepall', 'sepalw', 'petall', 'petalw', 'target'])

        self.iris['Type'] = ['white'] * 75 + ['red'] * 75
        self.iris['Type2'] = ['blue'] * 75 + ['yellow'] * 75
        # convert sepalw to categorical for testing
        self.iris['Type'] = pd.Categorical(self.iris['Type'])
        self.iris['Type2'] = pd.Categorical(self.iris['Type2'])
        # create cat_df and convert iris categories to numbers
        self.cat_df = self.iris.copy(deep=True)
        self.iris['Type'] = self.iris['Type'].cat.codes
        self.iris['Type2'] = self.iris['Type2'].cat.codes
        # set up randomforestregressor
        modelobj = RandomForestRegressor()

        modelobj.fit(self.iris.loc[:, self.iris.columns != 'target'],
                     self.iris['target'])

        self.modelobj = modelobj

    def test_wberror_notfitted(self):
        # test to ensure unfitted models are caught early
        df = pd.DataFrame({'col1': list(range(100)),
                           'col2': list(range(100))})

        # set up randomforestregressor
        modelobj = RandomForestRegressor()

        error_message = ''
        try:
            WhiteBoxError(
                            modelobj=modelobj,
                            model_df=df,
                            ydepend='col1',
                            cat_df=df,
                            groupbyvars=['col2'])
        except Exception as e:
            error_message += str(e)

            self.assertIn(
                            'not fitted', 
                            error_message,
                            """WhiteBoxError not correctly detecting unfitted models""")

    def test_run_outputs(self):
        # test whether outputs are assigned to instance after run
        wb = WhiteBoxError(
                            modelobj=self.modelobj,
                            model_df=self.iris,
                            ydepend='target',
                            groupbyvars=['Type'],
                            cat_df=self.cat_df)

        wb.run()

        self.assertIsInstance(wb.outputs, list,
                              msg="""WhiteBoxError is not producing list of outputs after run
                                       \nProducing class: {}""".format(type(wb.outputs)))

    def test_wberror_predict_errors(self):
        # test if error column created after predict method run in whiteboxerror
        wb = WhiteBoxError(
                            modelobj=self.modelobj,
                            model_df=self.iris,
                            ydepend='target',
                            groupbyvars=['Type'],
                            cat_df=self.cat_df)

        wb._predict()

        self.assertIn('errors', wb.cat_df.columns,
                      msg="""errors not in instance cat_df. Only cols present:
                      {}""".format(wb.cat_df.columns))

    def test_wberror_predict_predictedYSmooth_cat_df(self):
        # test if predictedYSmooth column created after predict method run in whiteboxerror
        # in cat_df
        wb = WhiteBoxError(
                            modelobj=self.modelobj,
                            model_df=self.iris,
                            ydepend='target',
                            groupbyvars=['Type'],
                            cat_df=self.cat_df)

        wb._predict()

        self.assertIn('predictedYSmooth', wb.cat_df.columns,
                      msg="""predictedYSmooth not in instances cat_df. Only cols present:
                      {}""".format(wb.cat_df.columns))

    def test_wberror_predict_predictedYSmooth_model_df(self):
        # test if predictedYSmooth column created after predict method run in whiteboxerror
        # in model_df
        wb = WhiteBoxError(
                            modelobj=self.modelobj,
                            model_df=self.iris,
                            ydepend='target',
                            groupbyvars=['Type'],
                            cat_df=self.cat_df)

        wb._predict()

        self.assertIn('predictedYSmooth', wb.model_df.columns,
                      msg="""predictedYSmooth not in instances model_df. \nOnly cols present:
                      {}""".format(wb.model_df.columns))

    def test_wberror_predict_predicted(self):
        # test whether predictedYSmooth column is present after whiteboxerror predict method called
        wb = WhiteBoxError(
                            modelobj=self.modelobj,
                            model_df=self.iris,
                            ydepend='target',
                            groupbyvars=['Type'],
                            cat_df=self.cat_df)

        wb._predict()

        self.assertIn('predictedYSmooth', wb.cat_df.columns,
                      msg="""predictedYSmooth not in instance cat_df. Only cols present:
                      {}""".format(wb.cat_df.columns))

    def test_wberror_transform_errPos(self):
        # test whether errNeg column is present after running whiteboxerror transform_function
        # on slice of data
        wb = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df)

        wb._predict()

        # create partial func
        cont_slice_partial = partial(wb._continuous_slice,
                                     col='sepalw',
                                     vartype='Continuous',
                                     groupby='Type')

        # run transform function
        errors = wb.cat_df.groupby(['Type']).apply(cont_slice_partial)

        self.assertIn('errPos', errors.columns,
                      msg="""errPos not in errordf after transform_function. Only cols present:
                          {}""".format(errors.columns))

    def test_wberror_transform_errNeg(self):
        # test whether errNeg column is present after running whiteboxerror transform_function
        # on slice of data
        wb = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df)

        wb._predict()

        # create partial func
        cont_slice_partial = partial(wb._continuous_slice,
                                     col='sepalw',
                                     vartype='Continuous',
                                     groupby='Type')

        # run transform function
        errors = wb.cat_df.groupby(['Type']).apply(cont_slice_partial)

        self.assertIn('errNeg', errors.columns,
                      msg="""errNeg not in errordf after transform_function. Only cols present:
                          {}""".format(errors.columns))

    def test_featuredict_subset(self):
        # create the featuredict
        featuredict = {'Type': 'type',
                       'target': 'target',
                       'sepall': 'sepall'}

        wb = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df,
                           featuredict=featuredict)

        if not isinstance(featuredict.keys(), list):
            featuredictkeys = list(featuredict.keys())
        else:
            featuredictkeys = featuredict.keys()

        self.assertListEqual(featuredictkeys, wb.cat_df.columns.values.tolist(),
                             """Featuredict and wb instance columns dont match.
                             \nFeaturedict: {}
                             \nwb Instance cat df: {}""".format(featuredictkeys, wb.cat_df.columns.values.tolist()))

    def test_whitebox_no_featuredict(self):
        wb = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df,
                           featuredict=None)

        self.assertEqual(self.iris.shape[1], len(wb.featuredict.keys()),
                         """When featuredict is not present, featuredict is not being
                         populated correctly with dataframe columns.
                         \nDataframe Columns: {}
                         \nFeaturedict Keys: {}""".format(
                                                            self.iris.columns,
                                                            wb.featuredict.keys()))

    def test_whitebox_var_check_continuous(self):
        # test case for var_check method of WhiteBoxError - checking outputs
        iris = self.cat_df.copy(deep=True)
        iris['errors'] = np.random.rand(iris.shape[0], 1)
        iris['predictedYSmooth'] = np.random.rand(iris.shape[0], 1)

        wb = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=iris,
                           featuredict=None)

        var_check = wb._var_check(
                                    col='sepall',
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

        self.assertIsInstance(var_check['Data'], list,
                              msg="""var check data output not of type list.
                              \nReturned Type: {}""".format(type(var_check['Data'])))

    def test_whitebox_var_check(self):
        # test case for var_check method for categorical WhiteBoxError - checking outputs
        iris = self.cat_df.copy(deep=True)
        iris['errors'] = np.random.rand(iris.shape[0], 1)
        iris['predictedYSmooth'] = np.random.rand(iris.shape[0], 1)

        wb = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=iris,
                           featuredict=None)

        var_check = wb._var_check(
                                    col='Type2',
                                    groupby='Type')

        self.assertIn('Type', var_check.keys(),
                      msg="""Type not in json output from var_check for categorical variable
                        \noutput keys: {}""".format(var_check.keys()))

        self.assertEqual(var_check['Type'], 'Categorical',
                         msg="""var check Type not Continuous for Categorical case.
                         \nVar check Type: {}""".format(var_check['Type']))

        self.assertIn('Data', var_check.keys(),
                      msg="""Data key not in var check output.
                      \nKeys: {}""".format(var_check.keys()))

        self.assertIsInstance(var_check['Data'], list,
                              msg="""var check data output not of type list.
                              \nReturned Type: {}""".format(type(var_check['Data'])))

    def test_wbox_class_name(self):
        # test that WhiteBoxError class name is WhiteBoxError in the __class__.__name__
        wb = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df,
                           featuredict=None)

        self.assertEqual(wb.__class__.__name__,
                         'WhiteBoxError',
                         msg="""Class name expected to be WhiteBoxError.
                         \nCurrent class name is: {}""".format(wb.__class__.__name__))

    def test_wbox_modelobj_switch(self):
        # test that whitebox can accurately detect classification model
        clf = RandomForestClassifier()

        clf.fit(self.iris.loc[:, self.iris.columns != 'target'],
                self.iris.loc[:, 'target'])

        wb = WhiteBoxError(modelobj=clf,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df,
                           featuredict=None)

        self.assertEqual(wb.model_type, 'classification',
                         """WhiteBoxBase unable to detect classification model.
                         \nAssigned: {} as model type""".format(wb.model_type))

    def test_wbox_modelobj_switch_regression(self):
        # test that whitebox can accurately detect regression model
        clf = RandomForestClassifier()
        wb = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df,
                           featuredict=None)

        self.assertEqual(wb.model_type, 'regression',
                         """WhiteBoxBase unable to detect classification model.
                         \nAssigned: {} as model type""".format(wb.model_type))

    def test_wbox_error_continuous_slice_outputs(self):
        # test that groupByValue is inserted into continuous slice results
        # copy iris data
        iris = self.cat_df.copy(deep=True)
        iris['errors'] = np.random.rand(iris.shape[0], 1)
        iris['predictedYSmooth'] = np.random.rand(iris.shape[0], 1)

        wb = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df,
                           featuredict=None)

        results = wb._continuous_slice(
                                        iris.groupby('Type').get_group('white'),
                                        groupby='Type2',
                                        col='sepall',
                                        vartype='Continuous')

        self.assertIn('groupByValue', results.columns,
                      msg="""groupByValue not found in continuous slice results.
                      \nColumns: {}""".format(results.columns))

        self.assertIn('groupByVarName', results.columns,
                      msg="""groupByVarName not found in continuous slice results.
                              \nColumns: {}""".format(results.columns))

        self.assertIn('errNeg', results.columns,
                      msg="""errNeg not found in continuous slice results.
                                      \nColumns: {}""".format(results.columns))

        self.assertIn('errPos', results.columns,
                      msg="""errPos not found in continuous slice results.
                                      \nColumns: {}""".format(results.columns))

        self.assertIn('predictedYSmooth', results.columns,
                      msg="""predictedYSmooth not found in continuous slice results.
                                              \nColumns: {}""".format(results.columns))

        self.assertIn('sepall', results.columns,
                      msg="""sulphates not found in continuous slice results.
                                                      \nColumns: {}""".format(results.columns))


if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()