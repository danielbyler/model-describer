#!/usr/bin/env python

from whitebox import WhiteBoxError
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
import pandas as pd
import numpy as np
import unittest
from functools import partial

__author__ = "Jason Lewris, Daniel Byler, Shruti Panda, Venkat Gangavarapu"
__copyright__ = ""
__credits__ = ["Brian Ray"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Jason Lewris"
__email__ = "jlewris@deloitte.com"
__status__ = "Beta"

class TestWhiteBoxError(unittest.TestCase):

    def setUp(self):
        # load iris data for testing WhiteBox functionality
        iris_data = datasets.load_iris()
        self.iris = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                                 columns = ['sepall', 'sepalw', 'petall', 'petalw', 'target'])

        self.iris['Type'] = ['white'] * 75 + ['red'] * 75

        # convert sepalw to categorical for testing
        self.iris['Type'] = pd.Categorical(self.iris['Type'])
        # create cat_df and convert iris categories to numbers
        self.cat_df = self.iris.copy(deep = True)
        self.iris['Type'] = self.iris['Type'].cat.codes
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
            WhiteBoxError(modelobj = modelobj,
                          model_df = df,
                          ydepend = 'col1',
                          cat_df = df,
                          groupbyvars = ['col2'])
        except Exception as e:
            error_message += str(e)

            self.assertIn('not fitted', error_message,
                      'WhiteBoxError not correctly detecting unfitted models')

    def test_run_outputs(self):

        # test whether outputs are assigned to instance after run
        WB = WhiteBoxError(modelobj = self.modelobj,
                      model_df = self.iris,
                      ydepend = 'target',
                      groupbyvars = ['Type'],
                           cat_df = self.cat_df)


        WB.run()

        self.assertIsInstance(WB.outputs, list,
                              msg = 'WhiteBoxError is not producing list of outputs after run')

    def test_wberror_predict_errors(self):
        # test if error column created after predict method run in whiteboxerror
        WB = WhiteBoxError(modelobj = self.modelobj,
                      model_df = self.iris,
                      ydepend = 'target',
                      groupbyvars = ['Type'],
                           cat_df = self.cat_df)

        WB.predict()

        self.assertIn('errors', WB.cat_df.columns,
                      msg = 'errors not in instance cat_df. Only cols present: '\
                      '{}'.format(WB.cat_df.columns))

    def test_wberror_predict_predicted(self):
        # test whether predictedYSmooth column is present after whiteboxerror predict method called
        WB = WhiteBoxError(modelobj = self.modelobj,
                      model_df = self.iris,
                      ydepend = 'target',
                      groupbyvars = ['Type'],
                           cat_df = self.cat_df)

        WB.predict()

        self.assertIn('predictedYSmooth', WB.cat_df.columns,
                      msg = 'predictedYSmooth not in instance cat_df. Only cols present: '\
                      '{}'.format(WB.cat_df.columns))

    def test_wberror_transform_errPos(self):
        # test whether errNeg column is present after running whiteboxerror transform_function
        # on slice of data
        WB = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df)

        WB.predict()

        # create partial func
        cont_slice_partial = partial(WhiteBoxError.continuous_slice,
                                     col='sepalw',
                                     vartype='Continuous',
                                     groupby='Type')

        # run transform function
        errors = WB.cat_df.groupby(['Type']).apply(cont_slice_partial)

        self.assertIn('errPos', errors.columns,
                      msg='errPos not in errordf after transform_function. Only cols present: ' \
                          '{}'.format(errors.columns))

    def test_wberror_transform_errNeg(self):
        # test whether errNeg column is present after running whiteboxerror transform_function
        # on slice of data
        WB = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df)

        WB.predict()

        # create partial func
        cont_slice_partial = partial(WhiteBoxError.continuous_slice,
                                     col='sepalw',
                                     vartype='Continuous',
                                     groupby='Type')

        # run transform function
        errors = WB.cat_df.groupby(['Type']).apply(cont_slice_partial)

        self.assertIn('errNeg', errors.columns,
                      msg='errNeg not in errordf after transform_function. Only cols present: ' \
                          '{}'.format(errors.columns))

    def test_featuredict_subset(self):
        # create the featuredict
        featuredict = {'Type': 'type',
                       'target': 'target',
                       'sepall': 'sepall'}

        WB = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df,
                           featuredict = featuredict)

        self.assertListEqual(featuredict.keys(), WB.cat_df.columns.values.tolist(),
                             'Featuredict and WB instance columns dont match.'\
                             '\nFeaturedict: {}'\
                             '\nWB Instance cat df: {}'.format(featuredict.keys(), WB.cat_df.columns.values.tolist()))

    def test_whitebox_no_featuredict(self):
        WB = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df,
                           featuredict=None)

        self.assertEqual(self.iris.shape[1], len(WB.featuredict.keys()),
                         'When featuredict is not present, featuredict is not being '\
                         'populated correctly with dataframe columns.'\
                         '\nDataframe Columns: {}'\
                         '\nFeaturedict Keys: {}'.format(self.iris.columns,
                                                         WB.featuredict.keys()))