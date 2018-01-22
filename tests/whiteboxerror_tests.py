from whitebox import WhiteBoxError
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
import pandas as pd
import numpy as np
import unittest

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
        # test whether errPos column present after running whiteboxerror transform_function
        # method on slice of data
        WB = WhiteBoxError(modelobj=self.modelobj,
                           model_df=self.iris,
                           ydepend='target',
                           groupbyvars=['Type'],
                           cat_df=self.cat_df)

        WB.predict()

        # run transform function
        errors = WB.cat_df.groupby(['Type']).apply(WB.transform_function)

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

        # run transform function
        errors = WB.cat_df.groupby(['Type']).apply(WB.transform_function)

        self.assertIn('errNeg', errors.columns,
                      msg='errNeg not in errordf after transform_function. Only cols present: ' \
                          '{}'.format(errors.columns))



