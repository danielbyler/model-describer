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

    def test_wberror_notfitted(self):
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
        # set up randomforestregressor
        modelobj = RandomForestRegressor()

        modelobj.fit(self.iris.loc[:, self.iris.columns != 'target'],
                    self.iris['target'])
        # test whether outputs are assigned to instance after run
        WB = WhiteBoxError(modelobj = modelobj,
                      model_df = self.iris,
                      ydepend = 'target',
                      groupbyvars = ['sepalw'])


        WB.run()

        self.assertIsInstance(WB.outputs, list,
                              msg = 'WhiteBoxError is not producing list of outputs after run')

