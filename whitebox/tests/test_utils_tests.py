#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from sklearn import datasets
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from whitebox import utils
except:
    from .whitebox import utils
finally:
    from whitebox import utils

__author__ = "Jason Lewris, Daniel Byler, Venkat Gangavarapu, Shruti Panda, Shanti Jha"
__credits__ = ["Brian Ray"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jason Lewris"
__email__ = "jlewris@deloitte.com"
__status__ = "Beta"


class TestUtils(unittest.TestCase):

    def setUp(self):
        # load iris data for testing WhiteBox functionality
        iris_data = datasets.load_iris()
        self.iris = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                                 columns=['sepall', 'sepalw', 'petall', 'petalw', 'target'])

    def test_getVectors_shape(self):
        # test final output of getVectors being length 100
        getvectors_results = utils.getvectors(self.iris)
        self.assertEqual(getvectors_results.shape[0], 100,
                         """Final shape of getVectors dataframe
                            is not 100 percentiles. Current shape: {}""".format(getvectors_results.shape[0])
                         )

    def test_getVectors_col_shape(self):
        # test final output of getVectors widths (columns) match original input
        getvectors_results = utils.getvectors(self.iris)
        self.assertEqual(getvectors_results.shape[1], self.iris.shape[1],
                         """Final shape of getVectors return columns 
                                does not match orig. input
                                Original shape: {}
                                Final shape: {}""".format(
                                                            self.iris.shape,
                                                            getvectors_results.shape))

    def test_wbox_html_error(self):
        # test wbox html is string for html_error
        html_error = utils.HTML.get_html(htmltype='html_error')
        self.assertIsInstance(html_error, str,
                              'Wbox HTML error class is not string. Current type: {}'.format(type(html_error)))

    def test_wbox_html_len_error(self):
        # test wbox html string length -- html_error
        html_error = utils.HTML.get_html(htmltype='html_error')
        self.assertGreater(len(html_error),
                           100, 'check length of HTML error string. Current length: {}'.format(len(html_error)))

    def test_wbox_html_sensitivity(self):
        # test wbox html is string for html_sensitivity
        html_sensitivity = utils.HTML.get_html(htmltype='html_sensitivity')
        self.assertIsInstance(html_sensitivity, str,
                              'Wbox HTML sensitivity class is not string. Current type: {}'.format(type(html_sensitivity)))

    def test_wbox_html_len_sensitivity(self):
        # test wbox html string length -- html_sensitivity
        html_sensitivity = utils.HTML.get_html(htmltype='html_sensitivity')
        self.assertGreater(
                            len(html_sensitivity),
                            100,
                            """check length of HTML sensitivity string. 
                            Current length: {}""".format(len(html_sensitivity)))

    def test_to_json(self):
        # test final output of to_json is class dict
        json = utils.to_json(self.iris, vartype='Continuous')
        self.assertIsInstance(json, dict,
                              'to_json not returning dict, cur class: {}'.format(type(json)))

    def test_to_json_var(self):
        # test that the users var type is inserted into json
        json = utils.to_json(self.iris, vartype='Continuous')
        self.assertEqual(json['Type'], 'Continuous',
                         'Vartype incorrect, current vartype is {}'.format(json['Type']))

    def test_convert_categorical_independent(self):
        # test the conversion of pandas categorical datatypes to integers
        # create simulated dataframe
        df = pd.DataFrame({'category1': list('abcabcabcabc'),
                           'category2': list('defdefdefdef')})

        df['category1'] = pd.Categorical(df['category1'])
        df['category2'] = pd.Categorical(df['category2'])

        num_df = utils.convert_categorical_independent(df)

        self.assertEqual(num_df.select_dtypes(include=[np.number]).shape[1],
                         df.shape[1],
                         """Numeric column shapes mismatched: Original: {}' \
                            Transformed: {}""".format(df.shape, num_df.shape))

    def test_convert_categorical_independent_warnings(self):
        # create only numeric dataframe
        df = pd.DataFrame({'col1': list(range(100)),
                           'col2': list(range(100))})
        warn_message = ''
        # capture warnings messages
        with warnings.catch_warnings(record=True) as w:
            utils.convert_categorical_independent(df)
            warn_message += str(w[-1].message)

        self.assertEqual(warn_message, 'Pandas categorical variable types not detected',
                         'Categorical warning not displayed with all number dataframe')

    def test_create_insights_mse(self):
        # create sample actual/preds data
        df = pd.DataFrame({'actual': np.random.rand(100),
                           'predicted': np.random.rand(100)})
        df['errors'] = df['actual'] - df['predicted']
        # set dummy name
        df.__setattr__('name', 'test')
        # capture MSE from create_insights
        msedf = utils.create_insights(df, group_var='test',
                                      error_type='MSE')
        mse = msedf['MSE'].values[0]
        # sklearn mse
        sklearn_mse = mean_squared_error(df['actual'],
                                         df['predicted'])

        self.assertEqual(round(mse, 4), round(sklearn_mse, 4),
                         msg="""MSE error miscalc.
                                \ncreate_insights MSE: {}
                                \nsklearn_mse: {}""".format(mse, sklearn_mse))

    def test_create_insights_mae(self):
        # create sample actual/preds data
        df = pd.DataFrame({'actual': np.random.rand(100),
                           'predicted': np.random.rand(100)})
        df['errors'] = df['predicted'] - df['actual']
        # set dummy name
        df.__setattr__('name', 'test')
        # capture MSE from create_insights
        maedf = utils.create_insights(df, group_var='test',
                                      error_type='MAE')
        mae = maedf['MAE'].values[0]
        # sklearn mse
        sklearn_mae = mean_absolute_error(
                                            df['actual'],
                                            df['predicted'])

        self.assertEqual(round(mae, 4), round(sklearn_mae, 4),
                         msg="""MAE error miscalc.
                                \ncreate_insights MAE: {}
                                \nsklearn_mse: {}""".format(mae, sklearn_mae))

    def test_create_html_error(self):
        # set up sample dependent variable
        ydepend = 'TESTVARIABLE'
        datastring = "{'Type': 'Categorical', 'Data': [1, 2, 3]}"
        output = utils.createmlerror_html(datastring, ydepend)

        self.assertIn(ydepend, output,
                      msg="""Dependent variable ({}) not found in final output
                                datastring""".format(ydepend))

    def test_flatten_json(self):
        # test the flattening of flatten_json utility function
        test_data = [{
                        'Type': 'Continuous',
                        'Data': [{
                                    'val1': 1,
                                    'val2': 2},
                                 {
                                    'val1': 1,
                                    'val2': 2}]},
                    {
                        'Type': 'Continuous',
                        'Data': [{
                                    'val1': 1,
                                    'val2': 2},
                                 {
                                    'val1': 1,
                                    'val2': 2}]},
                    {
                        'Type': 'Continuous',
                        'Data': [{
                                    'val1': 1,
                                    'val2': 2},
                                 {
                                    'val1': 1,
                                    'val2': 2}]}]

        flat = utils.flatten_json(test_data)

        self.assertEqual(len(flat), 2,
                         msg="""Flatten json not flattening in 
                                expected format. Took list of length 3 and retured 
                                object without length of 2.
                                \nReturned length: {}
                                \nReturned type: {}""".format(len(flat), type(flat)))

    def test_flatten_json_return_type(self):
        # test the type of the returned object from flatten_json
        test_data = [{'Type': 'Continuous', 'Data': [{'val1': 1, 'val2': 2}, {'val1': 1, 'val2': 2}]},
                     {'Type': 'Continuous', 'Data': [{'val1': 1, 'val2': 2}, {'val1': 1, 'val2': 2}]},
                     {'Type': 'Continuous', 'Data': [{'val1': 1, 'val2': 2}, {'val1': 1, 'val2': 2}]}]

        flat = utils.flatten_json(test_data)

        self.assertIsInstance(flat, dict,
                              msg="""Returned object from flatten_json 
                                    not dict object.
                                    \nReturn class: {}""".format(type(flat)))


if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
