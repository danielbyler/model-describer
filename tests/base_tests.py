import unittest
from sklearn import datasets
import pandas as pd
import numpy as np
import warnings
from utils.utils import getVectors, to_json, convert_categorical_independent

class TestWhiteBox(unittest.TestCase):

    def setUp(self):
        # load iris data for testing WhiteBox functionality
        iris_data = datasets.load_iris()
        self.iris = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']],
                            columns=iris_data['feature_names'] + ['target'])

    def test_getVectors_shape(self):
        # test final output of getVectors being lenght 100
        getVectors_results = getVectors(self.iris)
        self.assertEqual(getVectors_results.shape[0], 100, 'Final shape of getVectors dataframe'\
                                                             'is not 100 percentiles. Current shape: {}'.format(getVectors_results.shape[0])
                         )

    def test_getVectors_col_shape(self):
        # test final output of getVectors widths (columns) match original input
        getVectors_results = getVectors(self.iris)
        self.assertEqual(getVectors_results.shape[1], self.iris.shape[1],
                         'Final shape of getVectors return columns does not match orig. input'\
                         'Original shape: {}'\
                         'Final shape: {}'.format(self.iris.shape,
                                                  getVectors_results.shape))
    #todo Add unit tests for testing the HTML code for class string and len of string
    '''
    def test_wbox_html(self):
        # test wbox html is string
        self.assertIsInstance(HTML.wbox_html, str,
                              'Wbox HTML class is not string. Current type: {}'.format(type(HTML.wbox_html)))

    def test_wbox_html_len(self):
        # test wbox html string length
        self.assertGreater(len(HTML.wbox_html),
                           100, 'check length of HTML string. Current length: {}'.format(len(HTML.wbox_html)))
    '''
    def test_to_json(self):
        # test final output of to_json is class dict
        json = to_json(self.iris, vartype = 'Continuous')
        self.assertIsInstance(json, dict,
                              'to_json not returning dict, cur class: {}'.format(type(json)))

    def test_to_json_var(self):
        # test that the users var type is inserted into json
        json = to_json(self.iris, vartype = 'Continuous')
        self.assertEqual(json['Type'], 'Continuous',
                         'Vartype incorrect, current vartype is {}'.format(json['Type']))

    def test_convert_categorical_independent(self):
        # test the conversion of pandas categorical datatypes to integers
        # create simulated dataframe
        df = pd.DataFrame({'category1': list('abcabcabcabc'),
                           'category2': list('defdefdefdef')})

        df['category1'] = pd.Categorical(df['category1'])
        df['category2'] = pd.Categorical(df['category2'])

        num_df = convert_categorical_independent(df)

        self.assertEqual(num_df.select_dtypes(include = [np.number]).shape[1],
                         df.shape[1], 'Numeric column shapes mismatched: Original: {}' \
                                      'Transformed: {}'.format(df.shape, num_df.shape))


    def test_convert_categorical_independent_warnings(self):
        # create only numeric dataframe
        df = pd.DataFrame({'col1': list(range(100)),
                           'col2': list(range(100))})
        warn_message = ''
        # capture warnings messages
        with warnings.catch_warnings(record=True) as w:
            df2 = convert_categorical_independent(df)
            warn_message += str(w[-1].message)

        self.assertEqual(warn_message, 'Pandas categorical variable types not detected',
                         'Categorical warning not displayed with all number dataframe')






