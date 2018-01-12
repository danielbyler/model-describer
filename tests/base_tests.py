import unittest
from sklearn import datasets
import pandas as pd
import numpy as np
from utils.utils import HTML, getVectors, to_json

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

    def test_wbox_html(self):
        # test wbox html is string
        self.assertIsInstance(HTML.wbox_html, str,
                              'Wbox HTML class is not string. Current type: {}'.format(type(HTML.wbox_html)))

    def test_wbox_html_len(self):
        # test wbox html string length
        self.assertGreater(len(HTML.wbox_html),
                           100, 'check length of HTML string. Current length: {}'.format(len(HTML.wbox_html)))

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






