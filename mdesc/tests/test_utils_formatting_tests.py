#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from mdesc.utils import formatting
    from mdesc.utils import utils as wb_utils
except ImportError:
    from utils import formatting
    from utils import utils as wb_utils


class TestWhiteBoxError(unittest.TestCase):
    """ test percentiles functionality and outputs """

    def setUp(self):
        """ construct shared dataset for testing """

        self.df = pd.DataFrame({'col1': np.random.rand(1000),
                           'col2': np.random.rand(1000),
                           'col3': ['a'] * 500 + ['b'] * 500})

    def test_autoformat_conversion(self):
        df = self.df.copy(deep=True)
        df.loc[:, 'col3'] = pd.Categorical(df.loc[:, 'col3'])
        format_df = formatting.autoformat_types(df)

        cat_cols = format_df.select_dtypes(include=['category'])

        self.assertEqual(cat_cols.shape[1],
                         0,
                         """autoformat didn't convert all categories to object types""")

    def test_format_inputs_str_input(self):
        """ test format_inputs conversion of string """

        formatdict = {'test': 'finalout'}
        input_str = 'test'
        formatted = formatting.format_inputs(input_str,
                                             formatdict)

        self.assertEqual(formatted,
                         'finalout',
                         """formatting.format_inputs did not map string correctly""")

    def test_format_inputs_list_input(self):
        """ test format_inputs conversion of list """

        formatdict = {'test': 'finalout',
                      'test2': 'finalout2',
                      'test3': 'finalout3'}
        input_list = ['test', 'test2', 'test3']
        formatted = formatting.format_inputs(input_list,
                                             formatdict)

        self.assertEqual(formatted,
                         ['finalout', 'finalout2', 'finalout3'],
                         """formatting.format_inputs did not map list correctly""")

    def test_format_inputs_dataframe_input(self):
        """ test format_inputs conversion of dataframe """

        formatdict = {'test': 'finalout',
                      'test2': 'finalout2',
                      'test3': 'finalout3'}

        input_df = pd.DataFrame({'test': np.random.rand(100),
                                 'test2': np.random.rand(100),
                                 'test3': np.random.rand(100)})

        formatted = formatting.format_inputs(input_df,
                                             formatdict)

        self.assertEqual(formatted.columns.tolist(),
                         ['finalout', 'finalout2', 'finalout3'],
                         """formatting.format_inputs did not map dataframe correctly""")

    def test_format_inputs_dataframe_subset(self):
        """ test format_inputs conversion and subset of dataframe """

        formatdict = {'test': 'finalout',
                      'test2': 'finalout2'}

        input_df = pd.DataFrame({'test': np.random.rand(100),
                                 'test2': np.random.rand(100),
                                 'test3': np.random.rand(100)})

        formatted = formatting.format_inputs(input_df,
                                             formatdict,
                                             subset=True)

        self.assertEqual(formatted.columns.tolist(),
                         ['finalout', 'finalout2'],
                         """formatting.format_inputs did not subset and map dataframe correctly""")

    def test_format_FmtJson_to_json_html_type_error(self):
        """ format json output for html_type=error """

        json_out = formatting.FmtJson.to_json(self.df,
                                              html_type='error')

        self.assertEqual(list(json_out.keys()),
                         ['Type', 'Data'],
                         """json_out keys not expected for html_type error""")

    def test_format_FmtJson_to_json_html_type_percentiles(self):
        """ format json output for html_type=percentiles """

        json_out = formatting.FmtJson.to_json(self.df,
                                              html_type='percentile')

        self.assertEqual(list(json_out.keys()),
                         ['Type', 'Data'],
                         """json_out keys not expected for html_type percentiles""")

    def test_format_FmtJson_to_json_html_type_accuracy(self):
        """ format json output for html_type=accuracy """

        json_out = formatting.FmtJson.to_json(self.df,
                                              html_type='accuracy')

        self.assertEqual(list(json_out.keys()),
                         ['Type', 'ErrType', 'Yvar', 'Data'],
                         """json_out keys not expected for html_type accuracy""")

    def test_format_FmtJson_to_json_html_type_sensitivity(self):
        """ format json output for html_type=sensitivity """

        json_out = formatting.FmtJson.to_json(self.df,
                                              html_type='sensitivity')

        self.assertEqual(list(json_out.keys()),
                         ['Type', 'Change', 'Data'],
                         """json_out keys not expected for html_type sensitivity""")

    def test_flatten_json_out(self):
        """ test flatten_json flattening list of dicts """
        test_data = test = [{'Type': 'Categorical', 'Data': [{'FIXED ACIDITY_test': 'low', 'errNeg': -1.3387746358183215, 'errPos': 0.47370517928286787, 'groupByValue': 'White', 'groupByVarName': 'TYPE_test', 'predictedYSmooth': 4.626276031033032}, {'FIXED ACIDITY_test': 'low', 'errNeg': -1.475259740259743, 'errPos': 0.35873015873015873, 'groupByValue': 'Red', 'groupByVarName': 'TYPE_test', 'predictedYSmooth': 4.22933083176985}]},
                            {'Type': 'Categorical', 'Data': [{'VOLATILE ACIDITY BINS_test': 'bin_1', 'errNeg': -1.3387746358183215, 'errPos': 0.47370517928286787, 'groupByValue': 'White', 'groupByVarName': 'TYPE_test', 'predictedYSmooth': 4.626276031033032}, {'VOLATILE ACIDITY BINS_test': 'bin_0', 'errNeg': -1.475259740259743, 'errPos': 0.35873015873015873, 'groupByValue': 'Red', 'groupByVarName': 'TYPE_test', 'predictedYSmooth': 4.22933083176985}]},
                            {'Type': 'Categorical', 'Data': [{'AC_test': 'low', 'errNeg': -1.3387746358183215, 'errPos': 0.47370517928286787, 'groupByValue': 'White', 'groupByVarName': 'TYPE_test', 'predictedYSmooth': 4.626276031033032}, {'AC_test': 'low', 'errNeg': -1.475259740259743, 'errPos': 0.35873015873015873, 'groupByValue': 'Red', 'groupByVarName': 'TYPE_test', 'predictedYSmooth': 4.22933083176985}]}]

        test_flat = formatting.FmtJson.flatten_json(test_data)
        self.assertEqual(len(test_flat),
                         2,
                         """flatten_json returned unexpected number of keys - check outputs""")

    def test_get_html_error(self):
        """ test HTML.get_html method to retrieve correct HTML file for erro r"""

        html_error = formatting.HTML.get_html(htmltype='html_error')

        self.assertIn('Prediction Error',
                      html_error,
                      """Prediction Error not located in html that was loaded for html_error""")

    def test_get_html_sensitivity(self):
        """ test HTML.get_html method to retrieve correct HTML file for sensitivity """

        html_sensitivity = formatting.HTML.get_html(htmltype='html_sensitivity')

        self.assertIn('Impact By Variable',
                      html_sensitivity,
                      """Impact By Variable not located in html that was loaded for html_sensitivity""")

    def test_insert_data_html_error(self):
        """ test HTML.get_html method to retrieve correct HTML file for erro r"""

        testdatastring = '**TEST**'
        html_error = formatting.HTML.fmt_html_out(testdatastring,
                                                  'yDepend',
                                                  htmltype='html_error')
        self.assertIn('**TEST**',
                      html_error,
                      """TEST data insert not located in html that was created for html_error""")

    def test_insert_data_html_sensitivity(self):
        """ test HTML.get_html method to retrieve correct HTML file for sensitivity """

        testdatastring = '**TEST**'
        html_sensitivity = formatting.HTML.fmt_html_out(testdatastring,
                                                  'yDepend',
                                                  htmltype='html_error')
        self.assertIn('**TEST**',
                      html_sensitivity,
                      """TEST data insert not located in html that was created for html_sensitivity""")


if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()