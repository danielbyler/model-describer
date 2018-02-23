#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from whitebox.utils import formatting
    from whitebox.utils import utils as wb_utils
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
                         ['Type', 'ErrType', 'Data'],
                         """json_out keys not expected for html_type accuracy""")

    def test_format_FmtJson_to_json_html_type_sensitivity(self):
        """ format json output for html_type=sensitivity """

        json_out = formatting.FmtJson.to_json(self.df,
                                              html_type='sensitivity')

        self.assertEqual(list(json_out.keys()),
                         ['Type', 'Change', 'Data'],
                         """json_out keys not expected for html_type sensitivity""")

df = pd.DataFrame({'col1': np.random.rand(1000),
                           'col2': np.random.rand(1000),
                           'col3': ['a'] * 500 + ['b'] * 500})

test = [{'Type': 'Continuous', 'Data': [{'fixed.acid': 1, 'fixed.acid': 2, 'fixed.acid': 3}]},
        {'Type': 'Continuous', 'Data': ['fixed.acid': 1, 'fixed.acid': 2, 'fixed.acid': 3]}







