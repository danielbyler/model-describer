#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from mdesc.utils import categorical_conversions

except ImportError:
    from utils import categorical_conversions


class TestWhiteBoxError(unittest.TestCase):

    def setUp(self):
        df = pd.DataFrame({'col1': np.random.rand(1000),
                           'col2': np.random.rand(1000),
                           'col3': ['a'] * 600 + ['b'] * 400})

        df_dummies = pd.get_dummies(df)

        self.df = df
        self.df_dummies = df_dummies

    def test_cat_conversion_pandas_switch_modal_dummy_df_output(self):
        """test pandas_switch_modal_dummy output values in non modal columns"""

        modal_val, res, _ = categorical_conversions.pandas_switch_modal_dummy('col3',
                                                                           self.df,
                                                                           self.df_dummies)

        self.assertTrue(all(res['col3_b'] == 0),
                      """pandas_switch_modal_dummy did not convert non modal columns to 0""")

    def test_cat_conversion_pandas_switch_modal_dummy_df_modal_val(self):
        """test pandas_switch_modal_dummy output modal values"""

        modal_val, res, _ = categorical_conversions.pandas_switch_modal_dummy('col3',
                                                                           self.df,
                                                                           self.df_dummies)

        self.assertEqual(modal_val,
                         'a',
                         """pandas_switch_modal_dummy did not output correct modal value""")


if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()