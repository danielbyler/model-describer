#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np

try:
    import sys
    sys.path.insert(0, "/home/travis/build/Data4Gov/WhiteBox_Production")
    from mdesc.utils import percentiles
    from mdesc.utils import utils as wb_utils
except ImportError:
    from utils import percentiles
    from utils import utils as wb_utils


class TestWhiteBoxError(unittest.TestCase):
    """ test percentiles functionality and outputs """

    def setUp(self):
        """ construct shared dataset for testing """

        self.df = pd.DataFrame({'col1': np.random.rand(1000),
                           'col2': np.random.rand(1000),
                           'col3': ['a'] * 500 + ['b'] * 500})

        self.results = percentiles.create_group_percentiles(self.df,
                                     groupbyvars=['col3'],
                                     )

        self.Percentiles = percentiles.Percentiles(self.df,
                                                   groupbyvars=['col3'])

    def test_create_group_percentiles_output_variables(self):
        """ test create_group_percentiles output vars """

        seen_vars = []
        # iterate over data elements and pull variables
        for data_item in self.results['Data']:
            # append
            seen_vars.append(data_item['variable'])

        # ensure difference between seen_vars and expected is none
        diff = list(set(seen_vars).difference(set(['col1', 'col2'])))

        self.assertEqual(len(diff),
                         0,
                         """create_group_percentiles returning unexpected variables""")

    def test_create_group_percentiles_output_type(self):
        """ test create_group_percentiles output type is PercentileGroup """

        self.assertEqual(self.results['Type'],
                         'PercentileGroup',
                         """create_group_percentiles returned 
                         enexpected output type: {}""".format(self.results['Type']))

    def test_create_group_percentiles_output_groupbyvars(self):
        """ test create_group_percentiles output groupbyvars matches input levels """

        all_groupbys = []
        for data_item in self.results['Data']:
            for percent in data_item['percentileList']:
                all_groupbys.append(percent['groupByVar'])

        diff = list(set(all_groupbys).difference(set(self.df['col3'].unique())))

        self.assertEqual(len(diff),
                         0,
                         """create_group_percentiles output groupby levels
                         differs from input""")

    def test_create_group_percentiles_output_buckets(self):
        """ ensure create_group_percentiles output buckets are correct """

        created_percentiles = []

        desired_percentiles = [str(percent) + '%' for percent in wb_utils.Settings.formatted_percentiles]

        for data_item in self.results['Data']:
            for percent in data_item['percentileList']:
                percentile_keys = [perval['percentiles'] for perval in percent['percentileValues']]
                created_percentiles.append(percentile_keys)

        self.assertTrue([desired_percentiles == l for l in created_percentiles],
                        """create_group_percentiles returned unexecpted percentile buckets
                        returned: {}""".format(created_percentiles))

    def test_create_percentile_vecs_series_output_shape(self):
        """ test create_percentile_vecs output shape for series input """

        series_results = percentiles.create_percentile_vecs(self.df['col1'])

        self.assertEqual(series_results.shape[0],
                         100,
                         """create_percentile_vecs output shape did not match 100
                         \nOutput shape: {}""".format(series_results.shape[0]))

    def test_create_percentile_vecs_dataframe_output_shape(self):
        """ test create_percentile_vecs output shape for dataframe input """

        dataframe_results = percentiles.create_percentile_vecs(self.df)

        self.assertEqual(dataframe_results.shape,
                         (100, 2),
                         """create_percentile_vecs returned output shape inconsistent with (100, 2)
                         \nReturned shape: {}""".format(dataframe_results.shape))

    def test_Percentiles_population_percentile_creation(self):
        """ test attribute population_percentile_vecs assignment to Percentiles """

        self.Percentiles.population_percentiles()
        self.assertTrue(hasattr(self.Percentiles, 'population_percentile_vecs'),
                        """population_percentile_vecs not attribute after population_percentiles run""")

    def test_Percentiles_group_percentiles_out_creation(self):
        """ test attribute group_percentiles_out assignment to Percentiles """

        self.Percentiles.population_percentiles()
        self.assertTrue(hasattr(self.Percentiles, 'group_percentiles_out'),
                        """population_percentile_vecs not attribute after population_percentiles run""")

    def test_Percentiles_percentiles_creation(self):
        """ test attribute percentiles assignment to Percentiles """

        self.Percentiles.population_percentiles()
        self.assertTrue(hasattr(self.Percentiles, 'percentiles'),
                        """percentiles not attribute after population_percentiles run""")

    def test_Percentiles_percentiles_output_type(self):
        """ test attribute percentiles is dict type after run """

        self.Percentiles.population_percentiles()
        self.assertEqual(type(self.Percentiles.percentiles),
                              dict,
                              """Percentiles.percentiles not of dict type.
                              \nreturned type: {}""".format(type(self.Percentiles.percentiles)))


if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
