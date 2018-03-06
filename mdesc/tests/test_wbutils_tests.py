#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
import numpy as np

from mdesc.utils import utils


class TestWhiteBoxError(unittest.TestCase):

    def test_create_insights_mse(self):
        """test MSE calculation from utils.create_insights"""

        df = pd.DataFrame({'errors': list(range(100))})

        setattr(df, 'name', 'testname')

        insights_out = utils.create_insights(df, group_var='groupvar',
                                             error_type='MSE')

        mse_act = np.mean(df['errors'] ** 2)

        self.assertEqual(insights_out['MSE'].values[0], mse_act,
                         msg="""utils.create_insights returning incorrect results for MSE error
                                \nActual MSE: {}
                                \nRetuend MSE: {}""".format(mse_act, insights_out['MSE']))

    def test_create_insights_med(self):
        """test create_insights MED calc"""
        df = pd.DataFrame({'errors': list(range(100))})

        actual = np.median(df['errors'].values.tolist())

        setattr(df, 'name', 'testname')

        results = utils.create_insights(df,
                                        group_var='groupvar',
                                        error_type='MED')

        self.assertEqual(actual,
                         results['MED'].values[0],
                         """unexpected calculation create_insights MED param""")

    def test_create_insights_mean(self):
        """test create_insights MEAN calc"""
        df = pd.DataFrame({'errors': list(range(100))})

        actual = np.median(df['errors'].values.tolist())

        setattr(df, 'name', 'testname')

        results = utils.create_insights(df,
                                        group_var='groupvar',
                                        error_type='MEAN')

        self.assertEqual(actual,
                         results['MEAN'].values[0],
                         """unexpected calculation create_insights MEAN param""")

    def test_create_insights_output_shape(self):
        """test utils.create_insights output shape"""

        df = pd.DataFrame({'errors': list(range(100))})

        setattr(df, 'name', 'testname')

        insights_out = utils.create_insights(df, group_var='groupvar',
                                             error_type='MSE')

        self.assertEqual(insights_out.shape, (1, 4),
                         msg="""utils.create_insights did not return shape (1,4)
                         \nReturned Shape: {}""".format(insights_out.shape))

    def test_create_insights_raise_keyerror(self):
        """test utils.create_insights raises KeyError when errors not present in df"""

        df = pd.DataFrame({'random': list(range(100))})

        setattr(df, 'name', 'testname')

        with self.assertRaises(KeyError) as context:
            utils.create_insights(df, group_var='groupvar',
                                  error_type='MSE')

        self.assertIn('errors',
                      str(context.exception),
                      """utils.create_insights not raising keyerror 
                      when errors not present in df {}""".format(context.exception))

    def test_prob_acc_outputs(self):
        """test utils.prob_acc accuracy calculation output"""
        true_class = 1
        pred_prob = 0.8

        actual_result = (true_class * (1-pred_prob)) + ((1-true_class)*pred_prob)

        prob_acc_output = utils.prob_acc(true_class=true_class,
                                         pred_prob=pred_prob)

        self.assertEqual(actual_result,
                         prob_acc_output,
                         """utils.prob_acc return inccorect results
                         \nExpected: {}
                         \nActual: {}""".format(actual_result, prob_acc_output))

    def test_create_accuracy_output_shape(self):
        """test utils.create_accuracy output"""

        setup = pd.DataFrame({'errors': np.random.rand(100),
                              'col2': ['a'] * 50 + ['b'] * 50})

        output = utils.create_accuracy('regression',
                                       setup,
                                       error_type='MSE',
                                       groupby='col2')

        self.assertEqual(output.shape, (2, 4),
                         """shape does not equal (2, 4) -- returned shape: {}""".format(output.shape))


if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()