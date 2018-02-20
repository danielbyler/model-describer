#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='whitebox',
    version='0.0.7.7',
    packages=['whitebox'],
    url='https://github.com/Data4Gov/WhiteBox_Production',
    license='MIT',
    author='Jason Lewris, Daniel Byler, Venkat Gangavarapu, Shruti Panda',
    author_email='jlewris@deloitte.com',
    description="""How can I unlock what my model is thinking? WhiteBox helps answer this problem for sklearn machine learning models. 
    Specifically, WhiteBox addresses two key issues: error and model sensitivity. For error, WhiteBox analyzes how well the model
    is performing within key regions of the data in a visually compelling way. For sensitivity, WhiteBox identifies what parts of 
    different variable distributions have the biggest impact on model predictions and plots them. All WhiteBox output is created with 
    simple code and produces HTML files that are small enough to be emailed.""",
    download_url='https://github.com/data4gov/WhiteBox_Production/archive/v0.0.6.tar.gz',
    keywords=['whitebox', 'machine learning', 'data science'],
    package_data={'whitebox': ['html_error.txt', 'html_sensitivity.txt']},
)
