#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

def readme():
    with open('PYPIREADME.rst') as f:
        return f.read()

setup(
    name='model-describer',
    version='0.1.0',
    packages=find_packages("."),
    url='https://github.com/DataScienceSquad/model-describer',
    license='MIT',
    author='Jason Lewris, Venkat Gangavarapu, Daniel Byler,  Shruti Panda',
    author_email="jlewris@deloitte.com, vegangavarapu@deloitte.com, dybler@deloitte.com, shrupanda@deloitte.com",
    description="""How can I unlock what my model is thinking? model-describer helps answer this problem for sklearn machine learning models. Specifically, model-describer helps address two key issues: error and model sensitivity. For error, model-describer analyzes how well the model is performing within key regions of the data in a visually compelling way. For sensitivity, model-describer identifies what parts of different variable distributions have the biggest impact on model predictions and plots them.""",
    download_url='https://github.com/DataScienceSquad/model-describer/archive/0.1.0.tar.gz',
    long_description=readme(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Communications :: Email',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=['machine learning',
              'data science',
              'sklearn',
              'scikit-learn',
              'machine learning interpretability',
              'visualization',
              ],
    package_data={'mdesc': ['html_error.txt', 'html_sensitivity.txt']},
    test_suite='nose.collector',
    tests_require=['nose'],
)
