#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pkg_resources
import warnings

import pandas as pd

try:
    import utils.utils as wb_utils
except ImportError:
    import whitebox.utils.utils as wb_utils


def autoformat_types(inputdf):
    """
    cast dtype category to strings

    :param inputdf: dataframe input
    :return: dataframe output categories casted as strings
    :rtype: pd.DataFrame
    """
    # convert categorical dtypes to strings
    catcols = inputdf.select_dtypes(include=['category']).columns
    inputdf[catcols] = inputdf[catcols].apply(lambda x: x.astype(str))
    return inputdf


def format_inputs(input_val,
                  format_dict,
                  subset=False):
    """
    format input by format_dict key, vals

    :param input_val: input to format - supported types:
        list, str, pd.DataFrame
    :param format_dict: dict with key=original, value=formatted name
    :param subset: boolean, subset dataframe on format_dict.keys
    :return: formatted output same type as input
    :rtype: str|list|pd.DataFrame
    """
    # format string
    if isinstance(input_val, str):
        output = format_dict.get(input_val, input)
    # format pandas dataframe
    elif isinstance(input_val, pd.DataFrame):
        if subset is False:
            output = input_val.rename(columns=format_dict)
        else:
            output = input_val.rename(columns=format_dict).loc[:, list(format_dict.values())]
    # format list
    elif isinstance(input_val, list):
        output = [format_dict.get(list_val, list_val) for list_val in input_val]

    else:
        raise TypeError("""format_inputs received 
                        unexpected type: {}""".format(type(input_val)))

    return output


class FmtJson(object):
    """ utility class to house json formatting functionality """

    @staticmethod
    def to_json(
            dataframe,
            vartype='Continuous',
            html_type='error',
            incremental_val=None,
            err_type=None):
        """
        convert input dataframe to json

        :param dataframe: input dataframe
        :param vartype: variable type for conversion (Continuous, Categorical, Accuracy, Percentile)
        :param html_type: html output type (error, sensitivity, percentile, accuracy)
        :param incremental_val: If sensitivity used, include incremental value
            used to construct synthetic data
        :param err_type: User defined error type
        :return: formatted json output
        :rtype: dict
        """
        # convert dataframe values into a json like object for D3 consumption
        assert vartype in ['Continuous', 'Categorical', 'Accuracy', 'Percentile'], \
            """Vartypes should only be continuous, categorical,
            Percentile or accuracy"""

        assert html_type in ['error', 'sensitivity', 'percentile', 'accuracy'], \
            'html_type must be error, sensitivity, percentile, accuracy'

        json_dict = dict(percentile={'Type': vartype},
                         error={'Type': vartype},
                         accuracy={'Type': vartype,
                                   'ErrType': err_type},
                         sensitivity={'Type': vartype,
                                      'Change': str(incremental_val)})

        json_out = json_dict[html_type]
        # create data records from values in df
        json_out['Data'] = dataframe.to_dict(orient='records')

        return json_out

    @staticmethod
    def flatten_json(dictlist):
        """
        flatten lists of dictionaries of the same variable into one dict
            structure. Inputs: [{'Type': 'Continuous', 'Data': [fixed.acid: 1, ...]},
            {'Type': 'Continuous', 'Data': [fixed.acid : 2, ...]}]
            outputs: {'Type' : 'Continuous', 'Data' : [fixed.acid: 1, fixed.acid: 2]}}

        :param dictlist: list of dictionaries
        :return: flattened structure with column variable as key
        :rtype: dict
        """
        # make copy of dictlist
        copydict = dictlist[:]
        if len(copydict) > 1:
            for val in copydict[1:]:
                copydict[0]['Data'].extend(val['Data'])
            # take the revised first element of the list
            toreturn = copydict[0]
        else:
            if isinstance(copydict, list):
                # return the dictionary object if list type
                toreturn = copydict[0]
            else:
                # else return the dictionary itself
                toreturn = copydict
        assert isinstance(toreturn, dict), """flatten_json output object not of class dict.
                                            \nOutput class type: {}""".format(type(toreturn))
        return toreturn


class HTML(object):

    @staticmethod
    def get_html(htmltype='html_error'):
        assert htmltype in ['html_error', 'html_sensitivity'], 'htmltype must be html_error or html_sensitivity'
        html_path = pkg_resources.resource_filename('whitebox', '{}.txt'.format(htmltype))
        # utility class to hold whitebox files
        try:
            wbox_html = open('{}.txt'.format(htmltype), 'r').read()
        except IOError:
            wbox_html = open(html_path, 'r').read()
        return wbox_html

    @staticmethod
    def fmt_html_out(
            datastring,
            dependentvar,
            htmltype='html_error'):
        """
        create WhiteBox error plot html code
        :param datastring: json like object containing data
        :param dependentvar: name of dependent variable
        :return: html string
        """
        assert htmltype in ['html_error', 'html_sensitivity'], """htmltype must be html_error 
                                                                    or html_sensitivity"""
        output = HTML.get_html(htmltype=htmltype).replace('<***>',
                                                          datastring
                                                          ).replace('Quality', dependentvar)
        return output



def convert_categorical_independent(dataframe):
    """
    convert pandas dtypes 'categorical' into numerical columns
    :param dataframe: dataframe to perform adjustment on
    :return: dataframe that has converted strings to numbers
    """
    # we want to change the data, not copy and change
    dataframe = dataframe.copy(deep=True)
    # convert all strings to categories and format codes
    for str_col in dataframe.select_dtypes(include=['O', 'category']):
        dataframe.loc[:, str_col] =pd.Categorical(dataframe.loc[:, str_col])
    # convert all category datatypes into numeric
    cats = dataframe.select_dtypes(include=['category'])
    # warn user if no categorical variables detected
    if cats.shape[1] == 0:
        warnings.warn('Pandas categorical variable types not detected', UserWarning)
    # iterate over these columns
    for category in cats.columns:
        dataframe.loc[:, category] = dataframe.loc[:, category].cat.codes

    return dataframe