import pandas as pd
import math
import numpy as np
import warnings

def getVectors(dataframe):
    """
    getVectors calculates the percentiles 1 through 100 for data in each columns of dataframe.
    For categorical data, the mode is used.
    :param dataframe: pandas dataframe object
    :return: pandas dataframe object with percentiles
    """
    # ensure dataframe is pandas dataframe object
    assert isinstance(dataframe, pd.DataFrame), 'Supports pandas dataframe, current class type is {}'.format(type(dataframe))
    # calculate the percentiles for numerical data and use the mode for categorical, string data
    allresults = dataframe.describe(percentiles=np.linspace(0.01, 1, num=100), include=[np.number])# , 'category', 'O'])

    # if top (or mode) in index
    if 'top' in allresults.index:
        # Pull out the percentiles for numerical data otherwise use the mode
        tempVec = allresults.apply(lambda x: x['top'] if math.isnan(x['mean']) else x.filter(regex='[0-9]{1,2}\%',
                                                                                             axis=0), axis=0)
    else:
        tempVec = allresults.filter(regex='[0-9]{1,2}\%', axis=0)

    return(tempVec)

def convert_categorical_independent(dataframe):
    """
    convert pandas dtypes 'categorical' into numerical columns
    :param dataframe: dataframe to perform adjustment on
    :return: dataframe that has converted strings to numbers
    """
    dataframe = dataframe.copy(deep = True) # we want to change the data, not copy and change
    # convert all category datatypes into numeric
    cats = dataframe.select_dtypes(include=['category'])
    # warn user if no categorical variables detected
    if cats.shape[1] == 0:
        warnings.warn('Pandas categorical variable types not detected', UserWarning)
    # iterate over these columns
    for category in cats.columns:
        dataframe.loc[:, category] = dataframe.loc[:, category].cat.codes

    return dataframe

def create_insights(group, group_var = None,
                    error_type = 'MSE'):
    """
    create_insights develops various error metrics such as MSE, RMSE, MAE, etc.
    :param group: the grouping object from the pandas groupby
    :param group_var: the column that is being grouped on
    :return: dataframe with error metrics
    """
    assert error_type in ['MSE', 'RMSE', 'MAE'], 'Currently only supports'\
                                                ' MAE, MSE, RMSE'
    errors = group['errors']
    error_dict = {'MSE': np.mean(errors ** 2),
                  'RMSE': (np.mean(errors ** 2)) ** (1 / 2),
                  'MAE': np.sum(np.absolute(errors))/group.shape[0]}

    msedf = pd.DataFrame({'groupByValue': group.name,
                          'groupByVarName': group_var,
                          error_type: error_dict[error_type],
                          'Total': group.shape[0]}, index = [0])
    return msedf

def to_json(dataframe, vartype='Continuous'):
    # convert dataframe values into a json like object for D3 consumption
    assert vartype in ['Continuous', 'Categorical', 'Accuracy'], 'Vartypes should only be continuous, categorical' \
                                                                 'or accuracy'
    # specify data type
    json_out = {'Type': vartype}
    # create data list
    json_data_out = []
    # iterate over dataframe and convert to dict
    for index, row in dataframe.iterrows():
        # convert row to dict and append to data list
        json_data_out.append(row.to_dict())

    json_out['Data'] = json_data_out

    return json_out

#todo add HTML class to point to html text
'''
class HTML(object):
    # utility class to hold whitebox files
    wbox_html = open('./utils/HTML/html_error.txt', 'r').read()
'''
def createMLErrorHTML(datastring, dependentVar):
    """
    create WhiteBox error plot html code
    :param datastring: json like object containing data
    :param dependentVar: name of dependent variable
    :return: html string
    """
    output = HTML.wbox_html.replace('<***>', datastring
                                  ).replace('Quality', dependentVar)

    # convert single quotes to double
    output = output.replace("'", '"')

    # convert "null" to null
    output = output.replace('"null"', 'null')

    # return
    return output




