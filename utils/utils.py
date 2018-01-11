import pandas as pd
import math
import numpy as np

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
    allresults = dataframe.describe(percentiles=np.linspace(0.01, 1, num=100), include=[np.number, 'category', 'O'])
    # Pull out the percentiles for numerical data otherwise use the mode
    tempVec = allresults.apply(lambda x: x['top'] if math.isnan(x['mean']) else x.filter(regex='[0-9]{1,2}\%',
                                                                                         axis=0), axis=0)

    return(tempVec)

