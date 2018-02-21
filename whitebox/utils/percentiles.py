import pandas as pd
import numpy as np
import math

try:
    import utils.utils as wb_utils
except:
    import whitebox.utils.utils as wb_utils


def create_group_percentiles(df,
                             groupbyvars,
                             wanted_percentiles=None):
    """
    create percentile buckets for based on groupby for numeric columns
    :param df: dataframe
    :param groupbyvars: groupby variable list
    :param wanted_percentiles: desired percnetile lines for user intereface
    :return: json formatted percentile outputs
    """
    # calibrate default percentiles
    if not wanted_percentiles:
        wanted_percentiles = [0, .01, .1, .25, .5, .75, .9, 1]

    groupbyvars = list(groupbyvars)
    # subset numeric cols
    num_cols = df.select_dtypes(include=[np.number])
    final_out = {'Type': 'PercentileGroup'}
    final_list = []
    # iterate over
    for col in num_cols:
        data_out = {'variable': col}
        groupbylist = []
        # iterate groupbys
        for group_name in groupbyvars:
            # iterate over each slice of the groups
            for name, group in df.groupby(group_name):
                # get col of interest
                group = group.loc[:, col]
                # start data out for group
                group_out = {'groupByVar': name}
                # capture wanted percentiles
                group_percent = group.quantile(wanted_percentiles).reset_index().rename(columns={'index': 'percentiles',
                                                                                                 col: 'value'})
                # readjust percentiles to look nice
                group_percent.loc[:, 'percentiles'] = group_percent.loc[:, 'percentiles'].apply(lambda x: str(int(x*100))+'%')
                # convert percnetile dataframe into json format
                group_out['percentileValues'] = group_percent.to_dict(orient='records')
                # append group out to group placeholder list
                groupbylist.append(group_out)
        # assign groupbylist out
        data_out['percentileList'] = groupbylist
        final_list.append(data_out)
    final_out['Data'] = final_list
    return final_out


def create_percentile_vecs(input,
                           percentiles=None):
    """
    # support dataframe and series objects
    :param dataframe: pandas dataframe object
    :return: pandas dataframe object with percentiles
    """
    # ensure dataframe is pandas dataframe object
    if percentiles is None:
        percentiles = np.linspace(0.01, 1, num=100)

    # check dtype of input
    if isinstance(input, pd.DataFrame):
        # calculate the percentiles for numeric data
        allresults = input.describe(percentiles=percentiles,
                                        include=[np.number])


        tempvec = allresults.filter(regex='[0-9]{1,2}\%', axis=0)

    elif isinstance(input, pd.Series):
        tempvec = input.quantile(percentiles)

    else:
        raise TypeError("""unsupported type for percentile creation: {}""".format(type(input)))

    return tempvec

class Percentiles(object):

    def __init__(self, df,
                 groupbyvars):

        self._df = df
        self._groupbyvars = groupbyvars

    def population_percentiles(self):
        """
                create population percentiles, and group percentiles
                :return: NA
                """
        # create instance wide percentiles for all numeric columns
        # including 0% through 100%
        self.population_percentile_vecs = create_percentile_vecs(self._df,
                                                                 percentiles=np.linspace(0, 1, num=101))
        # create percentile bars for final out
        self._percentiles_out()
        # create groupby percentiles
        self.group_percentiles_out = create_group_percentiles(self._df,
                                                                        self._groupbyvars)

    def _percentiles_out(self):
        """
        Create designated percentiles for user interface percentile bars
            percentiles calculated include: 0, 1, 10, 25, 50, 75 and 90th percentiles
        :return: Save percentiles to instance for retrieval in final output
        """
        # send the percentiles to to_json to create percentile bars in UI
        percentiles = self.population_percentile_vecs.reset_index().rename(columns={"index": 'percentile'})
        # get 0, 1, 25, 50, 75, and 90th percentiles
        final_percentiles = percentiles.iloc[[0, 1, 10, 25, 50, 75, 90]].copy(deep=True)
        # melt to long format
        percentiles_melted = pd.melt(final_percentiles, id_vars='percentile')
        # convert to_json
        self.percentiles = wb_utils.to_json(dataframe=percentiles_melted,
                                   vartype='Percentile',
                                   html_type='percentile',
                                   incremental_val=None)

