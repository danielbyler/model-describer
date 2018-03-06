import pandas as pd
import numpy as np

try:
    import utils.utils as wb_utils
    import utils.formatting as formatting
except ImportError:
    import mdesc.utils.utils as wb_utils
    import mdesc.utils.formatting as formatting


def create_group_percentiles(df,
                             groupbyvars,
                             wanted_percentiles=None,
                             round_num=2):
    """
    create percentiles based on groupby variable

    :param df: dataframe to create percentiles from
    :param groupbyvars: list of groupby variables
    :param wanted_percentiles: list of desired percentiles
    :return json formatted percentile outputs
    :rtype dict
    """
    # calibrate default percentiles
    if not wanted_percentiles:
        wanted_percentiles = wb_utils.Settings.output_percentiles

    assert isinstance(groupbyvars, list),  """groupbyvars must be of type list"""

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
                                                                                                 col: 'value'}).round(round_num)
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


def create_percentile_vecs(input_var,
                           percentiles=None):
    """
    # create percentiles for series and dataframe objects

    :param input_var: series or dataframe object
    :param percentiles: desired percentile buckets
    :return: pandas dataframe or series object based on input
    :rtype pd.DataFrame | pd.Series
    """
    # ensure dataframe is pandas dataframe object
    if percentiles is None:
        percentiles = np.linspace(0.01, 1, num=100)

    # check dtype of input
    if isinstance(input_var, pd.DataFrame):
        # calculate the percentiles for numeric data
        allresults = input_var.describe(percentiles=percentiles,
                                        include=[np.number])

        tempvec = allresults.filter(regex='[0-9]{1,2}\%', axis=0)

    elif isinstance(input_var, pd.Series):
        tempvec = input_var.quantile(percentiles)

    else:
        raise TypeError("""unsupported type for percentile creation: {}""".format(type(input_var)))

    return tempvec


class Percentiles(object):

    def __init__(self,
                 df,
                 groupbyvars,
                 round_num=2):
        """
        Percentiles creates and holds percentile information for dataframe object

        :param df: dataframe input
        :param groupbyvars: list of groupby variables
        """
        self._df = df
        self.round_num = round_num
        self._groupbyvars = groupbyvars
        self.population_percentiles()


    def population_percentiles(self):
        """
        create and assign class attribute df population and groupby percentiles

        """
        # create instance wide percentiles for all numeric columns
        # including 0% through 100%
        self.population_percentile_vecs = create_percentile_vecs(self._df,
                                                                 percentiles=np.linspace(0, 1, num=101))
        # create percentile bars for final out
        self._percentiles_out()
        # create groupby percentiles
        self.group_percentiles_out = create_group_percentiles(self._df,
                                                              self._groupbyvars,
                                                              round_num=self.round_num)

    def _percentiles_out(self):
        """
        Create designated percentiles for user interface percentile bars
            percentiles calculated include: 0, 1, 10, 25, 50, 75 and 90th percentiles

        """
        # send the percentiles to to_json to create percentile bars in UI
        percentiles = self.population_percentile_vecs.reset_index().rename(columns={"index": 'percentile'})
        # get 0, 1, 25, 50, 75, and 90th percentiles
        final_percentiles = percentiles.iloc[wb_utils.Settings.fmt_percentiles_out].copy(deep=True)
        # melt to long format
        percentiles_melted = pd.melt(final_percentiles, id_vars='percentile').round(self.round_num)
        # convert to_json
        self.percentiles = formatting.FmtJson.to_json(dataframe=percentiles_melted,
                                                      vartype='Percentile',
                                                      html_type='percentile',
                                                      incremental_val=None)
