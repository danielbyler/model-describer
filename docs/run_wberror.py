from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from whitebox import utils
from whitebox.eval import WhiteBoxError


if __name__ == '__main__':

    #====================
    # wine quality dataset example
    # featuredict - cat and continuous variables

    # read in wine quality dataset
    wine = pd.read_csv('docs/datasets/winequality.csv')
    # init randomforestregressor
    modelObjc = RandomForestRegressor()

    ###
    #
    # Specify model parameters
    #
    ###
    yDepend = 'quality'
    # create second categorical variable by binning
    wine['volatile.acidity.bin'] = wine['volatile.acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
    # specify groupby variables
    groupbyVars = ['Type', 'volatile.acidity.bin']
    # subset dataframe down
    wine_sub = wine.copy(deep = True)
    # select all string columns so we can convert to pandas Categorical dtype
    string_categories = wine_sub.select_dtypes(include = ['O'])
    # iterate over string categories
    for cat in string_categories:
        wine_sub[cat] = pd.Categorical(wine_sub[cat])

    # create train dataset for fitting model
    xTrainData = wine_sub.loc[:, wine_sub.columns != yDepend].copy(deep = True)
    # convert all the categorical columns into their category codes
    xTrainData = utils.convert_categorical_independent(xTrainData)
    yTrainData = wine_sub.loc[:, yDepend]

    modelObjc.fit(xTrainData, yTrainData)

    # specify featuredict as a subset of columns we want to focus on
    featuredict = {'fixed.acidity': 'FIXED ACIDITY',
                   'Type': 'TYPE',
                   'quality': 'SUPERQUALITY',
                   'volatile.acidity.bin': 'VOLATILE ACIDITY BINS',
                   'AlcoholContent': 'AC',
                   'sulphates': 'SULPHATES'}


    WB = WhiteBoxError(modelobj = modelObjc,
                       model_df = xTrainData,
                       ydepend= yDepend,
                       cat_df = wine_sub,
                       groupbyvars = groupbyVars,
                       featuredict = featuredict,
                       verbose=None)

    WB.run()
    import os
    os.getcwd()
    WB.save(fpath='docs/PACKAGETEST_WBERROR2.html')