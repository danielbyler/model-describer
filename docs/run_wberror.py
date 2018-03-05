from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from whitebox.utils import utils
from whitebox.utils import percentiles
from whitebox.eval import WhiteBoxError
import numpy as np

#====================
# wine quality dataset example
# featuredict - cat and continuous variables
wine = utils.create_wine_data(None)

wine.head()


# init randomforestregressor
modelObjc = RandomForestRegressor(random_state=2)

###
#
# Specify model parameters
#
###
ydepend = 'quality'
# create second categorical variable by binning
wine['volatile.acidity.bin'] = wine['volatile acidity'].apply(lambda x: 'bin_0' if x > 0.29 else 'bin_1')
# specify groupby variables
groupbyVars = ['Type'] #, 'volatile.acidity.bin']
# subset dataframe down
wine_sub = wine.copy(deep = True)

mod_df = pd.get_dummies(wine_sub.loc[:, wine_sub.columns != ydepend])




modelObjc.fit(mod_df,
              wine_sub.loc[:, ydepend])

keepfeaturelist = ['fixed acidity',
                   'Type',
                   'quality',
                   'volatile.acidity.bin',
                   'alcohol',
                   'sulphates']

WB = WhiteBoxError(modelobj=modelObjc,
                   model_df=mod_df,
                   ydepend=ydepend,
                   cat_df=wine_sub,
                   groupbyvars=['Type', 'alcohol'],
                   keepfeaturelist=None,
                   verbose=None,
                   round_num=2,
                   autoformat_types=True)

import tqdm
tqdm.__version__
WB.run(output_type='html',
       output_path='REGRESSIONTEST2.html')

import math
df = pd.DataFrame({'col1': np.random.uniform(100000, 200000),
                   'col2': np.random.rand(100),
                   'col3': ['a'] * 50 + ['b'] * 50})

df['col3'].ix[50] = math.nan

res = df.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))

df.apply(lambda x: x.fillna('null') if x.dtype.kind == 'O' else x)

df[df['col3'].isnull()]

df['col3'].dtype.kind

res = df.select_dtypes(include=['O']).fillna('null')
res.ix[50]
df['col3'].ix[50]
preds = modelObjc.predict(mod_df)

wine_sub['errors'] = preds - wine_sub[ydepend]
errors = preds - wine_sub[ydepend]
def err(errors):
    errors = errors.values
    return np.nanmedian(errors[errors >=0]), np.nanmedian(errors[errors <= 0])

wine_sub.groupby(['Type', 'volatile.acidity.bin'])['errors'].apply(lambda x: err(x))

wine_sub.columns
WB._cat_df['predictedYSmooth'].mean()

#wine_sub['preds'] =modelObjc.predict(mod_df.loc[:, mod_df.columns != ydepend])

wine_sub['preds'] =WB.modelobj.predict(mod_df)
wine_sub['preds'].mean()
WB._cat_df['predictedYSmooth'].mean()

notwant = ['predictedYSmooth', 'errors', ydepend]
tes = WB.modelobj.predict(WB._model_df.loc[:, ~WB._model_df.columns.isin(notwant)])
tes.mean()
WB._model_df['predictedYSmooth'].tail()
wine_sub['quality'].mean()
WB._cat_df['quality'].mean()
wine_sub.columns

wine_sub['preds'].tail()

mod_df.head()
WB._model_df.head()

for col in mod_df.columns:
    boolean = mod_df.loc[:, col].values.tolist() == WB._model_df.loc[:, col].values.tolist()
    if not boolean:
        print(col)

from whitebox.utils.fmt_model_outputs import fmt_sklearn_preds

cat, mod = fmt_sklearn_preds(getattr(modelObjc, 'predict'),
                            modelObjc,
                             mod_df,
                             wine_sub,
                             ydepend,
                             'regression')


cat['predictedYSmooth'].mean()

engine = getattr(modelObjc, 'predict')
preds = engine(mod_df)
np.mean(preds)
wine_sub['errors'] = wine_sub['preds'] - wine_sub[ydepend]

white = wine_sub.groupby('Type').get_group('White')

errs = white.loc[white['volatile.acidity.bin'] == 'bin_1']['errors'].values

np.nanmedian(errs[errs <= 0])

wine_sub['preds'].mean()
WB._model_df['predictedYSmooth'].mean()

WB.agg_df.head()
WB._cat_df['errors'].mean()
wine_sub['errors'].mean()
np.mean(white['errors']**2)
from timeit import Timer

#T = Timer(lambda: WB.run(output_type='html',
#       output_path='REGRESSIONTEST2.html'))

#T.timeit(number=1)



WB.outputs

WB.agg_df.tail(100)

import logging
from whitebox.utils import utils as wb_utils

def fmt_sklearn_preds(predict_engine,
                      modelobj,
                      model_df,
                      cat_df,
                      ydepend,
                      model_type):
    """
    create preds based on model type - in the case of binary classification,
    pull predictions for the first index - preds corresponding to label 1

    :param predict_engine: modelobjs prediction attribute
    :param modelobj: sklearn model object
    :param model_df: dataframe used to train modelobj
    :param cat_df: dataframe in original form before categorical conversions
    :param ydepend: str dependent variable name
    :param model_type: str (classification, regression)
    :return: cat_df and model_df with predictions inserted
    :rtype: pd.DataFrame
    """
    logging.info("""Creating predictions using modelobj.
                    \nModelobj class name: {}""".format(modelobj.__class__.__name__))

    # create predictions, filter out extraneous columns
    preds = predict_engine(
        model_df)

    if model_type == 'regression':
        # calculate error
        diff = preds - cat_df.loc[:, ydepend]
    elif model_type == 'classification':
        # select the prediction probabilities for the class labeled 1
        preds = preds[:, 1].tolist()
        # create a lookup of class labels to numbers
        class_lookup = {class_: num for num, class_ in enumerate(modelobj.classes_)}
        # convert the ydepend column to numeric
        actual = cat_df.loc[:, ydepend].apply(lambda x: class_lookup[x]).values.tolist()
        # calculate the difference between actual and predicted probabilities
        diff = [wb_utils.prob_acc(true_class=actual[idx], pred_prob=pred) for idx, pred in enumerate(preds)]
    else:
        raise RuntimeError(""""unsupported model type
                                \nInput Model Type: {}""".format(model_type))

    # assign errors
    cat_df['errors'] = diff
    # assign predictions
    logging.info('Assigning predictions to instance dataframe')
    cat_df['predictedYSmooth'] = preds
    # return
    return cat_df, model_df, preds


del mod_df[ydepend]
import logging
modelObjc.fit(mod_df, wine_sub.loc[:, ydepend])

cat, mod, preds = fmt_sklearn_preds(getattr(modelObjc, 'predict'),
                            modelObjc,
                             mod_df,
                             wine_sub,
                             ydepend,
                             'regression')



np.mean(preds)

unwanted_pred_cols = [ydepend, 'predictedYSmooth']
    # create predictions, filter out extraneous columns
preds = engine(
        mod_df.loc[:, list(set(mod_df.columns).difference(set(unwanted_pred_cols)))])

np.mean(preds)
p2 = modelObjc.predict(mod_df.loc[:, list(set(mod_df.columns).difference(set(unwanted_pred_cols)))])

np.mean(p2)
p3 = engine(mod_df.loc[:, mod_df.columns!=ydepend])

r = mod_df.loc[:, list(set(mod_df.columns).difference(set(unwanted_pred_cols)))]




mod_df.loc[:, mod_df.columns!='predictedYSmooth'].shape
mod_df.shape
np.mean(p3)
