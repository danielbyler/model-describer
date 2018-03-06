import logging

from mdesc.utils import utils as wb_utils


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
    preds = predict_engine(model_df)

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
    return cat_df




