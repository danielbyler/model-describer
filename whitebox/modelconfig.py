import logging

try:
    import utils.utils as wb_utils
except:
    import whitebox.utils.utils as wb_utils


def sklearn_predict(predict_engine,
                    modelobj,
                    model_df,
                    cat_df,
                    ydepend,
                    model_type):
    """
    create predictions based on trained model object, dataframe, and dependent variables
    :return: dataframe with prediction column
    """
    logging.info("""Creating predictions using modelobj.
                    \nModelobj class name: {}""".format(modelobj.__class__.__name__))
    # create predictions
    preds = predict_engine(
        model_df.loc[:, model_df.columns != ydepend])

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
    model_df['predictedYSmooth'] = preds
    # return
    return cat_df, model_df

