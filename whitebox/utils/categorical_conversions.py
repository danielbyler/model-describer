

def pandas_switch_modal_dummy(cur_col,
                                rev_col,
                                cat_df,
                                copydf):
    # switch modal value for categorical variable converted
    # for modelling with pd.get_dummies
    # map categories with main column name to properly subset
    all_type_cols = ['{}_{}'.format(rev_col, cat) for cat in cat_df.loc[:, cur_col].unique()]
    # find the mode from the original cat_df for this column
    modal_val = str(cat_df[cur_col].mode().values[0])
    # find the columns within all_type_cols related to the mode_val
    mode_col = list(filter(lambda x: modal_val in x, all_type_cols))
    # convert mode cols to all 1's
    copydf.loc[:, mode_col] = 1
    # convert all other non mode cols to zeros
    non_mode_col = list(filter(lambda x: modal_val not in x, all_type_cols))
    # filter to columns present in the model dataframe
    non_mode_col = list(set(non_mode_col) & set(copydf.columns))
    # switch non modal columns to 0
    copydf.loc[:, non_mode_col] = 0
    # return df with switch modal column
    return modal_val, copydf