.. -*- mode: rst -*-

Universal WhiteBox Parameters
======================

modelobj : fitted model, required
------------
WhiteBox is designed first and foremost to work with sklearn. Anything passed as modelobj must be a single sklearn object that has been fit prior to being passed to WhiteBox. 

model_df : pandas DataFrame, required
------------
The data used for modeling (both continuous and dummy variables). This data frame must contain all of the 'X' columns and the 'y' column used in your modelobject.fit(X, y) modeling step. Dummy variables formed from categorical variables must have the form catvarname_value (Gender_Male, Gender_Female, etc.) or they will cause and error and not map to the output charts. 

ydepend : string, required
------------
The dependent 'y' variable you are trying to predict. Dependent variables can be continuous or binary. 

cat_df : pandas DataFrame, required
------------
DataFrame of variables with the categorical `data type <https://pandas.pydata.org/pandas-docs/stable/categorical.html>`_. This dataframe may contain string variables not present in the model. The groupbyvars must be contained in this dataset. 

groupbyvars : List, required
------------

List of variables that 'groups' the output into discrete segments for comparison. As a workaround, this column may only have a single common value if groups are not desired. 

featuredict : dictionary, optional
------------

Dictionary of variables that serves two purposes: Limiting and Labeling. 

Limiting: The keys of the dictionary limit the output. Only variables present in the keys will display in the final HTML output. 

Labeling: The values of the dictionary label the output. For example 'GenderSelected': 'Gender of Respondent' would replace the variable label for 'GenderSelected' with 'Gender of Respondent' in all of the HTML output. 

Note- all variables (including ydepend and groupbyvars) must be listed and labeled if featuredict is specified.

verbose : int, optional 
-------------
Logging level of output. Level -- 0 = debug, 1 = warning, 2 = error.


aggregate_func : numpy function, optional
---------------------

Numpy function which summarizes the center of the series in question (error or sensitivity depending on the function). For example, passing np.mean in WhiteBox error will make the central line in the plot for each group the average error. Passing np.median to WhiteBoxSensitivity will show the median sensitivity for each group selected. 

WhiteBoxError Specific Parameter
=======================

error_type : string, optional
---------------------

Aggregate error metric that summarizes the positive and negative error vectors. It can take the values: 'MSE' (mean squared error), 'MAE' (mean absolute error), or 'RMSE' (root mean squared error). By default, it is the MAE so errors of [-2,-1,3,4,5] would result in an average negative error of (2+1)/2 and an average positive error of (3+4+5)/3. 


WhiteBoxSensitivity Specific Parameter
=======================

std_num : float, optional
Number of standard deviations to push data for syntehtic variable creation in the sensitivity analysis. Larger values will result in larger 'leaps of faith' of the model where it will be pushing more data outside of the range of observed data. Only values between -3 and 3 will be accepted as it is generally unrealistic to change variables more than 3 standard deviations. 

