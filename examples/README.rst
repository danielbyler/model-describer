.. -*- mode: rst -*-

Impact Plot Parameters
======================

model_df : pandas DataFrame, required
------------
The data you use for modeling (both continuous and dummy variables). This data frame must contain all of the 'X' columns and the 'y' column used in your modelobject.fit(X, y) modeling step. Dummy variables formed from categorical variables must have the form catvarname_value (Gender_Male, Gender_Female, etc.) or they will cause and error and not map to the output charts. 

ydepend : string, required
------------
The dependent 'y' variable you are trying to predict. Dependent variables can be continuous or categorical. 

modelobj : fitted model, required
------------
WhiteBox is designed first and foremost to work with sklearn. All sklearn objects must have been fit prior to being passed to WhiteBox.  However, any object which has a .predict function and returns an sklearn-like result: array of shape = [n_samples] or [n_samples, n_outputs] will not cause an error. 

cat_df : pandas DataFrame, required
------------
DataFrame of variables with the categorical 'data type'_https://pandas.pydata.org/pandas-docs/stable/categorical.html. This dataframe may contain string variables not present in the model. The groupbyvars must be contained in this dataset. 

groupbyvars : List, required
------------

List of variables that 'groups' the output into discrete segments for comparison. As a workaround, this column may only have a single common value if groups are not desired. 

featuredict : dictionary, optional
------------

Dictionary of variables that serves two purposes: Limiting and Labeling. 

Limiting: The keys of the dictionary limit the output. Only variables present in the keys will display in the final HTML output. 

Labeling: The values of the dictionary label the output. For example 'GenderSelected': 'Gender of Respondent' would replace the variable label for 'GenderSelected' with 'Gender of Respondent' in all of the HTML output. 
