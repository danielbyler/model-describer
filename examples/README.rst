.. -*- mode: rst -*-

Impact Plot Parameters
======================

data: pandas DataFrame, required
------------
The data you use for modeling (both continuous and dummy variables). This data frame must contain all of the 'X' columns and the 'y' column used in your modelobject.fit(X, y) modeling step.  

dependentVar (required)
------------
The y variable you are trying to model. It can be continuous or categorical.

modelObject (required)
------------

cont_independentVar (optional)
------------
A list of variables

cat_independentVar  ouputPath chartTitle groupByVar = [] featureDict= {} cont_incrementalVal=[] cat_incrementalVal=[]

Error Plot Parameters
=====================
