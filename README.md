# WhiteBox: One line of code to make 'black box' machine learning models interpretable to humans. 
  
White Box makes it possible for everyday humans to understand 'black-box' machine learning models in two key ways:

1- WhiteBox makes it easy to understand how the model 'believes' different groups behave within the model (e.g. are the drivers of job satisfaction different across continents)

2- WhiteBox makes it clear where the model is making disproportionately poor quality estimations (e.g. do we do a worse job at predicting the job satisfaction in one particular continent)

To make communicating these findings to your team easy, WhiteBox outputs: 

- Are created with one line of Python code at the end of your existing machine learning workflow and require no model re-training

- Are interactive HTML files that are small enough to be emailed as attachment and only require your teammate/client to have a web browser to open your attachment. No server or messy installation required.

- Do not expose your potentially sensitive raw data. Only summaries of the data are included in the final HTML file. This also makes it possible to summarize models built on extremely large datasets into file sizes that are small enough for email. 

# Sample Outputs

## Impact

Currently, many people substitute [variable importance](https://en.wikipedia.org/wiki/Random_forest#Variable_importance) charts for an understanding of how the model works. While genuinely helpful, these plots do not help us understand how different subgroups behave differently under the hood of the model. In the example below ([full notebook here](https://github.com/Data4Gov/WhiteBox/blob/master/Example_Notebook/Random%20Forest%20Analysis.ipynb)), all you have to do to produce the interactive chart is this line of code: 
```python
sensitivity_plot( data = final_data, dependentVar = dependentVar, modelObject = Rf, outputPath = outputPath, 
groupByVar = [ 'AlgorithmUnderstandingLevel']
```
Please note all descriptive text is automatically generated by WhiteBox and use quartiles as cutoff points for the narrative text:

![Error Showing Impact Chart](https://github.com/Data4Gov/WhiteBox/blob/master/img/Impact_Gif.gif "What a Random Forest thinks about what makes good  alcohol")

In the above charts, each variable's chart is generated by going through the dataset and generating two predictions for each row. First, WhiteBox uses the modelObject to generate a prediction on all of the original data. Then each variable in question (like presence of politics at work) is increased by one standard deviation and the model is run again on the synthetic data. The average gap in predictions between the real data and the simulated data is the 'impact' that variable has on job satisfaction (our dependent variable). This is repeated for all variables we are interested in. For categorical variables, categories are set to the mode for the creation of synthetic data.   

## Error

There are a hundred ways to skin an error chart. Almost all of them are reasonable. However, few can be proceeded by the comment
#Send To Boss As Attachment With No Additional Editing

We hope our error charts fill that gap for you. These error charts group the level of error by type and show where the error may be less or more for different parts of different variables. Again, only one line of code is required to run it:
```python
err_plot(data  = final_data,dependentVar = dependentVar, outputPath = outputPath, modelObject = Rf,
groupByVar = [ 'AlgorithmUnderstandingLevel'])
```


![Error Showing Impact Chart](https://github.com/Data4Gov/WhiteBox/blob/master/img/Impact_Gif.gif "What a Random Forest Thinks About What Makes Good  Wine")

For a more detailed example, see our [example notebook](https://github.com/Data4Gov/WhiteBox/blob/master/Example_Notebook/Random%20Forest%20Analysis.ipynb)

# Helpful Tips

## Handling Categorical Variables


In many models, categorical variables are present as independent variables. To provide meaningful charts, WhiteBox requires two things:

- Categorical dummies must have the naming convention varname_category (for example Gender_Male and Gender_Female). One way to generate these is

```python
#find string variables
categorical = final_data.select_dtypes(include={'object'})
categorical_dummies = pd.get_dummies(categorical.applymap(str), prefix = categorical.columns)
```

- The 'data' parameter for WhiteBox must include the dependent variable, all continuous variables, all dummy variables, and all string variables that the dummy variables were created from. If the process of creating these dummy variables poses a problem, just pass an untrained model object and WhiteBox will train the model for you and return the trained model as an output. 

## Managing Output Length

Many times, models will have hundreds (or more) of independent variables. To downselect those to a more managable number, and improve the quality of the output, we recommend using the featureDict parameter (present in both functions). By feeding in a dictionary like {'var1' : 'Gender' , 'var2' : 'Income' }, you will make the HTML output only print output relating to var1 and var2. Also, instead of displaying the name in your dataframe, the HTML file will display the name you give it in your dictionary. 

# Supported Machine Learning Libraries

We currently support all sklearn classifiers. We will look to add support for things like H20 in the future. In all implementations, we are committed to keeping our 'one line of code' promise. 

We currently only support traditional tabular data. We are hoping to include text, audio, video, and images but they are not part of the current implementation. 

## Other Machine Learning Interpretability Projects

For those looking for intepretation of individual points, please see the [Lime](https://github.com/marcotcr/lime) project and its good work. 

# Authors:
[Daniel Byler](https://www.linkedin.com/in/danielbyler/), [Venkatesh Gangavarapu](https://www.linkedin.com/in/venkatesh-gangavarapu-9845b36b/), [Jason Lewris](https://www.linkedin.com/in/jasonlewris/), [Shruti Panda](https://www.linkedin.com/in/shruti-panda-1466216a/), and [Shanti Jha](https://www.linkedin.com/in/shantijha/) 




