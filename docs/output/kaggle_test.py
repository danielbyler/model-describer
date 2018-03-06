import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from mdesc.eval import SensitivityViz


df = pd.read_csv('docs/output/final_data.csv', low_memory=False)



dependentVar = 'JobSatisfaction'

#drop some variables as they are not needed
spike_cols = [col for col in df.columns if 'WorkToolsSelect' in col]
df = df.drop(spike_cols, axis=1)
spike_cols = [col for col in df.columns if 'WorkHardwareSelect' in col]
df = df.drop(spike_cols, axis=1)

#drop country specific axis as it's hard to interpret and clouds the impact of other variables
df= df.drop('Average Salary Within Country', axis=1)


#drop people who don't consider themselves Data Scientists
df = df[df['DataScienceIdentitySelect'] !='No']

#convert string variables to categorical variables
df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))

# select only those columns which are categorical in nature and make a copy for modeling
df.select_dtypes(include = ['category'])
model_df = df.copy(deep = True)

# create dummies example using all categorical columns
# this naming convention must be precise for WhiteBox to recognize what dummies are associated with what variables
dummies = pd.concat([pd.get_dummies(model_df.loc[:, col],dummy_na = True,  prefix = col) for col in model_df.select_dtypes(include = ['category']).columns], axis = 1)

# add the dummies to the numeric dataframe for modeling
finaldf = pd.concat([model_df.select_dtypes(include = [np.number]), dummies], axis = 1)

#fit gradient boosted regressor
est = GradientBoostingRegressor(n_estimators=200, learning_rate=.05,  min_weight_fraction_leaf = .01, loss='ls', random_state = 25)
est.fit(finaldf.loc[:, finaldf.columns != dependentVar], df.loc[:, dependentVar])

#keep only a subset of variables we determined were important

keepfeaturelist = ['WorkChallengeFrequencyPolitics',
                   'WorkDataVisualizations',
                   'MLToolNextYearSelect',
                   'TimeModelBuilding',
                   'Number of Algorithims',
                   'SalaryChange',
                   'EmployerSizeChange',
                   'Percent Above/Below Average Salary',
                   'LearningCategoryWork',
                   'LearningCategoryOnlineCourses',
                   'LearningPlatformUsefulnessCompany',
                   'WorkChallengeFrequencyDomainExpertise',
                   'WorkChallengeFrequencyTalent',
                   'WorkChallengeFrequencyML',
                   'RemoteWork',
                   'Age',
                   'TitleFit',
                   'DataScienceIdentitySelect',
                   'Continent',
                   'SalaryChange',
                   'JobSatisfaction',
                   'AlgorithmUnderstandingLevel',
                   'UniversityImportance'
                   ]

WB=None

#'Continent','DataScienceIdentitySelect','WorkChallengeFrequencyPolitics', 'AlgorithmUnderstandingLevel',
WB = SensitivityViz(est,
                    model_df=finaldf,
                    ydepend=dependentVar,
                    cat_df=df,
                    keepfeaturelist=keepfeaturelist,
                    groupbyvars=[ 'Continent','DataScienceIdentitySelect','WorkChallengeFrequencyPolitics', 'AlgorithmUnderstandingLevel','TitleFit'],
                    verbose=None,
                    std_num=1,
                    autoformat_types=True,
                    round_num=2
                    )

WB.run(output_type='html',
       output_path='kaggle_test.html')

docs/output/final_data.csv

