import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from whitebox.eval import WhiteBoxSensitivity


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
# there is a duplicate in keep feature list
keepfeaturelist = list(set(keepfeaturelist))

df = pd.read_csv('docs/notebooks/datasets/final_data.csv', low_memory=False)
df = df.loc[:, keepfeaturelist].copy(deep=True)

df.fillna('nan', inplace=True)




dependentVar = 'JobSatisfaction'

#drop some variables as they are not needed
spike_cols = [col for col in df.columns if 'WorkToolsSelect' in col]
df = df.drop(spike_cols, axis=1)
spike_cols = [col for col in df.columns if 'WorkHardwareSelect' in col]
df = df.drop(spike_cols, axis=1)

#drop people who don't consider themselves Data Scientists
df = df[df['DataScienceIdentitySelect'] !='No']

# copy datframe for creating dummies
model_df = df.loc[:, df.columns != dependentVar].copy(deep = True)

type(model_df)
finaldf = pd.get_dummies(model_df, dummy_na=True)

#fit gradient boosted regressor
est = GradientBoostingRegressor(n_estimators=200, learning_rate=.05,  min_weight_fraction_leaf = .01, loss='ls', random_state = 25)
est.fit(finaldf, df.loc[:, dependentVar])

#keep only a subset of variables we determined were important

#'Continent','DataScienceIdentitySelect','WorkChallengeFrequencyPolitics', 'AlgorithmUnderstandingLevel',
# df['TitleFit'] = df['TitleFit'].fillna('nan')

WB = WhiteBoxSensitivity(est,
                   model_df=finaldf,
                   ydepend=dependentVar,
                   cat_df=df,
                   keepfeaturelist=keepfeaturelist,
                   groupbyvars= [ 'Continent','DataScienceIdentitySelect','WorkChallengeFrequencyPolitics', 'AlgorithmUnderstandingLevel', 'TitleFit'],
                   verbose=None,
                    std_num=1,
                    autoformat_types=True,
                    round_num=2
                   )

WB.run(output_type='html',
       output_path='kaggle_test_5groupby.html')

finaldf.filter(regex='WorkChallengeFrequencyPolitics').columns
copydf = finaldf.copy(deep=True)
copydf.shape
non_mode_mask = copydf['WorkChallengeFrequencyPolitics_Never'] != 1

copydf = copydf.loc[non_mode_mask, :]
copydf.loc[:, 'WorkChallengeFrequencyPolitics_Most of the time']= 0
copydf.loc[:, 'WorkChallengeFrequencyPolitics_Never'] = 1
copydf.loc[:, 'WorkChallengeFrequencyPolitics_Often'] = 0
copydf.loc[:, 'WorkChallengeFrequencyPolitics_Rarely'] = 0
copydf.loc[:, 'WorkChallengeFrequencyPolitics_Sometimes'] = 0
copydf.loc[:, 'WorkChallengeFrequencyPolitics_nan'] = 0


preds = est.predict(copydf)
og_preds = est.predict(finaldf.loc[non_mode_mask, :])

diff = preds -og_preds

cat_df = df.loc[non_mode_mask, :].copy(deep=True)

cat_df['diff'] = diff

cat_df.groupby(['Continent', 'WorkChallengeFrequencyPolitics'])['diff'].mean()

df.groupby('WorkChallengeFrequencyPolitics')['TitleFit'].count()

WB.outputs

og_col = 'WorkDataVisualizations'
levels = cat_df['WorkDataVisualizations'].unique().tolist()
mode = cat_df['WorkDataVisualizations'].mode()[0]
mode_col = '{}_{}'.format(og_col, mode)
non_mode_mask = cat_df.loc[cat_df[mode_col] == 1, :]

print('hello')