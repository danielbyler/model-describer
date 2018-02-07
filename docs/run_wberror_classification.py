from whitebox.whitebox import WhiteBoxError
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# read in data
df = pd.read_csv('../datasets/winequality.csv')

# set up y var
# set up some params
ydepend = 'Type'

# convert categorical
df['AlcoholContent'] = pd.Categorical(df['AlcoholContent'])
model_df = df.copy(deep = True)

model_df['AlcoholContent'] = model_df['AlcoholContent'].cat.codes

x = model_df.loc[:, model_df.columns != ydepend]
y = model_df.loc[:, ydepend]

# build model
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x, y)