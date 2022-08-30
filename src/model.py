############ ML final project ############
import os
import joblib
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Reading the dataset
df_raw = pd.read_csv("../data/raw/results.csv")
# We make a copy of the original dataset
df = df_raw.copy()

######################
# Data Preprocessing #
######################
##### Date feature #####
# We split the date and store the year information in another column
df["year"]=pd.DatetimeIndex(df["date"]).year
# Date feature format to datetime (format: YEAR-MONTH-DAY --> %Y-%m-%d)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# Elimination of registers before 2012
date_lb = pd.Timestamp(2012,1,1)
df = df[(df['date'] > date_lb)]
# Drop the columns date
df = df.drop(['date'], axis=1)

##### Remove the outliers #####

# For home_score and away_score
df=df.drop(df[df['home_score'] > 5].index)
df=df.drop(df[df['away_score'] > 5].index)

##### Home team and Away team features #####

# We define the groups and teams for Qatar 2022
group_A = ['Qatar', 'Ecuador', 'Senegal', 'Netherlands']
group_B = ['England', 'Iran', 'United States', 'Wales']
group_C = ['Argentina', 'Saudi Arabia', 'Mexico', 'Poland']
group_D = ['France', 'Denmark', 'Tunisia', 'Australia']
group_E = ['Spain', 'Germany', 'Japan', 'Costa Rica']
group_F = ['Belgium', 'Canada', 'Morocco', 'Croatia' ]
group_G = ['Brazil', 'Serbia', 'Switzerland', 'Cameroon']
group_H = ['Portugal', 'Ghana', 'Uruguay', 'South Korea']
all_teams=group_A +group_B +group_C +group_D +group_E +group_F +group_G +group_H
# We remove the data of the countries that are not part of the world cup 2022 
df.drop(df[~df['home_team'].isin(all_teams)].index, inplace = True)
df.drop(df[~df['away_team'].isin(all_teams)].index, inplace = True)
# Encode the teams by FIFA Ranking at 23th of june 2022
team_points = {'Argentina' : 1770.65, 'Australia' : 1483.73, 
                'Belgium': 1821.92, 'Brazil' : 1837.56, 'Cameroon' : 1484.95, 
                'Canada' : 1473.82, 'Costa Rica' : 1500.06, 'Croatia' : 1632.15, 
                'Denmark' : 1665.47, 'Ecuador' : 1463.74, 'England' : 1737.46, 
                'France' : 1764.85, 'Germany' : 1658.96, 'Ghana' : 1389.68, 
                'Iran' : 1558.64, 'Japan' : 1552.77, 'Mexico' : 1649.57, 
                'Morocco' : 1558.9, 'Netherlands' : 1679.41, 'Poland' : 1546.18, 
                'Portugal' : 1678.65, 'Qatar' : 1441.41, 
                'Saudi Arabia' : 1435.74, 'Senegal' : 1593.45, 
                'Serbia' : 1549.53, 'South Korea' : 1526.2, 'Spain' : 1716.93, 
                'Switzerland' : 1621.43, 'Tunisia' : 1507.86, 
                'United States' : 1635.01, 'Uruguay' : 1643.71, 
                'Wales' : 1582.13}
# Map the countries ranking points
df ['h_team_points'] = df['home_team'].map(team_points)
df ['a_team_points'] = df['away_team'].map(team_points)
# Drop the columns not useful anymore
df = df.drop(['country', 'city', 'home_team', 'away_team'], axis=1)

##### Tournaments feature #####

# We group similar tournaments together
Friendly = ['Friendly', 'Superclásico de las Américas', 'Intercontinental Cup']
Qualifications = ['FIFA World Cup qualification']
WC = ['FIFA World Cup', 'Confederations Cup']
Cups = ['African Cup of Nations qualification', 'African Cup of Nations', 
        'African Nations Championship qualification', 
        'African Nations Championship', 'Copa América qualification', 
        'Copa América', 'UEFA Euro qualification', 'UEFA Euro', 
        'UEFA Nations League', 'AFC Asian Cup qualification', 'AFC Asian Cup', 
        'CFU Caribbean Cup qualification', 'Oceania Nations Cup qualification',
        'Oceania Nations Cup', 'CONCACAF Nations League qualification', 
        'CONCACAF Nations League', 'Gold Cup qualification', 'Gold Cup', 
        'Arab Cup qualification', 'Arab Cup', 'COSAFA Cup', 'CECAFA Cup',
        'Gulf Cup', 'EAFF Championship', 'CFU Caribbean Cup', 'Baltic Cup']
NoOthers = ['Friendly', 'Qualifications', 'WC', 'Cups']

df['tournament'] = df['tournament'].apply(lambda x: x if x not in Friendly else 'Friendly')
df['tournament'] = df['tournament'].apply(lambda x: x if x not in Qualifications else 'Qualifications')
df['tournament'] = df['tournament'].apply(lambda x: x if x not in WC  else 'WC')
df['tournament'] = df['tournament'].apply(lambda x: x if x not in Cups else 'Cups')
df['tournament'] = df['tournament'].apply(lambda x: x if x in NoOthers else 'Others')
# We map the data using the groups clasification and using the FIFA Ranking Procedures - importance of match we set the weight for each group.
tournament_points = {'Friendly':10, 'Qualifications':25, 'Cups':20, 'WC':55, 'Others':8}
df['tournament_points'] = df['tournament'].map(tournament_points)
# Drop the column not useful anymore
df = df.drop(['tournament'], axis=1)

##### Match result feature #####
# Setting the match winner using the goal difference
df['match_result'] = np.where(df['home_score'] - df['away_score'] > 0, 'WIN', np.where(df['home_score'] - df['away_score'] < 0, 'LOSE', 'DRAW'))
# Drop the column not useful anymore
df = df.drop(['home_score', 'away_score' ], axis=1)

##### Neutral result feature #####
df ['neutral'] = df['neutral'].map({True: 1, False: 0})

# dump final csv
df.to_csv('../data/processed/results_processed.csv')

#####################
# Model and results #
#####################
# train-test split: train: before 2018(Russia WC), test: after 2018 (including WC)
Train = df[df['year'] < 2018]
Test = df[df['year'] >= 2018]

# Our target is match_result
X_train = Train.drop('match_result', axis='columns')
y_train = Train['match_result']
X_test = Test.drop('match_result', axis='columns')
y_test = Test['match_result']

# Gradient Boosting:
model_GB = GradientBoostingClassifier(learning_rate=0.1, n_estimators=90, max_depth=4, random_state=13)
model_GB.fit(X_train, y_train)

# Metrics & Results:
y_pred = model_GB.predict(X_test)

print(f'CLASSIFICATION REPORT: match_result \n {classification_report(y_test, y_pred)}')

# Get the score of train data just to verify its 1.
score = model_GB.score(X_train, y_train)
print(f'The score for Gradient Boosting with X_train & y_train is: {score}')

#Get the score for the predictions:
score = model_GB.score(X_test, y_test)
print(f'The score for Gradient Boosting with X_test & y_test is: {score}')

# Tree params
print(f'GBoost params: \n {model_GB.get_params()} \n')

# save the model:
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../models/GB_WC.pkl')
pickle.dump(model_GB, open(filename, 'wb'))

model = pickle.load(open(filename, 'rb'))

##############
# Prediction #
##############
match = [1, 2022, 'Brazil', 'Cameroon', 'WC']

# If 'home_team' == 'Qatar': neutral=False, else neutral=True
if match[2] == 'Qatar' :
  match[0] = 0
else :
  match[0] = 1

match[2], match[3], match[4] = team_points[match[2]], team_points[match[3]], tournament_points[match[4]]
#print(f'Prediction for match {match} is {model_GB.predict([match])}')

# predict group results:
def predict_group(group : list) :
  for i in range(4) :
    for j in range(i+1,4) :
      match = [1, 2002, group[i], group[j], 'WC']
      print(f'{match}')
      if match[2] == 'Qatar' :
        match[0] = 0
      else :
        match[0] = 1
      match[2], match[3], match[4] = team_points[match[2]], team_points[match[3]], tournament_points[match[4]]
      print(f'Prediction for match {match} {group[i]} vs {group[j]} is: {model.predict([match])}')

# predict all_group results:
all_groups = [group_A, group_B, group_C, group_D, group_E, group_F, group_G, group_H]
for group in all_groups:
  predict_group(group)
