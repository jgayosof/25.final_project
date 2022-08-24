############ ML final project ############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, make_scorer

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
# Elimination of registers before 2010.
date_lb = pd.Timestamp(2010,1,1)
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
teams_points = {'Argentina' : 1770.65, 'Australia' : 1483.73, 
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
df ['h_team_points'] = df['home_team'].map(teams_points)
df ['a_team_points'] = df['away_team'].map(teams_points)
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
df['tournament_point'] = df['tournament'].map({'Friendly':10, 'Qualifications':25, 'Cups':20, 'WC':55, 'Others':8})
# Drop the column not useful anymore
df = df.drop(['tournament'], axis=1)

##### Match result feature #####
# Setting the match winner using the goal difference
df['match_result'] = np.where(df['home_score'] - df['away_score'] > 0, 'WIN', np.where(df['home_score'] - df['away_score'] < 0, 'LOSE', 'DRAW'))
# Drop the column not useful anymore
df = df.drop(['home_score',	'away_score' ], axis=1)

##### Neutral result feature #####
df ['neutral'] = df['neutral'].map({True: 1, False: 0})

#####################
# Model and results #
#####################