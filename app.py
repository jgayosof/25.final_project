from multiprocessing.connection import answer_challenge
import pickle
from unittest import result
import pandas as pd
import numpy as np
import streamlit as st


### FRONTEND ###
# We define the groups and teams for Qatar 2022
group_A = ['Qatar', 'Ecuador', 'Senegal', 'Netherlands']
group_B = ['England', 'Iran', 'United States', 'Wales']
group_C = ['Argentina', 'Saudi Arabia', 'Mexico', 'Poland']
group_D = ['France', 'Denmark', 'Tunisia', 'Australia']
group_E = ['Spain', 'Germany', 'Japan', 'Costa Rica']
group_F = ['Belgium', 'Canada', 'Morocco', 'Croatia' ]
group_G = ['Brazil', 'Serbia', 'Switzerland', 'Cameroon']
group_H = ['Portugal', 'Ghana', 'Uruguay', 'South Korea']
all_teams = group_A + group_B + group_C + group_D + group_E + group_F + group_G + group_H

home_team = st.selectbox('Select "Home Team"', all_teams, key='home_team')
away_team = st.selectbox('Select "Away Team"', all_teams, key='away_team')


model = pickle.load(open('GB_WC.pkl', 'rb'))
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

# complete with default match data: neutral=1, year=2022, and WC=55
match = [1, 2022, home_team, away_team, 55]
# If 'home_team' == 'Qatar': neutral=False, else neutral=True
if home_team == 'Qatar' :
  match[0] = 0
else :
  match[0] = 1

# complete match data with year=2022, and WC=55
match[2], match[3] = teams_points[home_team], teams_points[away_team]

#print(f'match: {match}')

prediction = model.predict(match)
prediction_text = prediction.tolist()[0]

if prediction_text == 'WIN':
    match_result = f"{home_team} wins."
elif prediction_text == 'DRAW' :
    match_result = f"It's a draw."
elif prediction_text == 'LOSE' :
    match_result = f"{away_team} wins."


# Show Prediction on App
st.code(match_result)