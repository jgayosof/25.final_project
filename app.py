import pickle
import pandas as pd
import streamlit as st
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

@st.cache
def load_data() :
  #return team_points, all_teams, model
  return 1
#team_points, all_teams, model = load_data()

def same_group(home_team:str, away_team:str, all_groups:list):
  for group in all_groups :
    if home_team in group and away_team in group :
      return True
  return False

def group_stage() :
  group_results = []
  for group in all_groups:
    group_i = dict((team, 0) for team in group)
    for i in range(0, 4) :
      for j in range(i+1, 4) :
        match = [1, 2022, group[i], group[j], 55]
        if match[2] == 'Qatar' :
          match[0] = 0
        else :
          match[0] = 1
        match[2], match[3] = team_points[match[2]], team_points[match[3]]
        match_result = model.predict([match])
        #print(f'Prediction for match {match} {group[i]} vs {group[j]} is: {match_result}')
        if match_result == 'WIN':
          group_i[group[i]] += 3
        elif match_result == 'LOSE' :
          group_i[group[j]] += 3
        else:
          group_i[group[i]] += 1
          group_i[group[j]] += 1
    group_results.append(group_i)
  return group_results

# We define the groups and teams for Qatar 2022:
group_A = ['Qatar', 'Ecuador', 'Senegal', 'Netherlands']
group_B = ['England', 'Iran', 'United States', 'Wales']
group_C = ['Argentina', 'Saudi Arabia', 'Mexico', 'Poland']
group_D = ['France', 'Denmark', 'Tunisia', 'Australia']
group_E = ['Spain', 'Germany', 'Japan', 'Costa Rica']
group_F = ['Belgium', 'Canada', 'Morocco', 'Croatia' ]
group_G = ['Brazil', 'Serbia', 'Switzerland', 'Cameroon']
group_H = ['Portugal', 'Ghana', 'Uruguay', 'South Korea']
all_teams = [''] + group_A + group_B + group_C + group_D + group_E + group_F + group_G + group_H
all_groups = [group_A, group_B, group_C, group_D, group_E, group_F, group_G, group_H]

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

# load the model:
model = pickle.load(open('models/GB_WC.pkl', 'rb'))

st.title('WC Qatar 2022 match predictor')

groups = pd.DataFrame(data=[group_A, group_B, group_C, group_D, group_E, group_F, group_G, group_H]).transpose()
groups.columns = ['Group A', 'Group B', 'Group C', 'Group D',
                  'Group E', 'Group F', 'Group G', 'Group H' ]
st.dataframe(groups)

home_team = st.selectbox('Select team:', all_teams, key='home_team', index=0)
if (home_team != '') :
  away_team = st.selectbox('Select team:', all_teams, key='away_team', index=0)
  
  if (away_team != '') :
    if (away_team == home_team) :
      st.warning('Please select two different teams.')
    else :
      if same_group(home_team, away_team, all_groups):
        st.subheader(body='Group stage match:', anchor=None)
      else:
        st.subheader(body='This is not a group stage match, anyway:', anchor=None)
      # complete with default match data: neutral=1, year=2022, and WC=55
      match = [1, 2022, home_team, away_team, 55]
      # If 'away_team' == 'Qatar': swap teams because Qatar is home_team
      if away_team == 'Qatar' :
        home_team, away_team = away_team, home_team
      # If 'home_team' == 'Qatar': neutral=False, else neutral=True
      if home_team == 'Qatar' :
        match[0] = 0
      # complete match data with year=2022, and WC=55
      match[2], match[3] = team_points[home_team], team_points[away_team]

      prediction = model.predict([match])
      prediction_text = prediction.tolist()[0]

      if prediction_text == 'WIN':
          match_result = f"{home_team} wins."
      elif prediction_text == 'DRAW' :
          match_result = f"It's a draw."
      elif prediction_text == 'LOSE' :
          match_result = f"{away_team} wins."

      # Show Prediction on App
      st.subheader(body=match_result, anchor=None)

col1, col2, col3 = st.columns(3)
with col1:
    pass
with col3:
    pass
with col2 :
  but = st.button('PREDICT GROUP STAGE', help='Press this button to predict all group results')
  if but :
    group_resuls = group_stage()
    for group in group_resuls :
      group_table = pd.DataFrame(data=group, index=[0]).transpose()
      group_table.columns = ['Points']
      st.dataframe(group_table)