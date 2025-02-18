import streamlit as st
import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
import time

# Page setup
st.set_page_config(page_title="NBA Game Predictions", page_icon="🏀", layout="wide")
st.markdown("<h1>NBA Player Points Analysis 🏀</h1>", unsafe_allow_html=True)

# Constants
SEASON_CURRENT = '2024-25'
SEASON_PREVIOUS = '2023-24'
CACHE_TTL = 3600

def get_team_abbreviations():
    return [t['abbreviation'] for t in teams.get_teams()]

def get_team_by_abbreviation(abbrev):
    return teams.find_team_by_abbreviation(abbrev)

@st.cache_data(ttl=CACHE_TTL)
def get_team_roster(team_abbrev):
    team_info = get_team_by_abbreviation(team_abbrev)
    if not team_info:
        st.error(f"Team '{team_abbrev}' not found.")
        return []
    team_id = team_info['id']
    roster_df = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
    return roster_df['PLAYER'].tolist()

@st.cache_data(ttl=CACHE_TTL)
def get_player_id(player_name):
    result = players.find_players_by_full_name(player_name)
    if result:
        return result[0]['id']
    st.warning(f"Player '{player_name}' not found.")
    return None

@st.cache_data(ttl=CACHE_TTL)
def fetch_player_gamelog(player_id, season):
    try:
        return playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=60).get_data_frames()[0]
    except Exception as e:
        st.error(f"Error fetching game log for season {season}: {e}")
        return None

def get_player_data(player_name):
    player_id = get_player_id(player_name)
    if not player_id:
        return None
    gamelogs = []
    for season in [SEASON_CURRENT, SEASON_PREVIOUS]:
        df = fetch_player_gamelog(player_id, season)
        if df is not None:
            gamelogs.append(df)
    return pd.concat(gamelogs, ignore_index=True) if gamelogs else None

def preprocess_game_log(df):
    if df is None or df.empty:
        return None
    # Select relevant features and target
    features = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK']
    target = 'PTS'
    # Create rolling averages for each feature and the target
    windows = [3, 5, 10]
    for col in features + [target]:
        for w in windows:
            df[f'{col}_roll_{w}'] = df[col].rolling(window=w, min_periods=1).mean()
    return df.dropna()

def main():
    st.sidebar.header('Team Selection')
    team_list = get_team_abbreviations()
    selected_team = st.sidebar.selectbox('Select Team', team_list)
    
    roster = get_team_roster(selected_team)
    if not roster:
        st.error("No roster available for the selected team.")
        return
    
    selected_player = st.sidebar.selectbox('Select Player', roster)
    if st.sidebar.button('Analyze Player'):
        with st.spinner('Fetching player data...'):
            player_df = get_player_data(selected_player)
        if player_df is None or player_df.empty:
            st.error("No data found for the player.")
            return
        
        processed_df = preprocess_game_log(player_df)
        if processed_df is None or processed_df.empty:
            st.error("Not enough data after processing.")
            return
        
        # Display recent performance
        st.subheader('Recent Performance')
        recent_games = processed_df.head(5)[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'AST', 'REB']]
        st.dataframe(recent_games)
        
        # Display aggregated statistics
        st.subheader('Player and Team Statistics')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Team Statistics")
            team_stats = {
                'Points Per Game': processed_df['PTS'].mean(),
                'Assists Per Game': processed_df['AST'].mean(),
                'Rebounds Per Game': processed_df['REB'].mean(),
                'Field Goal %': processed_df['FG_PCT'].mean() * 100,
                '3PT %': processed_df['FG3_PCT'].mean() * 100
            }
            for stat, value in team_stats.items():
                st.metric(stat, f"{value:.1f}")
        with col2:
            st.markdown("#### Player Impact")
            impact_metrics = {
                'Usage Rate': processed_df['MIN'].mean(),
                'Offensive Rating': (processed_df['FGM'].sum() / processed_df['FGA'].sum()) * 100 if processed_df['FGA'].sum() != 0 else 0,
                'Defensive Impact': processed_df['STL'].mean() + processed_df['BLK'].mean(),
                'Efficiency': (processed_df['PTS'].sum() / processed_df['MIN'].sum()) * 48 if processed_df['MIN'].sum() != 0 else 0
            }
            for metric, value in impact_metrics.items():
                st.metric(metric, f"{value:.1f}")

if __name__ == '__main__':
    main()
