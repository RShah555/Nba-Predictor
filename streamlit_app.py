import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from requests.exceptions import ReadTimeout, ConnectionError
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt
import xgboost as xgb

# Set page config
st.set_page_config(
    page_title="NBA Game Predictions",
    page_icon="🏀",
    layout="wide"
)

# Add CSS styling
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">NBA Player Points Predictions 🏀</p>', unsafe_allow_html=True)

# Constants
SEASON_CURRENT = '2024-25'
SEASON_PREVIOUS = '2023-24'
MAX_RETRIES = 3
CACHE_TTL = 3600  # Cache TTL in seconds

# Helper Functions
def get_team_abbreviations():
    """Retrieve a list of team abbreviations."""
    return [team['abbreviation'] for team in teams.get_teams()]

def get_team_by_abbreviation(abbreviation):
    """Retrieve team information by its abbreviation."""
    return teams.find_team_by_abbreviation(abbreviation)

@st.cache_data(ttl=CACHE_TTL)
def get_team_roster(team_abbreviation):
    """Fetch the roster for a given team."""
    try:
        team_info = get_team_by_abbreviation(team_abbreviation)
        if not team_info:
            st.error(f"Team '{team_abbreviation}' not found.")
            return []

        team_id = team_info['id']
        roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
        player_names = roster['PLAYER'].tolist()
        return player_names
    except Exception as e:
        st.error(f"Error fetching roster for team '{team_abbreviation}': {e}")
        return []

@st.cache_data(ttl=CACHE_TTL)
def get_player_id(player_name):
    """Fetch the player ID given the full name."""
    try:
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            st.warning(f"Player '{player_name}' not found.")
            return None
        return player_dict[0]['id']
    except Exception as e:
        st.error(f"Error fetching player ID for '{player_name}': {e}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def fetch_player_gamelog(player_id, season):
    """Fetch the game log for a player for a given season."""
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=60).get_data_frames()[0]
        return gamelog
    except Exception as e:
        st.error(f"Error fetching game log for player ID '{player_id}' in season '{season}': {e}")
        return None

def get_player_data(player_name, max_retries=MAX_RETRIES):
    """Retrieve and combine game logs for current and previous seasons for a player."""
    player_id = get_player_id(player_name)
    if not player_id:
        return None

    gamelogs = []
    seasons = [SEASON_CURRENT, SEASON_PREVIOUS]

    for season in seasons:
        for attempt in range(max_retries):
            try:
                gamelog = fetch_player_gamelog(player_id, season)
                if gamelog is not None:
                    gamelogs.append(gamelog)
                break
            except (ReadTimeout, ConnectionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    st.error(f"Failed to fetch data for {player_name} in season {season} after {max_retries} attempts.")
                    return None
    if gamelogs:
        combined_data = pd.concat(gamelogs, ignore_index=True)
        return combined_data
    return None

def fetch_all_player_data(roster):
    """Fetch game data for all players in the roster concurrently."""
    player_data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_player = {executor.submit(get_player_data, player): player for player in roster}
        for future in as_completed(future_to_player):
            player = future_to_player[future]
            try:
                data = future.result()
                if data is not None and not data.empty:
                    player_data[player] = data
                else:
                    st.warning(f"No data available for player: {player}")
            except Exception as e:
                st.error(f"Error fetching data for player '{player}': {e}")
    return player_data

def preprocess_game_log(game_log):
    """Preprocess game log data for model training."""
    if game_log is None or game_log.empty:
        return None

    # Select relevant features
    features = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK']
    target = 'PTS'

    # Create rolling averages
    rolling_windows = [3, 5, 10]
    for feature in features + [target]:
        for window in rolling_windows:
            game_log[f'{feature}_rolling_{window}'] = game_log[feature].rolling(window=window, min_periods=1).mean()

    # Drop rows with missing values
    game_log = game_log.dropna()

    return game_log

def train_models(X, y):
    """Train multiple models and return the best one."""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    best_model = None
    best_score = float('-inf')
    results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        results[name] = {
            'model': model,
            'score': score,
            'predictions': model.predict(X_test),
            'y_test': y_test
        }
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, results

# Main App Layout
def main():
    # Sidebar
    st.sidebar.header('Team Selection')
    team_abbreviations = get_team_abbreviations()
    selected_team = st.sidebar.selectbox('Select Team', team_abbreviations)

    # Get team roster
    roster = get_team_roster(selected_team)
    if not roster:
        st.error("No roster data available for the selected team.")
        return

    # Player selection
    selected_player = st.sidebar.selectbox('Select Player', roster)

    if st.sidebar.button('Analyze Player'):
        with st.spinner('Fetching player data...'):
            player_data = get_player_data(selected_player)

            if player_data is not None and not player_data.empty:
                # Preprocess data
                processed_data = preprocess_game_log(player_data)

                if processed_data is not None and not processed_data.empty:
                    # Display recent performance
                    st.subheader('Recent Performance')
                    recent_games = processed_data.head(5)[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'AST', 'REB']]
                    st.dataframe(recent_games)

                    # Prepare data for modeling
                    features = [col for col in processed_data.columns if '_rolling_' in col]
                    X = processed_data[features]
                    y = processed_data['PTS']

                    # Train models
                    with st.spinner('Training models...'):
                        best_model, model_results = train_models(X, y)

                        # Display team comparison
                        st.subheader('Team Performance Analysis')
                        team_stats_col1, team_stats_col2 = st.columns(2)
                        
                        with team_stats_col1:
                            st.markdown('### Team Statistics')
                            team_stats = {
                                'Points Per Game': processed_data['PTS'].mean(),
                                'Assists Per Game': processed_data['AST'].mean(),
                                'Rebounds Per Game': processed_data['REB'].mean(),
                                'Field Goal %': processed_data['FG_PCT'].mean() * 100,
                                '3PT %': processed_data['FG3_PCT'].mean() * 100
                            }
                            for stat_name, stat_value in team_stats.items():
                                st.metric(stat_name, f"{stat_value:.1f}")
                        
                        with team_stats_col2:
                            st.markdown('### Player Impact')
                            impact_metrics = {
                                'Usage Rate': processed_data['MIN'].mean(),
                                'Offensive Rating': processed_data['FGM'].sum() / processed_data['FGA'].sum() * 100,
                                'Defensive Impact': (processed_data['STL'].mean() + processed_data['BLK'].mean()),
                                'Efficiency': processed_data['PTS'].sum() / processed_data['MIN'].sum() * 48
                            }
                            for metric_name, metric_value in impact_metrics.items():
                                st.metric(metric_name, f"{metric_value:.1f}")

                        # Visualization
                        st.subheader('Points Prediction vs Actual')
                        best_result = model_results[max(model_results.items(), key=lambda x: x[1]['score'])[0]]
                        chart_data = pd.DataFrame({
                            'Actual': best_result['y_test'],
                            'Predicted': best_result['predictions']
                        }).reset_index()
                        
                        chart = alt.Chart(chart_data).mark_line(point=True).encode(
                            x=alt.X('index:Q', title='Game Number'),
                            y=alt.Y('value:Q', title='Points', scale=alt.Scale(zero=False)),
                            color=alt.Color('variable:N', title='Type'),
                            tooltip=['index:Q', 'value:Q', 'variable:N']
                        ).transform_fold(
                            ['Actual', 'Predicted'],
                            as_=['variable', 'value']
                        ).properties(
                            width=800,
                            height=400,
                            title='Player Points: Predicted vs Actual'
                        ).configure_axis(
                            labelFontSize=12,
                            titleFontSize=14
                        )
                        st.altair_chart(chart)

if __name__ == '__main__':
    main()