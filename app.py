import streamlit as st
import pandas as pd
from model1 import (
    engineer_player_features,
    create_player_history_features,
    add_match_context_features,
    prepare_final_features,
    build_train_and_evaluate_model,
    predict_player_fantasy_points
)

st.title("🏏 Fantasy Points Predictor")

# Load and prepare full data
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("cricinfo.csv", quotechar='"', encoding='utf-8', on_bad_lines='warn')
    numeric_cols = ['series_id', 'match_id', 'innings_number', 'ball_number', 'bowler_id', 'striker_id', 'non_striker_id', 'runs_scored']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['wickets'] = df['wickets'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
    df.dropna(inplace=True)

    # players
    batters = df[['striker_id', 'striker_name']].rename(columns={'striker_id': 'player_id', 'striker_name': 'player_name'})
    bowlers = df[['bowler_id', 'bowler_name']].rename(columns={'bowler_id': 'player_id', 'bowler_name': 'player_name'})
    players_df = pd.concat([batters, bowlers], ignore_index=True).drop_duplicates().dropna()
    players_df['team_id'] = 0
    players_df['player_role'] = 'All-rounder'

    # matches
    matches_df = df[['match_id']].drop_duplicates()
    matches_df['match_name'] = 'Match #' + matches_df['match_id'].astype(str)
    matches_df['match_date'] = '2023-01-01'
    matches_df['team1_id'] = 1
    matches_df['team2_id'] = 2

    tables = {
        'commentary': df,
        'players': players_df,
        'matches': matches_df
    }

    player_features = engineer_player_features(tables)
    player_history = create_player_history_features(player_features)
    player_context = add_match_context_features(player_history, tables)
    X_full, y = prepare_final_features(player_context)
    model, encoders, _ = build_train_and_evaluate_model(X_full, y)

    return X_full, model, encoders

X_full, model, encoders = load_and_prepare_data()

# UI: team selection
unique_players = sorted(X_full['player_name'].unique())

st.header("Team A Playing XI")
team_a_xi = st.multiselect("Select 11 players for Team A", unique_players, max_selections=11)

st.header("Team B Playing XI")
team_b_xi = st.multiselect("Select 11 players for Team B", unique_players, max_selections=11)

col1, col2 = st.columns(2)
with col1:
    batting_first = st.selectbox("Who is batting first?", ["Team A", "Team B"])
with col2:
    toss_winner = st.selectbox("Who won the toss?", ["Team A", "Team B"])

group_id = st.text_input("Enter Group or Match Name", "Qualifier 1")

# Prediction
if st.button("Predict Fantasy Points"):
    if len(team_a_xi) != 11 or len(team_b_xi) != 11:
        st.error("❌ Please select 11 players for both Team A and Team B.")
    else:
        innings_map = {"Team A": 1, "Team B": 2}
        all_players = team_a_xi + team_b_xi
        input_df = X_full[X_full['player_name'].isin(all_players)].copy()

        input_df['innings_number'] = input_df['player_name'].apply(
            lambda name: innings_map[batting_first] if name in all_players else 0
        )
        input_df['toss_winner'] = toss_winner
        input_df['group_id'] = group_id

        if input_df.empty:
            st.error("❌ None of the selected players were found in data.")
        else:
            pred_df = predict_player_fantasy_points(model, encoders, input_df)
            pred_df = pred_df[['player_name', 'innings_number', 'predicted_fantasy_points']]
            pred_df = pred_df.sort_values(by='predicted_fantasy_points', ascending=False).reset_index(drop=True)

            st.success("✅ Fantasy points predicted!")
            st.dataframe(pred_df)

            pred_df.to_csv("match_predictions.csv", index=False)
            st.download_button("Download CSV", data=pred_df.to_csv(index=False), file_name="match_predictions.csv")
