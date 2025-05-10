import pandas as pd
from model1 import (
    engineer_player_features,
    create_player_history_features,
    add_match_context_features,
    prepare_final_features,
    predict_player_fantasy_points,
    build_train_and_evaluate_model
)

# Load data
print("📥 Loading full cricinfo.csv...")
commentary_df = pd.read_csv("cricinfo.csv", quotechar='"', encoding='utf-8', on_bad_lines='warn')

# Convert data types
numeric_columns = [
    'series_id', 'match_id', 'innings_number', 'ball_number',
    'bowler_id', 'striker_id', 'non_striker_id', 'runs_scored'
]
for col in numeric_columns:
    commentary_df[col] = pd.to_numeric(commentary_df[col], errors='coerce')
commentary_df['wickets'] = commentary_df['wickets'].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
commentary_df.dropna(inplace=True)

# Generate synthetic players/matches tables
batters = commentary_df[['striker_id', 'striker_name']].rename(columns={'striker_id': 'player_id', 'striker_name': 'player_name'})
bowlers = commentary_df[['bowler_id', 'bowler_name']].rename(columns={'bowler_id': 'player_id', 'bowler_name': 'player_name'})
players_df = pd.concat([batters, bowlers]).drop_duplicates().dropna()
players_df['team_id'] = 1
players_df['player_role'] = 'All-rounder'

matches_df = commentary_df[['match_id']].drop_duplicates()
matches_df['match_name'] = 'Match #' + matches_df['match_id'].astype(str)
matches_df['match_date'] = '2023-01-01'
matches_df['team1_id'] = 1
matches_df['team2_id'] = 2

tables = {
    'commentary': commentary_df,
    'players': players_df,
    'matches': matches_df
}

# Rebuild training data & model
print("⚙️ Rebuilding model on full data...")
player_features = engineer_player_features(tables)
player_history = create_player_history_features(player_features)
player_context = add_match_context_features(player_history, tables)

X_full, y = prepare_final_features(player_context)
model, encoders, _ = build_train_and_evaluate_model(X_full, y)

# ------------------ Prediction Input ------------------

# 👇 Replace this list with actual player names or IDs (striker_name or player_name)
playing_xi = [
    "P Jayeshbhai Patel", "Meet Ahir", "Breeze Ahir",
    "D Bourne", "R Johnson", "Sohel Patel"
]

# Filter players in current prediction set
input_df = X_full[X_full['player_name'].isin(playing_xi)]

if input_df.empty:
    print("❌ No matching players found in dataset. Check names.")
else:
    predictions = predict_player_fantasy_points(model, encoders, input_df)
    predictions = predictions[['player_name', 'predicted_fantasy_points']].sort_values(by='predicted_fantasy_points', ascending=False)

    # Output
    print("\n🏏 Fantasy Point Predictions for Playing XI:")
    print(predictions.to_string(index=False))

    predictions.to_csv("predicted_playing_xi.csv", index=False)
    print("\n✅ Saved as predicted_playing_xi.csv")
