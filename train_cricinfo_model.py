# train_cricinfo_model.py

import pandas as pd
from model1 import (
    engineer_player_features,
    create_player_history_features,
    add_match_context_features,
    prepare_final_features,
    build_train_and_evaluate_model,
    predict_player_fantasy_points
)

# Step 1: Load the CSV safely
print("📥 Loading commentary data from cricinfo.csv...")
commentary_df = pd.read_csv(
    "cricinfo.csv",
    quotechar='"',
    encoding='utf-8',
    on_bad_lines='warn'  # For Python >= 3.10. Use error_bad_lines=False for older versions.
)

# Step 2: Convert columns to numeric
print("🔢 Converting numeric fields...")
numeric_columns = [
    'series_id', 'match_id', 'innings_number', 'ball_number',
    'bowler_id', 'striker_id', 'non_striker_id', 'runs_scored'
]

for col in numeric_columns:
    commentary_df[col] = pd.to_numeric(commentary_df[col], errors='coerce')

# Step 3: Convert 'wickets' from 'YES'/'NO' to 1/0
print("⚙️ Cleaning 'wickets' column...")
commentary_df['wickets'] = commentary_df['wickets'].apply(
    lambda x: 1 if str(x).strip().upper() == 'YES' else 0
)

# Step 4: Drop rows with missing critical fields
commentary_df.dropna(subset=['match_id', 'striker_id', 'runs_scored'], inplace=True)

# Step 5: Generate players and matches info
print("🧱 Creating players and matches tables...")
batters = commentary_df[['striker_id', 'striker_name']].rename(columns={'striker_id': 'player_id', 'striker_name': 'player_name'})
bowlers = commentary_df[['bowler_id', 'bowler_name']].rename(columns={'bowler_id': 'player_id', 'bowler_name': 'player_name'})
players_df = pd.concat([batters, bowlers], ignore_index=True).drop_duplicates().dropna()
players_df['team_id'] = 1  # Placeholder
players_df['player_role'] = 'All-rounder'  # Placeholder

matches_df = commentary_df[['match_id']].drop_duplicates()
matches_df['match_name'] = 'Match #' + matches_df['match_id'].astype(str)
matches_df['match_date'] = '2023-01-01'  # Placeholder
matches_df['team1_id'] = 1
matches_df['team2_id'] = 2

tables = {
    'commentary': commentary_df,
    'players': players_df,
    'matches': matches_df
}

# Step 6: Run full model pipeline
print("⚙️ Running feature engineering...")
player_features = engineer_player_features(tables)
player_history = create_player_history_features(player_features)
player_context = add_match_context_features(player_history, tables)

print("📊 Preparing features...")
X, y = prepare_final_features(player_context)

print("🤖 Training model...")
model, encoders, feature_importances = build_train_and_evaluate_model(X, y)

# Step 7: Save outputs
print("💾 Saving results...")
feature_importances.to_csv("feature_importance.csv", index=True)
predictions = predict_player_fantasy_points(model, encoders, X)
predictions.to_csv("player_predictions.csv", index=False)

# Summary
print("\n✅ Model training complete!")
print(f"🔢 Total players analyzed: {X['player_id'].nunique()}")
print(f"📅 Total matches analyzed: {X['match_id'].nunique()}")
print("📂 Output saved: feature_importance.csv, player_predictions.csv")
