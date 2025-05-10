# train_sample_cricinfo_model.py

import pandas as pd
from model1 import (
    engineer_player_features,
    create_player_history_features,
    add_match_context_features,
    prepare_final_features,
    build_train_and_evaluate_model,
    predict_player_fantasy_points
)

# --- Use built-in sample data ---
print("📥 Using built-in sample data from model1.py...")
from model1 import load_sample_data
tables = load_sample_data()

# --- Pipeline ---
print("⚙️ Running feature engineering...")
player_features = engineer_player_features(tables)
player_history = create_player_history_features(player_features)
player_context = add_match_context_features(player_history, tables)

print("📊 Preparing features...")
X, y = prepare_final_features(player_context)

print("🤖 Training model on sample data...")
model, encoders, feature_importances = build_train_and_evaluate_model(X, y)

print("💾 Saving results...")
feature_importances.to_csv("sample_feature_importance.csv", index=True)
predictions = predict_player_fantasy_points(model, encoders, X)
predictions.to_csv("sample_player_predictions.csv", index=False)

print("\n✅ Sample training complete!")
print(f"🔢 Players: {X['player_id'].nunique()} | Matches: {X['match_id'].nunique()}")
print("📂 Output: sample_feature_importance.csv, sample_player_predictions.csv")
