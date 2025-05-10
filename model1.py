import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------
# 1. Data Loading Functions
# -------------------------------------------

def load_data_from_sql(sql_file_path, chunk_size=10000):
    """
    Load data from large SQL file in chunks to avoid memory issues
    Returns a dictionary of DataFrames, one for each table
    """
    print(f"Loading data from {sql_file_path}...")
    
    # Create a temporary SQLite database in memory
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Read SQL file in chunks to handle large files
    tables = {}
    current_table = None
    create_statement = ""
    
    with open(sql_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('--'):
                continue
                
            # Detect table creation
            if line.upper().startswith('CREATE TABLE'):
                table_name = line.split('`')[1] if '`' in line else line.split('TABLE ')[1].split(' ')[0]
                current_table = table_name
                create_statement = line
                continue
                
            # Add lines to create statement until complete
            if current_table and ';' not in line:
                create_statement += " " + line
                continue
                
            # Execute create statement when complete
            if current_table and ';' in line:
                create_statement += " " + line
                try:
                    cursor.execute(create_statement)
                    print(f"Created table: {current_table}")
                    tables[current_table] = pd.DataFrame()
                    current_table = None
                    create_statement = ""
                except Exception as e:
                    print(f"Error creating table {current_table}: {str(e)}")
                continue
                
            # Process INSERT statements
            if line.upper().startswith('INSERT INTO'):
                table_name = line.split('`')[1] if '`' in line else line.split('INTO ')[1].split(' ')[0]
                try:
                    cursor.execute(line)
                except Exception as e:
                    print(f"Error inserting into {table_name}: {str(e)}")
    
    # Commit changes
    conn.commit()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [table[0] for table in cursor.fetchall()]
    
    # Load each table into DataFrame
    for table in table_names:
        try:
            tables[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            print(f"Loaded table {table} with {len(tables[table])} rows")
        except Exception as e:
            print(f"Error loading table {table}: {str(e)}")
    
    conn.close()
    return tables

def load_sample_data():
    """
    For testing when SQL file is not available - creates DataFrame from provided sample
    """
    # Sample data from commentary table
    commentary_data = [
        [1480822, 1481296, 1, 0.1, 119816, 'D Bourne', None, None, 105175, 'P Jayeshbhai Patel', None, None, None, 2, 'NO', 'NO'],
        [1480822, 1481296, 1, 0.1, 119816, 'D Bourne', None, None, 119827, 'Meet Ahir', 105175, None, None, 1, 'NO', 'NO'],
        [1480822, 1481296, 1, 0.2, 119816, 'D Bourne', None, None, 105175, 'P Jayeshbhai Patel', 119827, None, None, 0, 'NO', 'NO'],
        [1480822, 1481296, 1, 0.3, 119816, 'D Bourne', None, None, 105175, 'P Jayeshbhai Patel', 119827, None, None, 0, 'NO', 'NO'],
        [1480822, 1481296, 1, 0.4, 119816, 'D Bourne', None, None, 105175, 'P Jayeshbhai Patel', 119827, None, None, 0, 'NO', 'NO'],
        [1480822, 1481296, 1, 0.5, 119816, 'D Bourne', None, None, 105175, 'P Jayeshbhai Patel', 119827, None, None, 0, 'NO', 'NO'],
        [1480822, 1481296, 1, 0.6, 119816, 'D Bourne', None, None, 105175, 'P Jayeshbhai Patel', 119827, None, None, 1, 'NO', 'NO'],
        [1480822, 1481296, 1, 1.1, 119820, 'R Johnson', None, None, 105175, 'P Jayeshbhai Patel', 119827, None, None, 0, 'YES', 'bowled'],
        [1480822, 1481296, 1, 1.2, 119820, 'R Johnson', None, None, 116173, 'Breeze Ahir', 119827, None, None, 0, 'NO', 'NO'],
        [1480822, 1481296, 1, 1.3, 119820, 'R Johnson', None, None, 116173, 'Breeze Ahir', 119827, None, None, 1, 'NO', 'NO'],
        [1480822, 1481296, 1, 1.4, 119820, 'R Johnson', None, None, 119827, 'Meet Ahir', 116173, None, None, 0, 'NO', 'NO'],
        [1480822, 1481296, 1, 1.5, 119820, 'R Johnson', None, None, 119827, 'Meet Ahir', 116173, None, None, 1, 'NO', 'NO'],
        [1480822, 1481296, 1, 1.6, 119820, 'R Johnson', None, None, 116173, 'Breeze Ahir', 119827, None, None, 2, 'NO', 'NO'],
        [1480822, 1481296, 1, 1.6, 119820, 'R Johnson', None, None, 119827, 'Meet Ahir', 116173, None, None, 1, 'NO', 'NO'],
    ]
    
    commentary_columns = [
        'series_id', 'match_id', 'innings_number', 'ball_number', 'bowler_id', 'bowler_name', 
        'bowling_type', 'pitching_spot', 'striker_id', 'striker_name', 'non_striker_id', 
        'non_striker_name', 'shot_spot', 'runs_scored', 'wickets', 'wicket_type'
    ]
    
    # Create sample match data
    match_data = [
        [1481296, 'Team A vs Team B', '2023-05-01', 101, 102],
        [1481297, 'Team C vs Team D', '2023-05-02', 103, 104],
    ]
    
    match_columns = ['match_id', 'match_name', 'match_date', 'team1_id', 'team2_id']
    
    # Create sample player data
    player_data = [
        [105175, 'P Jayeshbhai Patel', 101, 'Batsman'],
        [119827, 'Meet Ahir', 101, 'All-rounder'],
        [116173, 'Breeze Ahir', 101, 'Batsman'],
        [119816, 'D Bourne', 102, 'Bowler'],
        [119820, 'R Johnson', 102, 'Bowler'],
    ]
    
    player_columns = ['player_id', 'player_name', 'team_id', 'player_role']
    
    tables = {
        'commentary': pd.DataFrame(commentary_data, columns=commentary_columns),
        'matches': pd.DataFrame(match_data, columns=match_columns),
        'players': pd.DataFrame(player_data, columns=player_columns)
    }
    
    print("Loaded sample data")
    for table_name, df in tables.items():
        print(f"Table {table_name}: {len(df)} rows")
    
    return tables

# -------------------------------------------
# 2. Feature Engineering Functions
# -------------------------------------------

def calculate_fantasy_points(commentary_df, player_role='all'):
    """
    Calculate fantasy points for each player based on cricket performance
    
    Standard Fantasy Cricket Points:
    - Batting:
      - 1 point per run
      - 20 bonus for 50s
      - 40 bonus for 100s
      - -5 for duck (0 runs)
      - 2 per boundary (4s)
      - 6 per six (6s)
    - Bowling:
      - 25 per wicket
      - 8 per maiden over
      - 4 per dot ball
      - Bonus for 3/4/5 wickets: 4/8/16
    - Fielding:
      - 8 per catch
      - 12 per stumping/run out
    """
    print("Calculating fantasy points...")
    
    # Group data by player
    batsman_stats = commentary_df.groupby('striker_id').agg(
        runs=('runs_scored', 'sum'),
        balls_faced=('ball_number', 'count'),
        fours=('runs_scored', lambda x: (x == 4).sum()),
        sixes=('runs_scored', lambda x: (x == 6).sum()),
        got_out=('wickets', lambda x: (x == 'YES').sum())
    ).reset_index()
    
    bowler_stats = commentary_df.groupby('bowler_id').agg(
        overs_bowled=('ball_number', lambda x: len(x) / 6),  # Approximation
        wickets=('wickets', lambda x: (x == 'YES').sum()),
        runs_conceded=('runs_scored', 'sum'),
        dot_balls=('runs_scored', lambda x: (x == 0).sum())
    ).reset_index()
    
    # Calculate maiden overs (approximation)
    # A more accurate version would group by over and check if all balls are dots
    bowler_stats['maiden_overs'] = bowler_stats.apply(
        lambda x: int(x['dot_balls'] / 6) if x['dot_balls'] >= 6 else 0, axis=1
    )
    
    # Calculate batting points
    batsman_stats['is_duck'] = (batsman_stats['runs'] == 0) & (batsman_stats['got_out'] > 0)
    batsman_stats['fifty'] = (batsman_stats['runs'] >= 50) & (batsman_stats['runs'] < 100)
    batsman_stats['hundred'] = batsman_stats['runs'] >= 100
    
    batsman_stats['batting_points'] = (
        batsman_stats['runs'] * 1 +
        batsman_stats['fours'] * 2 +
        batsman_stats['sixes'] * 6 +
        batsman_stats['fifty'] * 20 +
        batsman_stats['hundred'] * 40 +
        batsman_stats['is_duck'] * -5
    )
    
    # Calculate bowling points
    bowler_stats['three_wickets'] = (bowler_stats['wickets'] >= 3) & (bowler_stats['wickets'] < 4)
    bowler_stats['four_wickets'] = (bowler_stats['wickets'] >= 4) & (bowler_stats['wickets'] < 5)
    bowler_stats['five_wickets'] = bowler_stats['wickets'] >= 5
    
    bowler_stats['bowling_points'] = (
        bowler_stats['wickets'] * 25 +
        bowler_stats['maiden_overs'] * 8 +
        bowler_stats['dot_balls'] * 4 +
        bowler_stats['three_wickets'] * 4 +
        bowler_stats['four_wickets'] * 8 +
        bowler_stats['five_wickets'] * 16
    )
    
    # Rename ID columns for merging
    batsman_stats = batsman_stats.rename(columns={'striker_id': 'player_id'})
    bowler_stats = bowler_stats.rename(columns={'bowler_id': 'player_id'})
    
    # Merge stats
    all_stats = pd.merge(batsman_stats[['player_id', 'batting_points']], 
                     bowler_stats[['player_id', 'bowling_points']], 
                     on='player_id', how='outer').fillna(0)
    
    # Total fantasy points
    all_stats['total_fantasy_points'] = all_stats['batting_points'] + all_stats['bowling_points']
    
    return all_stats

def engineer_player_features(tables):
    """
    Engineer features for predicting player fantasy points
    """
    print("Engineering player features...")
    
    commentary_df = tables['commentary']
    
    # Extract basic features
    player_features = {}
    
    # 1. Batting features
    batting_stats = commentary_df.groupby(['match_id', 'striker_id']).agg(
        runs=('runs_scored', 'sum'),
        balls_faced=('ball_number', 'count'),
        fours=('runs_scored', lambda x: (x == 4).sum()),
        sixes=('runs_scored', lambda x: (x == 6).sum())
    ).reset_index()
    
    batting_stats['strike_rate'] = (batting_stats['runs'] / batting_stats['balls_faced']) * 100
    batting_stats.loc[batting_stats['balls_faced'] == 0, 'strike_rate'] = 0
    
    # 2. Bowling features
    bowling_stats = commentary_df.groupby(['match_id', 'bowler_id']).agg(
        overs=('ball_number', lambda x: len(x) / 6),  # Approximation
        wickets=('wickets', lambda x: (x == 'YES').sum()),
        runs_conceded=('runs_scored', 'sum'),
        dot_balls=('runs_scored', lambda x: (x == 0).sum())
    ).reset_index()
    
    bowling_stats['economy'] = bowling_stats['runs_conceded'] / bowling_stats['overs']
    bowling_stats.loc[bowling_stats['overs'] == 0, 'economy'] = 0
    
    # 3. Calculate fantasy points
    fantasy_points = calculate_fantasy_points(commentary_df)
    
    # 4. Merge all features
    batting_stats = batting_stats.rename(columns={'striker_id': 'player_id'})
    bowling_stats = bowling_stats.rename(columns={'bowler_id': 'player_id'})
    
    # Create final feature dataframe
    player_match_features = pd.merge(
        batting_stats, 
        bowling_stats,
        on=['match_id', 'player_id'], 
        how='outer'
    ).fillna(0)
    
    # Add fantasy points
    player_match_features = pd.merge(
        player_match_features, 
        fantasy_points[['player_id', 'total_fantasy_points']], 
        on='player_id', 
        how='left'
    ).fillna(0)
    
    # Add player roles if available
    if 'players' in tables:
        player_match_features = pd.merge(
            player_match_features, 
            tables['players'][['player_id', 'player_role']], 
            on='player_id', 
            how='left'
        )
    
    # Add match details if available
    if 'matches' in tables:
        player_match_features = pd.merge(
            player_match_features, 
            tables['matches'][['match_id', 'match_date']], 
            on='match_id', 
            how='left'
        )
        # Convert date to datetime
        player_match_features['match_date'] = pd.to_datetime(player_match_features['match_date'], errors='coerce')
    
    return player_match_features

def create_player_history_features(player_features):
    """
    Create historical features for each player
    """
    print("Creating player history features...")
    
    # Ensure data is sorted by date
    if 'match_date' in player_features.columns:
        player_features = player_features.sort_values(['player_id', 'match_date'])
    
    # Group by player
    player_history = []
    
    for player_id, player_data in player_features.groupby('player_id'):
        # Calculate rolling averages
        player_data = player_data.copy()
        player_data['rolling_avg_runs'] = player_data['runs'].rolling(window=3, min_periods=1).mean()
        player_data['rolling_avg_wickets'] = player_data['wickets'].rolling(window=3, min_periods=1).mean()
        player_data['rolling_avg_fantasy'] = player_data['total_fantasy_points'].rolling(window=3, min_periods=1).mean()
        player_data['rolling_avg_sr'] = player_data['strike_rate'].rolling(window=3, min_periods=1).mean()
        player_data['rolling_avg_economy'] = player_data['economy'].rolling(window=3, min_periods=1).mean()
        
        # Calculate form indicators (trend)
        player_data['form_indicator'] = player_data['total_fantasy_points'].diff().fillna(0)
        player_data['form_direction'] = player_data['form_indicator'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        player_history.append(player_data)
    
    # Combine all players
    player_history_df = pd.concat(player_history, ignore_index=True)
    
    return player_history_df

def add_match_context_features(player_features, tables):
    """
    Add contextual features about the match
    """
    print("Adding match context features...")
    
    if 'matches' not in tables:
        print("Matches table not available, skipping context features")
        return player_features
    
    # Get match data
    matches_df = tables['matches']
    
    # Add team information
    player_features = pd.merge(
        player_features,
        matches_df[['match_id', 'team1_id', 'team2_id']],
        on='match_id',
        how='left'
    )
    
    # Add opponent team information
    if 'players' in tables:
        players_df = tables['players']
        player_features = pd.merge(
            player_features,
            players_df[['player_id', 'team_id']],
            on='player_id',
            how='left'
        )
        
        # Determine opponent team
        player_features['opponent_team_id'] = player_features.apply(
            lambda x: x['team2_id'] if x['team_id'] == x['team1_id'] else x['team1_id'],
            axis=1
        )
    
    return player_features

def prepare_final_features(player_history_df):
    """
    Prepare final features for ML model
    """
    print("Preparing final features...")
    
    # Select relevant features
    feature_cols = [
        'player_id', 'match_id', 'runs', 'balls_faced', 'fours', 'sixes',
        'strike_rate', 'wickets', 'overs', 'runs_conceded', 'dot_balls', 'economy',
        'rolling_avg_runs', 'rolling_avg_wickets', 'rolling_avg_fantasy',
        'rolling_avg_sr', 'rolling_avg_economy', 'form_direction'
    ]
    
    # Add player role if available
    if 'player_role' in player_history_df.columns:
        feature_cols.append('player_role')
    
    # Add team context if available
    if 'team_id' in player_history_df.columns:
        feature_cols.extend(['team_id', 'opponent_team_id'])
    
    # Select columns that exist
    existing_cols = [col for col in feature_cols if col in player_history_df.columns]
    
    X = player_history_df[existing_cols]
    y = player_history_df['total_fantasy_points']
    
    return X, y

# -------------------------------------------
# 3. Model Building Functions
# -------------------------------------------

def encode_categorical_features(X):
    """
    Encode categorical features
    """
    print("Encoding categorical features...")
    
    X_encoded = X.copy()
    
    # Encode player_id
    if 'player_id' in X_encoded.columns:
        player_encoder = LabelEncoder()
        X_encoded['player_id_encoded'] = player_encoder.fit_transform(X_encoded['player_id'])
        X_encoded = X_encoded.drop('player_id', axis=1)
    
    # Encode match_id
    if 'match_id' in X_encoded.columns:
        match_encoder = LabelEncoder()
        X_encoded['match_id_encoded'] = match_encoder.fit_transform(X_encoded['match_id'])
        X_encoded = X_encoded.drop('match_id', axis=1)
    
    # Encode player role
    if 'player_role' in X_encoded.columns:
        role_encoder = LabelEncoder()
        X_encoded['player_role_encoded'] = role_encoder.fit_transform(X_encoded['player_role'])
        X_encoded = X_encoded.drop('player_role', axis=1)
    
    # Encode team_id and opponent_team_id
    if 'team_id' in X_encoded.columns:
        team_encoder = LabelEncoder()
        X_encoded['team_id_encoded'] = team_encoder.fit_transform(X_encoded['team_id'])
        X_encoded = X_encoded.drop('team_id', axis=1)
    
    if 'opponent_team_id' in X_encoded.columns:
        opponent_encoder = LabelEncoder()
        X_encoded['opponent_team_id_encoded'] = opponent_encoder.fit_transform(X_encoded['opponent_team_id'])
        X_encoded = X_encoded.drop('opponent_team_id', axis=1)
    
    # Return encoded features and encoders for future use
    encoders = {
        'player_encoder': player_encoder if 'player_id' in X.columns else None,
        'match_encoder': match_encoder if 'match_id' in X.columns else None,
        'role_encoder': role_encoder if 'player_role' in X.columns else None,
        'team_encoder': team_encoder if 'team_id' in X.columns else None,
        'opponent_encoder': opponent_encoder if 'opponent_team_id' in X.columns else None
    }
    
    return X_encoded, encoders

def build_train_and_evaluate_model(X, y):
    """
    Build, train and evaluate the fantasy points prediction model
    """
    print("Building and training model...")
    
    # Encode categorical features
    X_encoded, encoders = encode_categorical_features(X)
    
    # Fill any remaining NaN values
    X_encoded = X_encoded.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Define model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=42))
    ])
    
    # Define hyperparameters for grid search
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.1, 0.05],
        'model__max_depth': [3, 5]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1
    )
    
    # Train model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Get feature importance
    feature_importances = pd.DataFrame(
        best_model['model'].feature_importances_,
        index=X_encoded.columns,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    print("\nTop Features:")
    print(feature_importances.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    feature_importances.head(15).plot(kind='barh')
    plt.title('Feature Importance for Fantasy Points Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Fantasy Points')
    plt.ylabel('Predicted Fantasy Points')
    plt.title('Fantasy Points: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    
    return best_model, encoders, feature_importances

def predict_player_fantasy_points(model, encoders, player_data):
    """
    Predict fantasy points for new player data
    """
    # Prepare data similarly to training data
    X_new, _ = encode_categorical_features(player_data)
    X_new = X_new.fillna(0)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    # Add predictions to player data
    player_data['predicted_fantasy_points'] = predictions
    
    return player_data

# -------------------------------------------
# 4. Main Execution Function
# -------------------------------------------

def main(sql_file_path=None):
    """
    Main function to run the fantasy points prediction system
    """
    print("Starting Cricket Fantasy Points Prediction System...")
    
    # Load data
    if sql_file_path and os.path.exists(sql_file_path):
        tables = load_data_from_sql(sql_file_path)
    else:
        print("SQL file not provided or not found. Using sample data...")
        tables = load_sample_data()
    
    # Engineer features
    player_features = engineer_player_features(tables)
    player_history = create_player_history_features(player_features)
    player_context = add_match_context_features(player_history, tables)
    
    # Prepare final features
    X, y = prepare_final_features(player_context)
    
    # Build and evaluate model
    model, encoders, feature_importances = build_train_and_evaluate_model(X, y)
    
    print("\nModel training complete!")
    print(f"Total players analyzed: {X['player_id'].nunique() if 'player_id' in X.columns else 'N/A'}")
    print(f"Total matches analyzed: {X['match_id'].nunique() if 'match_id' in X.columns else 'N/A'}")
    
    return model, encoders, feature_importances, player_context

# -------------------------------------------
# 5. Example Usage
# -------------------------------------------

if __name__ == "__main__":
    # Replace with your actual SQL file path
    sql_file_path = "cricket_data.sql"
    
    # Run the system
    model, encoders, feature_importances, player_data = main(sql_file_path)
    
    # Example of making predictions for upcoming matches
    # (This would need actual upcoming match data)
    # upcoming_matches = prepare_upcoming_matches_data()
    # predictions = predict_player_fantasy_points(model, encoders, upcoming_matches)
    # print("Top Fantasy Picks for Upcoming Matches:")
    # print(predictions.sort_values('predicted_fantasy_points', ascending=False).head(10))