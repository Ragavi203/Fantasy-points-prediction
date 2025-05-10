import pandas as pd
import numpy as np
import sqlite3
import os
import time
import gc
from tqdm import tqdm
import psutil

def get_memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def load_large_sql_optimized(sql_file_path, output_db_path='temp_cricket.db', tables_to_load=None):
    """
    Optimized function to load large SQL file into SQLite database
    
    Parameters:
    - sql_file_path: Path to the SQL dump file
    - output_db_path: Path to create SQLite database
    - tables_to_load: List of table names to load (None = all tables)
    
    Returns:
    - conn: SQLite connection
    """
    print(f"Memory before loading SQL: {get_memory_usage():.2f} MB")
    start_time = time.time()
    
    # Check if the SQLite DB already exists
    if os.path.exists(output_db_path):
        print(f"Using existing SQLite database: {output_db_path}")
        return sqlite3.connect(output_db_path)
    
    # Create SQLite database
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    
    print(f"Processing SQL file: {sql_file_path}")
    print(f"This might take several minutes for a 500MB file...")
    
    # Process SQL file in chunks to reduce memory usage
    current_statement = ""
    current_table = None
    table_count = 0
    insert_count = 0
    
    # Helper to execute a statement and handle errors
    def safe_execute(stmt, is_create=False):
        nonlocal table_count, insert_count
        try:
            cursor.execute(stmt)
            if is_create:
                table_count += 1
            else:
                insert_count += 1
            return True
        except sqlite3.Error as e:
            if "syntax error" not in str(e).lower():  # Ignore common syntax differences
                print(f"Error executing statement: {str(e)}")
            return False
    
    # Process file
    with open(sql_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for idx, line in enumerate(tqdm(file, desc="Processing SQL lines")):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('--') or line.startswith('/*'):
                continue
            
            # Detect table creation
            if line.upper().startswith('CREATE TABLE'):
                table_name = None
                if '`' in line:
                    # Extract name from `table_name`
                    parts = line.split('`')
                    if len(parts) >= 2:
                        table_name = parts[1]
                else:
                    # Extract name from CREATE TABLE table_name
                    parts = line.upper().split('CREATE TABLE ')
                    if len(parts) >= 2:
                        table_name = parts[1].split('(')[0].strip()
                
                if table_name:
                    current_table = table_name
                    # Skip if we're only loading specific tables
                    if tables_to_load and current_table not in tables_to_load:
                        current_table = None
                        continue
            
            # Skip lines for tables we're not interested in
            if tables_to_load and current_table not in tables_to_load and not line.upper().startswith('CREATE TABLE'):
                continue
                
            # Add line to current statement
            current_statement += " " + line
            
            # Execute statement when complete
            if line.endswith(';'):
                statement_type = "unknown"
                
                if current_statement.upper().strip().startswith('CREATE TABLE'):
                    statement_type = "create"
                    safe_execute(current_statement, is_create=True)
                    if table_count % 10 == 0:
                        print(f"Created {table_count} tables...")
                        
                elif current_statement.upper().strip().startswith('INSERT INTO'):
                    statement_type = "insert"
                    safe_execute(current_statement)
                    if insert_count % 10000 == 0:
                        print(f"Processed {insert_count} inserts... Memory: {get_memory_usage():.2f} MB")
                        conn.commit()  # Periodic commit to save progress
                
                # Clear for next statement
                current_statement = ""
    
    # Final commit
    conn.commit()
    
    # Report stats
    duration = time.time() - start_time
    print(f"SQL processing completed in {duration:.2f} seconds")
    print(f"Created {table_count} tables")
    print(f"Processed {insert_count} insert statements")
    print(f"Final memory usage: {get_memory_usage():.2f} MB")
    
    return conn

def get_table_info(conn):
    """Get information about tables in the database"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    table_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        table_info[table_name] = {
            'row_count': row_count,
            'columns': column_names
        }
    
    return table_info

def load_table_chunk(conn, table_name, chunk_size=100000, where_clause=""):
    """Load a table in chunks to avoid memory issues"""
    query = f"SELECT * FROM {table_name} {where_clause}"
    
    for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
        yield chunk

def process_commentary_in_chunks(conn, chunk_size=100000):
    """Process commentary data in chunks and calculate fantasy points"""
    print("Processing commentary data in chunks...")
    
    # Get all match IDs
    match_ids = pd.read_sql_query(
        "SELECT DISTINCT match_id FROM commentary", 
        conn
    )['match_id'].tolist()
    
    all_fantasy_points = []
    
    # Process by match to reduce memory usage
    for match_id in tqdm(match_ids, desc="Processing matches"):
        # Load commentary for this match
        match_commentary = pd.read_sql_query(
            f"SELECT * FROM commentary WHERE match_id = {match_id}",
            conn
        )
        
        # Calculate fantasy points for this match
        fantasy_points = calculate_fantasy_points_for_match(match_commentary)
        all_fantasy_points.append(fantasy_points)
        
        # Force garbage collection
        del match_commentary
        gc.collect()
    
    # Combine all fantasy points
    return pd.concat(all_fantasy_points, ignore_index=True)

def calculate_fantasy_points_for_match(match_commentary):
    """Calculate fantasy points for a specific match"""
    # Similar to the original calculate_fantasy_points function
    # but optimized for single match processing
    
    # Group data by player
    batsman_stats = match_commentary.groupby('striker_id').agg(
        runs=('runs_scored', 'sum'),
        balls_faced=('ball_number', 'count'),
        fours=('runs_scored', lambda x: (x == 4).sum()),
        sixes=('runs_scored', lambda x: (x == 6).sum()),
        got_out=('wickets', lambda x: (x == 'YES').sum())
    ).reset_index()
    
    bowler_stats = match_commentary.groupby('bowler_id').agg(
        overs_bowled=('ball_number', lambda x: len(x) / 6),  # Approximation
        wickets=('wickets', lambda x: (x == 'YES').sum()),
        runs_conceded=('runs_scored', 'sum'),
        dot_balls=('runs_scored', lambda x: (x == 0).sum())
    ).reset_index()
    
    # Calculate maiden overs (approximation)
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
    
    # Add match_id to both dataframes
    match_id = match_commentary['match_id'].iloc[0]
    batsman_stats['match_id'] = match_id
    bowler_stats['match_id'] = match_id
    
    # Merge stats
    all_stats = pd.merge(batsman_stats[['player_id', 'match_id', 'batting_points']], 
                     bowler_stats[['player_id', 'match_id', 'bowling_points']], 
                     on=['player_id', 'match_id'], how='outer').fillna(0)
    
    # Total fantasy points
    all_stats['total_fantasy_points'] = all_stats['batting_points'] + all_stats['bowling_points']
    
    return all_stats

def main_optimized(sql_file_path="cricinfo.sql", output_dir="outputs"):
    """
    Optimized main function for large SQL files
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data into SQLite database
    sqlite_db_path = "temp_cricket.db"
    conn = load_large_sql_optimized(sql_file_path, sqlite_db_path)
    
    # Get info about tables
    table_info = get_table_info(conn)
    print("\nDatabase Tables:")
    for table, info in table_info.items():
        print(f"- {table}: {info['row_count']} rows")
    
    # Process tables in memory-efficient way
    if 'commentary' in table_info:
        # Process commentary in chunks by match
        fantasy_points = process_commentary_in_chunks(conn)
        
        # Save fantasy points
        fantasy_points.to_csv(f"{output_dir}/fantasy_points.csv", index=False)
        print(f"Fantasy points saved to {output_dir}/fantasy_points.csv")
        
        # Proceed with feature engineering and model training
        # (This part would be similar to the original code but adapted to work with chunks)
    else:
        print("No commentary table found in the database.")
    
    return conn

if __name__ == "__main__":
    # Replace with your actual SQL file path
    sql_file_path = "cricinfo.sql"
    
    # Run the optimized system
    conn = main_optimized(sql_file_path)