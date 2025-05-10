# 🏏 Fantasy Points Predictor using AI & Cricket Data

A complete machine learning pipeline that transforms raw cricket commentary into fantasy point predictions — built using Python, scikit-learn, and Streamlit.

## 📌 Overview

This project processes 3M+ rows of ball-by-ball commentary to:  
- Calculate player-level fantasy points  
- Train a predictive model on historical data  
- Provide a web app to forecast fantasy points for future matches  

## 🚀 Features

- End-to-end ML pipeline: data → features → model → predictions  
- Gradient Boosting Regressor to predict fantasy points  
- Custom cricket-based scoring logic for batting & bowling  
- Feature importance, evaluation plots, and downloadable results  
- Streamlit app for entering teams and seeing predictions in real time  

## 📂 Project Structure

```
├── cricinfo.csv              # Main dataset (ball-by-ball commentary)
├── model1.py                 # ML pipeline (features + training)
├── train_cricinfo_model.py   # Trains the model from full dataset
├── fantasy_model.pkl         # Saved model for reuse
├── app_with_model.py         # Streamlit prediction UI
├── convert_sql_to_csv.py     # Converts .sql INSERTs to usable CSV
├── feature_importance.csv    # Output from training
├── player_predictions.csv    # Fantasy point predictions
```

## 🛠️ Setup

```bash
git clone https://github.com/yourusername/fantasy-points-predictor
cd fantasy-points-predictor
pip install -r requirements.txt
pip install streamlit joblib
```

## 🔁 Add New Data

```bash
python convert_sql_to_csv.py  # If you have new .sql dump
```

Then append new data to existing:

```python
import pandas as pd
old = pd.read_csv("cricinfo.csv")
new = pd.read_csv("new_data.csv")
pd.concat([old, new]).to_csv("cricinfo.csv", index=False)
```

## 📈 Train the Model

```bash
python train_cricinfo_model.py
```

This outputs:  
- fantasy_model.pkl  
- feature_importance.csv  
- player_predictions.csv  

## 🧪 Predict Fantasy Points (Streamlit App)

```bash
streamlit run app_with_model.py
```

- Select Team A & B playing XI  
- Choose toss winner and batting order  
- See fantasy point predictions instantly  
- Option to download results as CSV  

## 🧠 Fantasy Scoring Logic

**Batting:**  
- +1 per run  
- +2 per four  
- +6 per six  
- +20 for 50s, +40 for 100s  
- -5 for duck  

**Bowling:**  
- +25 per wicket  
- +4 per dot ball  
- +8 per maiden over  
- +4 / +8 / +16 for 3 / 4 / 5-wicket hauls  

## 🧠 Model Summary

- Model: GradientBoostingRegressor  
- Features: runs, wickets, economy, strike rate, form, rolling averages  
- Training: GridSearchCV (3-fold)  
- Metrics: MAE, RMSE, R²  
- Output files: feature_importance.csv, predictions_vs_actual.png  

## 👩‍💻 Author

**Ragavi Muthukrishnan**  

## 📜 License

MIT License
