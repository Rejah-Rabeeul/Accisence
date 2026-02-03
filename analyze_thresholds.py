import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np

# --- Configuration ---
INPUT_FILE = "kozhikode_accident_history.csv"
MODEL_FILE = "accident_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def analyze_thresholds():
    print("Loading data and model...")
    df = pd.read_csv(INPUT_FILE)
    
    le_weather = LabelEncoder()
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    
    features = ['curvature_score', 'maxspeed', 'is_junction', 'is_night', 
                'weather_encoded', 'hour_of_day', 'is_holiday']
    
    X = df[features]
    y = df['is_accident']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    model = joblib.load(MODEL_FILE)
    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    print(f"{'Threshold':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 60)

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'threshold': t,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        print(f"{t:<10.2f} | {acc:<10.2%} | {prec:<10.2%} | {rec:<10.2%} | {f1:<10.4f}")

    # Special check for Top-5 Spots on "Safe" subsets
    # Let's find cases where actual is 0 (Safe trips) and see what the max prob is.
    safe_trips_prob = y_prob[y_test == 0]
    percentage_blocked = (safe_trips_prob < 0.35).mean()
    print(f"\nSAFETY FILTER EFFECT:")
    print(f"By using 0.35 threshold, we SILENCE alerts for {percentage_blocked:.2%} of safe road segments.")
    print("This directly answers the 'Accuracy Issue' for safe routes.")

if __name__ == "__main__":
    analyze_thresholds()
