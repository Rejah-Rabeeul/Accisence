import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

# --- Configuration ---
INPUT_FILE = "kozhikode_accident_history.csv"
MODEL_FILE = "accident_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def train_model():
    print(f"Loading dataset from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. Data Preprocessing
    print("Preprocessing data...")
    
    # Encoder for categorical variables
    # Weather is a string (Clear, Rain, Fog)
    le_weather = LabelEncoder()
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    
    print("Weather mapping:", dict(zip(le_weather.classes_, le_weather.transform(le_weather.classes_))))
    
    # Feature Selection
    # X features: curvature_score, maxspeed, is_junction, is_night, weather_encoded, hour_of_day, is_holiday
    features = [
        'curvature_score', 
        'maxspeed', 
        'is_junction', 
        'is_night', 
        'weather_encoded', 
        'hour_of_day', 
        'is_holiday'
    ]
    
    X = df[features]
    y = df['is_accident']
    
    print(f"Feature set: {features}")
    print(f"Target distribution:\n{y.value_counts()}")

    # 2. Train-Test Split
    print(f"Splitting data ({1-TEST_SIZE*100:.0f}/{TEST_SIZE*100:.0f})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Model Initialization
    # scale_pos_weight=10 to handle 1:10 imbalance
    print("Initializing XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=10,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1  # Use all cores
    )

    # 4. Training
    print("Training model (this may take a while)...")
    model.fit(X_train, y_train)

    # 5. Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate specificity and sensitivity manually for clarity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) # Recall for class 1
    specificity = tn / (tn + fp) # Recall for class 0
    precision = tp / (tp + fp)
    
    print(f"\nRecall (Accident Detection): {sensitivity:.4f}")
    print(f"Precision (Accident correct): {precision:.4f}")
    
    # 6. Export
    print(f"Saving model to {MODEL_FILE}...")
    joblib.dump(model, MODEL_FILE)
    
    # Also save the label encoder to handle future inference correctly
    joblib.dump(le_weather, "weather_encoder.pkl")
    print("Training pipeline complete.")

if __name__ == "__main__":
    train_model()
