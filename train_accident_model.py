import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
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
    
    # --- Feature Engineering ---
    # Interaction: Night * Weather (Weather is 0,1,2 so it works as an interaction ordinal)
    df['night_weather'] = df['is_night'] * df['weather_encoded']
    
    # Feature Selection
    features = [
        'curvature_score', 
        'maxspeed', 
        'is_junction', 
        'is_night', 
        'weather_encoded', 
        'hour_of_day', 
        'is_holiday',
        'night_weather' # New Interaction
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

    # 3. Model Initialization & Tuning
    print("Initializing Tuning (RandomizedSearchCV)...")
    
    xgb_pipeline = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=4 # Parallel tree building
    )
    
    # Param Grid
    # Focused on Accuracy AND Recall (Balanced Dataset)
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.05, 0.1, 0.2],
        'scale_pos_weight': [1], # Standard balanced training
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_dist,
        n_iter=15, # Increased for better tuning
        scoring='accuracy', # Optimize for Accuracy directly
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=1, # Sequential loop over params
        error_score='raise'
    )

    # 4. Training
    print("Tuning model (this may take a while)...")
    search.fit(X_train, y_train)
    
    print(f"Best Params: {search.best_params_}")
    print(f"Best F1 Score: {search.best_score_:.4f}")
    
    model = search.best_estimator_

    # 5. Evaluation
    print("Evaluating best model...")
    y_pred = model.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) # Recall
    precision = tp / (tp + fp)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall (Accident Detection): {sensitivity:.4f}")
    print(f"Precision (Accident correct): {precision:.4f}")
    
    # Save Metrics
    with open("final_scores.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.2%}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Recall (Accident Detection): {sensitivity:.2%}\n")
        f.write(f"Precision (Accident Alert): {precision:.2%}\n")
        f.write(f"Best Params: {search.best_params_}\n")
    
    # 6. Export
    print(f"Saving model to {MODEL_FILE}...")
    joblib.dump(model, MODEL_FILE)
    
    # Also save the label encoder
    joblib.dump(le_weather, "weather_encoder.pkl")
    print("Training pipeline complete.")

if __name__ == "__main__":
    train_model()
