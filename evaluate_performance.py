import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# --- Configuration ---
INPUT_FILE = "kozhikode_accident_history.csv"
MODEL_FILE = "accident_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def evaluate():
    print(f"Loading data from {INPUT_FILE}...")
    # Loading full dataset might be slow. 
    # If it's too large, we can sample, but for accurate "test set" reproduction:
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Loaded {len(df)} rows.")

    # 1. Preprocessing (Must match training logic)
    print("Preprocessing...")
    
    le_weather = LabelEncoder()
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    
    # Interaction: Night * Weather (Must match training logic)
    df['night_weather'] = df['is_night'] * df['weather_encoded']
    
    features = [
        'curvature_score', 
        'maxspeed', 
        'is_junction', 
        'is_night', 
        'weather_encoded', 
        'hour_of_day', 
        'is_holiday',
        'night_weather' # Added interaction
    ]
    
    X = df[features]
    y = df['is_accident']

    # 2. Split (Recreate the Test Set)
    print("Splitting data to isolate Test Set...")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 3. Load Model
    print(f"Loading {MODEL_FILE}...")
    if not joblib.os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found.")
        return
        
    model = joblib.load(MODEL_FILE)
    
    # 4. Predict
    print("Running predictions on Test Set...")
    y_pred = model.predict(X_test)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 5. Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall_accidents = tp / (tp + fn)
    precision_accidents = tp / (tp + fp)
    
    # Top-K Precision
    results_df = pd.DataFrame({'actual': y_test, 'prob': y_prob})
    top_5_df = results_df.sort_values(by='prob', ascending=False).head(5)
    precision_at_5 = top_5_df['actual'].mean()

    with open("final_scores.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.2%}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Recall (Accident Detection): {recall_accidents:.2%}\n")
        f.write(f"Precision (Accident Alert): {precision_accidents:.2%}\n")
        f.write(f"Top-5 Precision: {precision_at_5:.2%}\n")

    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    print("-" * 30)
    print(f"Safety Critical Metrics:")
    print(f"Recall (Accident Detection): {recall_accidents:.2%} (Percentage of actual accidents correctly predicted)")
    print(f"Precision (Accident Alert):  {precision_accidents:.2%} (Percentage of predicted accidents that were real)")
    print("-" * 30)

if __name__ == "__main__":
    evaluate()
