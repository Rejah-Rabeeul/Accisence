import pandas as pd
import joblib
import os
from realtime_inference_utils import prepare_live_features

def test_inference():
    print("Loading Model...")
    if not os.path.exists("accident_model.pkl"):
        print("Error: accident_model.pkl not found.")
        return
    
    model = joblib.load("accident_model.pkl")
    
    # Define Scenarios
    scenarios = [
        {
            "name": "High Risk (Night, Rain, Curve, Fast)",
            "road": {'curvature_score': 1.5, 'maxspeed': 80.0, 'is_junction': 1},
            "weather": "Rain",
            "time": {'hour_of_day': 23, 'is_night': 1, 'is_holiday': 0}
        },
        {
            "name": "Low Risk (Day, Clear, Straight, Slow)",
            "road": {'curvature_score': 1.0, 'maxspeed': 30.0, 'is_junction': 0},
            "weather": "Clear",
            "time": {'hour_of_day': 10, 'is_night': 0, 'is_holiday': 0}
        },
        {
            "name": "Moderate Risk (Day, Fog, Curve, Normal Speed)",
            "road": {'curvature_score': 1.2, 'maxspeed': 50.0, 'is_junction': 0},
            "weather": "Fog",
            "time": {'hour_of_day': 9, 'is_night': 0, 'is_holiday': 1}
        }
    ]
    
    print("\n--- Running Inference Tests ---")
    
    for sc in scenarios:
        print(f"\nTesting Scenario: {sc['name']}")
        
        # Prepare features using the utility function to ensure exact matching
        # Note: prepare_live_features handles encoding
        features_df = prepare_live_features(sc['road'], sc['weather'], sc['time'])
        
        # Predict
        # We use predict_proba to see the confidence
        prob = model.predict_proba(features_df)[0][1]
        prediction = model.predict(features_df)[0]
        
        print(f"Features: {features_df.to_dict(orient='records')[0]}")
        print(f"Accident Probability: {prob:.4f}")
        print(f"Prediction: {'ACCIDENT' if prediction==1 else 'Safe'}")
        
        # Basic assertion logic for demonstration
        if "High Risk" in sc['name'] and prob < 0.5:
            print("WARNING: Model predicted Low Risk for a High Risk scenario.")
        elif "Low Risk" in sc['name'] and prob > 0.5:
             print("WARNING: Model predicted High Risk for a Low Risk scenario.")

if __name__ == "__main__":
    test_inference()
