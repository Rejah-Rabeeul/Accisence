import requests
import datetime
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
# Replace with your actual OpenWeatherMap API Key
OWM_API_KEY = "04dcf262093cb04cb8db9f2d259e24a5"
KOZHIKODE_LAT = 11.2588
KOZHIKODE_LON = 75.7804
WEATHER_ENCODER_PATH = "weather_encoder.pkl"

# Feature order must match the training script EXACTLY
MODEL_FEATURES = [
    'curvature_score', 
    'maxspeed', 
    'is_junction', 
    'is_night', 
    'weather_encoded', 
    'hour_of_day', 
    'is_holiday',
    'night_weather'  # New Interaction Feature
]

def load_weather_encoder():
    """Loads the LabelEncoder used during training."""
    if os.path.exists(WEATHER_ENCODER_PATH):
        return joblib.load(WEATHER_ENCODER_PATH)
    else:
        # If missing, creating a dummy one for demonstration/fallback
        # In production this should raise an error
        print(f"Warning: {WEATHER_ENCODER_PATH} not found. Creating temporary encoder.")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(['Clear', 'Rain', 'Fog']) 
        return le

def get_live_weather():
    """
    Fetches current weather for Kozhikode from OpenWeatherMap.
    Returns: Weather string ('Clear', 'Rain', 'Fog', etc.)
    Defaults to 'Clear' on error.
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={KOZHIKODE_LAT}&lon={KOZHIKODE_LON}&appid={OWM_API_KEY}"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # OWM 'main' weather conditions: Thunderstorm, Drizzle, Rain, Snow, Mist, Smoke, Haze, Dust, Fog, Sand, Ash, Squall, Tornado, Clear, Clouds
        weather_main = data['weather'][0]['main']
        
        # Map OWM conditions to our training categories: Clear, Rain, Fog
        # Simple mapping logic
        if weather_main in ['Rain', 'Thunderstorm', 'Drizzle']:
            return 'Rain'
        elif weather_main in ['Fog', 'Mist', 'Haze', 'Smoke']:
            return 'Fog'
        else:
            return 'Clear'  # Clouds/Clear/Others -> Clear (Baseline)
            
    except requests.exceptions.RequestException as e:
        print(f"Weather API Error: {e}. Defaulting to 'Clear'.")
        return 'Clear'
    except Exception as e:
        print(f"Unexpected error parsing weather: {e}. Defaulting to 'Clear'.")
        return 'Clear'

def get_temporal_features():
    """
    Captures system time and derives temporal features.
    Returns: dict with 'hour_of_day', 'is_night', 'is_holiday'
    """
    now = datetime.datetime.now()
    hour = now.hour
    
    # is_night: Between 7 PM (19) and 6 AM (6)
    # Training script used: ((hour >= 18) | (hour < 6)) -> 6PM to 6AM
    # Requirement here says: "7 PM and 6 AM". I will align with Requirement for this function, 
    # BUT consistency with training is crucial. The training used 6PM (18). 
    # I will stick to the Training Logic (18) to match the model's learned patterns 
    # unless the user strictly insists on 7PM. Training consistency usually wins.
    # However, user explicitly asked for "7 PM". I will follow user Prompt: 19.
    is_night = 1 if (hour >= 19 or hour < 6) else 0
    
    # is_weekened/holiday
    # Weekday: Mon=0 ... Sun=6
    is_weekend = 1 if now.weekday() >= 5 else 0
    
    return {
        'hour_of_day': hour,
        'is_night': is_night,
        'is_holiday': is_weekend # Mapping weekend to holiday for inference
    }

def prepare_live_features(road_feature_dict, weather_str, time_data):
    """
    Assembles static and dynamic features into a DataFrame for inference.
    
    Args:
        road_feature_dict (dict): {'curvature_score': float, 'maxspeed': float, 'is_junction': int}
        weather_str (str): 'Clear', 'Rain', or 'Fog'
        time_data (dict): Output from get_temporal_features()
        
    Returns:
        pd.DataFrame: A single-row dataframe ready for model.predict()
    """
    le = load_weather_encoder()
    
    # Encode weather
    # Handle unseen labels safely
    try:
        weather_encoded = le.transform([weather_str])[0]
    except ValueError:
        # Fallback for unknown labels (should correspond to mapping in get_live_weather)
        print(f"Warning: Unknown weather '{weather_str}'. Encoding as 'Clear'.")
        weather_encoded = le.transform(['Clear'])[0]
        
    # Assemble feature dictionary
    input_data = {
        'curvature_score': road_feature_dict.get('curvature_score', 1.0),
        'maxspeed': road_feature_dict.get('maxspeed', 40.0),
        'is_junction': road_feature_dict.get('is_junction', 0),
        'is_night': time_data['is_night'],
        'weather_encoded': weather_encoded,
        'hour_of_day': time_data['hour_of_day'],
        'is_holiday': time_data['is_holiday'],
        'night_weather': time_data['is_night'] * weather_encoded # New Interaction
    }
    
    # Create DataFrame with exact column ordering
    df_features = pd.DataFrame([input_data])
    df_features = df_features[MODEL_FEATURES]
    
    return df_features

# --- Usage Example ---
if __name__ == "__main__":
    print("Initializing Nervous System...")
    
    # 1. Get Environment State
    current_weather = get_live_weather()
    current_time = get_temporal_features()
    
    print(f"Live Environment: Weather={current_weather}, Time={current_time}")
    
    # 2. Mock Road Data (This would come from your Graph Database in prod)
    mock_road_segment = {
        'curvature_score': 1.05,
        'maxspeed': 60.0,
        'is_junction': 0
    }
    
    # 3. Prepare Features
    features_df = prepare_live_features(mock_road_segment, current_weather, current_time)
    
    print("\nCorrectly assembled Feature Vector for Inference:")
    print(features_df)
    
    # 4. Load Model and Predict (Optional check)
    if os.path.exists("accident_model.pkl"):
        print("\nTesting Prediction...")
        model = joblib.load("accident_model.pkl")
        prob = model.predict_proba(features_df)[0][1]
        print(f"Accident Probability: {prob:.4f}")
