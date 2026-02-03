import requests
import json

url = "http://localhost:8000/analyze_route"

# Coordinates close to Thondayad Junction
# Thondayad: 11.2721, 75.8052
# Let's route from slightly west to slightly east of it.
source = "11.2720,75.8000"
destination = "11.2720,75.8100"

payload = {
    "source": source,
    "destination": destination,
    "weather": "Clear", # Should trigger Complex Intersection risk if configured
    "time_of_day": "Evening"
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print("Success!")
        print(f"Google Maps URL: {data['google_maps_url']}")
        print(f"Risk Points Found: {len(data['risk_points'])}")
        for rp in data['risk_points']:
            print(f" - {rp['reason']} at ({rp['lat']}, {rp['lon']}) Risk: {rp['risk_score']:.2f}")
    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"Connection failed: {e}")
    print("Ensure the server is running on port 8000.")
