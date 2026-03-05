import pandas as pd
import requests
import time

df = pd.read_csv('kozhikode_accident_history.csv')
accidents = df[df['is_accident'] == 1]
top_spots = accidents.groupby(['latitude', 'longitude']).size().reset_index(name='count').sort_values('count', ascending=False).head(5)

results = []
for idx, row in top_spots.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    try:
        r = requests.get(url, headers={'User-Agent': 'accisence-app'})
        data = r.json()
        name = data.get('display_name', '').split(',')[0] if 'display_name' in data else f"{lat:.4f}, {lon:.4f}"
        results.append(f"{name} ({row['count']} accidents)")
    except Exception as e:
        results.append(f"{lat:.4f}, {lon:.4f} ({row['count']} accidents)")
    time.sleep(1) # Be nice to Nominatim

print("TOP 5 SPOTS:")
for r in results:
    print(r)
