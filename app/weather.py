import requests

def get_weather_disease_risk(lat: float = 11.0168, lon: float = 76.9558) -> dict:
    """
    Fetches real-time weather from Open-Meteo API and calculates spore outbreak risk.
    Default coordinates: Coimbatore, TN (11.0168, 76.9558)
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,rain"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        curr = data.get("current", {})
        temp = curr.get("temperature_2m", 25.0)
        humidity = curr.get("relative_humidity_2m", 60.0)
        rain = curr.get("rain", 0.0)

        # Environmental Disease Risk Heuristic
        if humidity >= 75 and 18 <= temp <= 30:
            risk_level = "CRITICAL (High Fungal Spore Risk)"
            badge_color = "red"
        elif humidity >= 60:
            risk_level = "MODERATE (Watch Spot Infections)"
            badge_color = "orange"
        else:
            risk_level = "LOW (Favorable Growth Conditions)"
            badge_color = "green"

        return {
            "temperature": f"{temp}°C",
            "humidity": f"{humidity}%",
            "rain": f"{rain} mm",
            "risk_level": risk_level,
            "badge_color": badge_color
        }
    except Exception as e:
        print(f"[WEATHER API ERROR]: {e}")
        return {
            "temperature": "25.0°C",
            "humidity": "65.0%",
            "rain": "0.0 mm",
            "risk_level": "MODERATE (Offline Mode)",
            "badge_color": "orange"
        }