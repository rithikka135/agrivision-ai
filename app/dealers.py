import requests

def fetch_nearby_dealers(lat: float, lon: float, radius_km: int = 15) -> list:
    overpass_url = "https://overpass-api.de/api/interpreter"
    radius_meters = radius_km * 1000
    
    query = f"""
    [out:json][timeout:10];
    (
      node["shop"="agrarian"](around:{radius_meters},{lat},{lon});
      node["shop"="trade"](around:{radius_meters},{lat},{lon});
      node["shop"="chemist"](around:{radius_meters},{lat},{lon});
    );
    out body 5;
    """
    try:
        response = requests.post(overpass_url, data={"data": query}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            dealers = []
            for el in elements:
                tags = el.get("tags", {})
                name = tags.get("name", "Local Agrochemical / Supply Store")
                dealers.append({
                    "name": name,
                    "lat": el.get("lat"),
                    "lon": el.get("lon"),
                    "maps_url": f"https://www.google.com/maps/search/?api=1&query={el.get('lat')},{el.get('lon')}"
                })
            if dealers:
                return dealers
    except Exception as e:
        print(f"[DEALER API ERROR] {e}")
        
    return [{
        "name": "Search Nearby Registered Agrochemical Outlets",
        "lat": lat,
        "lon": lon,
        "maps_url": f"https://www.google.com/maps/search/pesticide+fertilizer+store/@{lat},{lon},12z"
    }]