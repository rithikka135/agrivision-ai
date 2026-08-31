CROP_MARKET_DATA = {
    "Potato": {"avg_yield_kg_per_acre": 8000, "price_per_kg": 20},
    "Tomato": {"avg_yield_kg_per_acre": 10000, "price_per_kg": 25},
    "Pepper Bell": {"avg_yield_kg_per_acre": 6000, "price_per_kg": 40}
}

DISEASE_SEVERITY_INDEX = {
    "Late Blight": 0.85,
    "Early Blight": 0.40,
    "Bacterial Spot": 0.50,
    "Leaf Mold": 0.30,
    "Healthy": 0.0
}

def calculate_economic_risk(disease_name: str, field_acres: float) -> dict:
    is_healthy = "healthy" in disease_name.lower()
    
    crop_type = "Tomato"
    if "potato" in disease_name.lower():
        crop_type = "Potato"
    elif "pepper" in disease_name.lower():
        crop_type = "Pepper Bell"

    market = CROP_MARKET_DATA.get(crop_type, CROP_MARKET_DATA["Tomato"])
    
    if is_healthy:
        total_yield = market["avg_yield_kg_per_acre"] * field_acres
        estimated_val = total_yield * market["price_per_kg"]
        return {
            "status": "Optimal Yield Projected",
            "crop_type": crop_type,
            "projected_yield_kg": f"{total_yield:,.0f} Kg",
            "estimated_market_value": f"₹{estimated_val:,.0f}",
            "estimated_financial_loss": "₹0 (Healthy Field)",
            "action_note": "Field conditions are optimal. Continue standard crop care to maintain maximum market yield."
        }

    severity = 0.45
    for key, val in DISEASE_SEVERITY_INDEX.items():
        if key.lower() in disease_name.lower():
            severity = val
            break

    total_potential_yield = market["avg_yield_kg_per_acre"] * field_acres
    total_potential_val = total_potential_yield * market["price_per_kg"]
    
    estimated_loss_kg = total_potential_yield * severity
    estimated_loss_val = estimated_loss_kg * market["price_per_kg"]

    return {
        "status": "Financial Loss Risk Warning",
        "crop_type": crop_type,
        "projected_yield_kg": f"{total_potential_yield - estimated_loss_kg:,.0f} Kg (Loss of {estimated_loss_kg:,.0f} Kg)",
        "yield_loss_percentage": f"{int(severity * 100)}%",
        "estimated_financial_loss": f"₹{estimated_loss_val:,.0f}",
        "action_note": f"Applying recommended treatments within 48 hours can prevent up to {int(severity * 100)}% of crop destruction."
    }