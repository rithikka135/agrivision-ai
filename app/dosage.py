import math

# Agronomic Chemical Application Database (Rates per Acre)
# Standard spray volume: 150 to 200 Liters of water per acre
CHEMICAL_DOSAGE_DATABASE = {
    "Tomato Leaf Mold": {
        "chemical_name": "Difenoconazole 25% EC",
        "rate_per_acre_ml": 120,          # 120 ml per acre
        "water_volume_per_acre_l": 150,   # 150 Liters water per acre
        "knapsack_pump_capacity_l": 15,   # Standard 15L back-pack sprayer
        "application_method": "Foliar Spray",
        "pre_harvest_interval_days": 7
    },
    "Pepper Bell Bacterial Spot": {
        "chemical_name": "Copper Oxychloride 50% WP + Streptocycline",
        "rate_per_acre_ml": 500,          # 500 grams per acre
        "water_volume_per_acre_l": 200,
        "knapsack_pump_capacity_l": 15,
        "application_method": "High-Volume Foliar Spray",
        "pre_harvest_interval_days": 14
    },
    "Tomato Early Blight": {
        "chemical_name": "Chlorothalonil 75% WP",
        "rate_per_acre_ml": 400,          # 400 grams per acre
        "water_volume_per_acre_l": 150,
        "knapsack_pump_capacity_l": 15,
        "application_method": "Foliar Spray",
        "pre_harvest_interval_days": 7
    },
    "Tomato Late Blight": {
        "chemical_name": "Metalaxyl 8% + Mancozeb 64% WP",
        "rate_per_acre_ml": 600,          # 600 grams per acre
        "water_volume_per_acre_l": 200,
        "knapsack_pump_capacity_l": 15,
        "application_method": "Systemic Foliar Drench",
        "pre_harvest_interval_days": 10
    }
}

DEFAULT_DOSAGE = {
    "chemical_name": "Broad-Spectrum Bio-Fungicide (Neem-Based 10,000 PPM)",
    "rate_per_acre_ml": 500,
    "water_volume_per_acre_l": 150,
    "knapsack_pump_capacity_l": 15,
    "application_method": "Foliar Spray",
    "pre_harvest_interval_days": 3
}

def calculate_field_dosage(disease_name: str, field_acres: float) -> dict:
    """Calculates total chemical quantity, water required, and pump filling ratios."""
    is_healthy = "healthy" in disease_name.lower()
    
    if is_healthy:
        return {
            "status": "No Chemical Required",
            "message": "Crop is healthy. Zero synthetic chemicals or bio-pesticides required.",
            "field_acres": field_acres
        }

    # Fetch chemical parameters
    info = CHEMICAL_DOSAGE_DATABASE.get(disease_name, DEFAULT_DOSAGE)
    
    total_chemical_qty = round(info["rate_per_acre_ml"] * field_acres, 2)
    total_water_qty = round(info["water_volume_per_acre_l"] * field_acres, 2)
    
    # Calculate number of 15L Knapsack Sprayer pump fills
    pump_capacity = info["knapsack_pump_capacity_l"]
    total_pumps = math.ceil(total_water_qty / pump_capacity)
    chemical_per_pump = round(total_chemical_qty / total_pumps, 2)

    return {
        "status": "Action Required",
        "field_acres": field_acres,
        "chemical_name": info["chemical_name"],
        "total_chemical_qty": f"{total_chemical_qty} ml/g",
        "total_water_required": f"{total_water_qty} Liters",
        "application_method": info["application_method"],
        "pre_harvest_interval": f"{info['pre_harvest_interval_days']} Days",
        "mixing_instructions": {
            "total_15L_pumps": total_pumps,
            "chemical_per_pump": f"{chemical_per_pump} ml/g per 15L pump fill",
            "guideline": f"Fill each 15L sprayer tank with 15L water and add exactly {chemical_per_pump} ml/g of {info['chemical_name']}."
        }
    }