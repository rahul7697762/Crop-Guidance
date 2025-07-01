from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import os
import pandas as pd
import json
import requests
from datetime import datetime

# Flask app initialization
app = Flask(__name__)

# Model file paths
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/standscaler.pkl"
MINMAX_PATH = "models/minmaxscaler.pkl"
CLUSTERING_PATH = "models/clustering_model.pkl"

# Check if model files exist before loading
required_files = [MODEL_PATH, SCALER_PATH, MINMAX_PATH, CLUSTERING_PATH]
if not all(os.path.exists(f) for f in required_files):
    raise FileNotFoundError("One or more model/scaler files are missing. Please check the 'models' folder.")

# Load the machine learning model and scalers
model = pickle.load(open(MODEL_PATH, 'rb'))
sc = pickle.load(open(SCALER_PATH, 'rb'))
ms = pickle.load(open(MINMAX_PATH, 'rb'))
clustering_model = pickle.load(open(CLUSTERING_PATH, 'rb'))

# Load dataset for finding similar crops
df = pd.read_csv('Crop_recommendation.csv')

# Crop dictionary with detailed information
crop_dict = {
    1: {
        "name": "Rice",
        "growing_season": "Kharif",
        "water_req": "High",
        "growing_time": "120-150 days",
        "tips": "Maintain standing water of 2.5cm during tillering",
        "soil_type": "Clay or clay loam",
        "market_price": "₹1,800-2,200/quintal",
        "diseases": ["Blast", "Brown Spot", "Bacterial Leaf Blight"],
        "nutrients": "High nitrogen requirement during vegetative growth"
    },
    2: {
        "name": "Maize",
        "growing_season": "Kharif/Rabi",
        "water_req": "Medium",
        "growing_time": "95-105 days",
        "tips": "Ensure proper drainage and regular weeding",
        "soil_type": "Well-drained loamy soil",
        "market_price": "₹1,600-1,900/quintal",
        "diseases": ["Leaf Blight", "Stalk Rot", "Ear Rot"],
        "nutrients": "Requires balanced NPK fertilization"
    },
    3: {
        "name": "Jute",
        "growing_season": "Kharif",
        "water_req": "High",
        "growing_time": "120-150 days",
        "tips": "Requires proper water management during growth",
        "soil_type": "Well-drained alluvial soil",
        "market_price": "₹4,500-5,000/quintal",
        "diseases": ["Stem Rot", "Root Rot", "Leaf Mosaic"],
        "nutrients": "Responds well to nitrogen fertilization"
    },
    4: {
        "name": "Cotton",
        "growing_season": "Kharif",
        "water_req": "Medium",
        "growing_time": "150-180 days",
        "tips": "Regular pest monitoring is essential",
        "soil_type": "Deep black soil or alluvial soil",
        "market_price": "₹5,500-6,500/quintal",
        "diseases": ["Bacterial Blight", "Wilt", "Root Rot"],
        "nutrients": "High potassium requirement during boll formation"
    },
    5: {
        "name": "Coconut",
        "growing_season": "Year-round",
        "water_req": "High",
        "growing_time": "5-7 years to first yield",
        "tips": "Requires good drainage and spacing",
        "soil_type": "Sandy loam to clay loam",
        "market_price": "₹2,000-2,500/100 nuts",
        "diseases": ["Bud Rot", "Stem Bleeding", "Root Wilt"],
        "nutrients": "Regular application of organic manure recommended"
    },
    6: {
        "name": "Papaya",
        "growing_season": "Year-round",
        "water_req": "Medium",
        "growing_time": "8-10 months",
        "tips": "Proper spacing and pruning required",
        "soil_type": "Well-drained sandy loam",
        "market_price": "₹20-40/kg",
        "diseases": ["Viral Diseases", "Anthracnose", "Root Rot"],
        "nutrients": "Rich in organic matter and calcium"
    }
}

def get_soil_health_status(n, p, k, ph):
    """Determine soil health based on NPK values and pH"""
    health = {
        'status': 'Good',
        'messages': [],
        'improvements': []
    }
    
    # Check NPK levels
    if n < 50:
        health['messages'].append("Low Nitrogen")
        health['improvements'].append("Add nitrogen-rich fertilizers or organic matter")
    elif n > 140:
        health['messages'].append("High Nitrogen")
        health['improvements'].append("Reduce nitrogen fertilization")

    if p < 30:
        health['messages'].append("Low Phosphorus")
        health['improvements'].append("Add phosphate fertilizers or bone meal")
    elif p > 100:
        health['messages'].append("High Phosphorus")
        health['improvements'].append("Avoid phosphorus fertilization")

    if k < 30:
        health['messages'].append("Low Potassium")
        health['improvements'].append("Add potash fertilizers or wood ash")
    elif k > 100:
        health['messages'].append("High Potassium")
        health['improvements'].append("Reduce potassium fertilization")

    # Check pH levels
    if ph < 5.5:
        health['messages'].append("Acidic soil")
        health['improvements'].append("Add agricultural lime to raise pH")
    elif ph > 7.5:
        health['messages'].append("Alkaline soil")
        health['improvements'].append("Add sulfur or organic matter to lower pH")

    if len(health['messages']) > 2:
        health['status'] = 'Poor'
    elif len(health['messages']) > 0:
        health['status'] = 'Fair'

    return health

def get_similar_crops(features):
    """Find similar crops based on input features using clustering"""
    # Scale the features
    scaled_features = sc.transform(ms.transform(features))
    
    # Get the cluster
    cluster = clustering_model.predict(scaled_features)[0]
    
    # Scale the dataset features
    df_features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    df_scaled = sc.transform(ms.transform(df_features))
    
    # Get all crops in the same cluster
    cluster_mask = clustering_model.predict(df_scaled) == cluster
    similar_crops = df[cluster_mask]['label'].unique().tolist()
    
    return similar_crops[:5]

def get_market_trends():
    """Simulate market trend data"""
    return {
        "Rice": {"trend": "Stable", "forecast": "Slight increase expected", "demand": "High"},
        "Maize": {"trend": "Rising", "forecast": "Strong demand in coming months", "demand": "Medium"},
        "Jute": {"trend": "Fluctuating", "forecast": "Price stability expected", "demand": "Medium"},
        "Cotton": {"trend": "Rising", "forecast": "High demand in textile sector", "demand": "High"},
        "Coconut": {"trend": "Stable", "forecast": "Consistent demand", "demand": "Medium"},
        "Papaya": {"trend": "Rising", "forecast": "Growing export demand", "demand": "High"}
    }

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/form", methods=['GET', 'POST'])
def form():
    return render_template('form.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve input values
        N = request.form.get('Nitrogen', '').strip()
        P = request.form.get('Phosporus', '').strip()
        K = request.form.get('Potassium', '').strip()
        temp = request.form.get('Temperature', '').strip()
        humidity = request.form.get('Humidity', '').strip()
        ph = request.form.get('Ph', '').strip()
        rainfall = request.form.get('Rainfall', '').strip()

        # Check for empty fields
        if not all([N, P, K, temp, humidity, ph, rainfall]):
            raise ValueError("All fields are required!")

        # Convert input values to float
        feature_list = list(map(float, [N, P, K, temp, humidity, ph, rainfall]))

        # Reshape for model prediction
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply scaling transformations
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Get model prediction
        prediction = model.predict(final_features)[0]

        # Get crop details
        crop_info = crop_dict.get(prediction, {"name": "Unknown Crop"})
        
        # Get soil health status
        soil_health = get_soil_health_status(float(N), float(P), float(K), float(ph))

        # Get similar crops
        similar_crops = get_similar_crops(single_pred)
        
        # Get market trends
        market_trends = get_market_trends()
        crop_market_trend = market_trends.get(crop_info["name"], {})
        
        # Prepare response data
        response_data = {
            "status": "success",
            "crop": crop_info,
            "similar_crops": similar_crops,
            "soil_health": soil_health,
            "market_trend": crop_market_trend,
            "conditions": {
                "temperature": float(temp),
                "humidity": float(humidity),
                "rainfall": float(rainfall),
                "soil_ph": float(ph)
            }
        }

        return render_template('form.html', 
                             result=response_data,
                             current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    except ValueError as ve:
        return render_template('form.html', error=f"Invalid input: {ve}")
    except Exception as e:
        return render_template('form.html', error=f"An error occurred: {str(e)}")

@app.route("/get_weather", methods=['POST'])
def get_weather():
    try:
        lat = request.json.get('lat')
        lon = request.json.get('lon')
        api_key = "YOUR_OPENWEATHERMAP_API_KEY"  # In production, use environment variables
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        
        response = requests.get(url)
        weather_data = response.json()
        
        return jsonify(weather_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
