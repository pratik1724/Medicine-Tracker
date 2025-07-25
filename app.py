from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from datetime import datetime, timedelta
from pymongo import MongoClient
import re

from prophet import Prophet
import pandas as pd

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb://localhost:27017/")
db = client["surgical_medicine"]
log_collection = db["medicine_usage"]        
thresholds_collection = db["medicine_thresholds"] 

INITIAL_CAPACITY_ML = 100

DEFAULT_MEDICINE_THRESHOLDS = {
    # "betadine": 20,
    # "lidocaine": 15,
    # "hydrogen peroxide": 30
}

YELLOW_THRESHOLD = 40
ORANGE_THRESHOLD = 30
RED_THRESHOLD = 29

def initialize_medicine_thresholds():
    for med, threshold in DEFAULT_MEDICINE_THRESHOLDS.items():
        if not thresholds_collection.find_one({"medicine": med}):
            thresholds_collection.insert_one({"medicine": med, "threshold": threshold})
            print(f"Added default medicine threshold for: {med}")

with app.app_context():
    initialize_medicine_thresholds()


def calculate_current_remaining(medicine_name):
    """
    Assumes:
    - Positive 'quantity' in log_collection means usage (stock decreases).
    - Negative 'quantity' in log_collection means restock (stock increases).
    - An initial conceptual capacity of INITIAL_CAPACITY_ML (e.g., 100ml).
    """
    total_net_change_from_logs = sum([entry['quantity'] for entry in log_collection.find({"medicine": medicine_name})])
    
    remaining = INITIAL_CAPACITY_ML - total_net_change_from_logs
    return remaining

def predict_depletion(medicine, current_remaining):

    usage_entries = list(log_collection.find({"medicine": medicine, "quantity": {"$gt": 0}}))
    
    if not usage_entries:
        return "N/A - No usage data"

    usage_entries.sort(key=lambda x: x['timestamp'])

    first_usage_date = usage_entries[0]['timestamp']
    total_usage_since_first = sum([entry['quantity'] for entry in usage_entries])

    time_span_days = (datetime.now() - first_usage_date).days
    if time_span_days == 0: 
        time_span_days = 1 

    daily_avg_usage = total_usage_since_first / time_span_days

    if daily_avg_usage > 0:
        days_to_depletion = current_remaining / daily_avg_usage
        
        if days_to_depletion < 0:
            return "Already Depleted"

        depletion_date = datetime.now() + timedelta(days=days_to_depletion)
        return depletion_date.strftime("%Y-%m-%d")
    else:
        return "N/A - No average usage or sufficient stock"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/usage_log")
def usage_log():
    logs = list(log_collection.find().sort("timestamp", -1))
    return render_template("usage_log.html", logs=logs)

@app.route("/restock")
def restock():
    summary = {}
    medicines_with_thresholds = list(thresholds_collection.find({}))
    
    for med_data in medicines_with_thresholds:
        med = med_data["medicine"]
        
        current_remaining = calculate_current_remaining(med)

        depletion = predict_depletion(med, current_remaining) 
        
        summary[med] = {
            "remaining": current_remaining, 
            "depletion_date": depletion
        }
    return render_template("restock.html", summary=summary)

@app.route("/buy_list")
def buy_list():
    buy_list = []
    medicines_with_thresholds = list(thresholds_collection.find({}))

    for med_data in medicines_with_thresholds:
        med = med_data["medicine"]
        
        current_remaining = calculate_current_remaining(med)

        item_color = ""
        if current_remaining <= RED_THRESHOLD: 
            item_color = "red"
        elif current_remaining <= ORANGE_THRESHOLD: 
            item_color = "orange"
        elif current_remaining <= YELLOW_THRESHOLD: 
            item_color = "yellow"
        else:
            continue 

        if item_color: 
            buy_list.append({
                "medicine": med,
                "remaining": current_remaining,
                "color": item_color 
            })
    
    color_priority = {"red": 1, "orange": 2, "yellow": 3}
    buy_list.sort(key=lambda x: (color_priority.get(x['color'], 99), x['remaining']))

    return render_template("buy_list.html", buy_list=buy_list)

@app.route("/log_usage", methods=["POST"])
def log_usage():
    data = request.get_json()
    text = data["text"].lower()

    match = re.search(r"([a-zA-Z ]+?)\s(?:used\s)?(\d+)", text)
    if match:
        medicine = match.group(1).strip()
        quantity = int(match.group(2))

        if not thresholds_collection.find_one({"medicine": medicine}):
            thresholds_collection.insert_one({"medicine": medicine, "threshold": 20}) 

        log_collection.insert_one({
            "medicine": medicine,
            "quantity": quantity, 
            "timestamp": datetime.now()
        })

        return jsonify({"success": True})

    return jsonify({"success": False})

@app.route("/restock_medicine", methods=["GET", "POST"])
def restock_medicine():
    message = ""
    if request.method == "POST":
        med = request.form["medicine"]
        quantity = int(request.form["quantity"])

        log_collection.insert_one({
            "medicine": med,
            "quantity": -quantity,  # negative quantity means restocked
            "timestamp": datetime.now()
        })

        message = f"{quantity}ml of {med} restocked successfully."

    medicines = [doc["medicine"] for doc in thresholds_collection.find({}, {"_id": 0, "medicine": 1})]

    return render_template("restock_medicine.html", message=message, medicines=medicines)

@app.route("/forecast")
def forecast():
    all_logs = list(log_collection.find())
    
    usage_data = []
    for log in all_logs:
        if log['quantity'] > 0: 
            usage_data.append({
                'ds': log['timestamp'],
                'medicine': log['medicine'],
                'y': log['quantity']
            })

    if not usage_data:
        return render_template("forecast.html", stock_alerts={"info": "No sufficient usage data to generate forecasts."})

    df_all_usage = pd.DataFrame(usage_data)
    df_all_usage['ds'] = pd.to_datetime(df_all_usage['ds'])

    stock_alerts = {}
    
    for med in df_all_usage['medicine'].unique():
        med_df = df_all_usage[df_all_usage['medicine'] == med][['ds', 'y']]
        
        if med_df.empty or len(med_df) < 2: 
            stock_alerts[med] = "Insufficient usage data for forecasting"
            continue

        try:
            model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05) 
            model.fit(med_df)
            
            future = model.make_future_dataframe(periods=14)
            forecast_result = model.predict(future)

            current_remaining_stock = calculate_current_remaining(med)
            
            depletion_day = None
            simulated_remaining = current_remaining_stock

            for _, row in forecast_result[-14:].iterrows(): 
                predicted_usage = max(row['yhat'], 0) # Ensure predicted usage is not negative
                simulated_remaining -= predicted_usage
                
                if simulated_remaining <= 0:
                    depletion_date = row['ds']
                    days_until_depletion = (depletion_date - datetime.today()).days
                    depletion_day = days_until_depletion
                    break

            if depletion_day is not None and depletion_day >= 0:
                stock_alerts[med] = f"needed in {depletion_day} days"
            elif depletion_day is not None: 
                stock_alerts[med] = "already depleted"
            else: 
                stock_alerts[med] = "sufficient for now"
        
        except Exception as e:
            stock_alerts[med] = f"Forecast error: {str(e)}"

    return render_template("forecast.html", stock_alerts=stock_alerts)

if __name__ == "__main__":
    app.run(debug=True)