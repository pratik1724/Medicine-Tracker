from flask import Blueprint, request, jsonify
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict_stock', methods=['GET'])
def predict_stock():
    medicine = request.args.get('medicine')
    
    # Simulated historical data - replace with actual DB or CSV source
    df = pd.read_csv('medicine_usage_sample.csv')  # ensure this file exists
    df = df[df['medicine_name'] == medicine]
    
    df = df.rename(columns={"date": "ds", "quantity_used": "y"})
    df['ds'] = pd.to_datetime(df['ds'])
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    
    # Save plot
    plot_path = f"static/predictions/{medicine}_forecast.png"
    fig = model.plot(forecast)
    plt.title(f"7-Day Forecast for {medicine}")
    plt.savefig(plot_path)

    return jsonify({
        "message": "Forecast generated successfully.",
        "plot_url": f"/{plot_path}"
    })
