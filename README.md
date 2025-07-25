# Medicine-Tracker

⚠️ Important Notes
MongoDB Connection: The application connects to mongodb://localhost:27017/. Ensure your MongoDB instance is running on this default port.

Speech Recognition: The speech recognition feature relies on the Web Speech API, which is supported by most modern browsers (e.g., Chrome, Edge). It requires an internet connection for optimal performance as it uses Google's speech recognition service.

Prophet Model: The Prophet model needs sufficient historical data to make accurate predictions. The medicine_usage_sample.csv provides some sample data. For real-world usage, you would accumulate data over time.

Production Deployment: This setup is for development purposes. For production, you would need a more robust WSGI server (like Gunicorn or uWSGI) and a proper web server (like Nginx or Apache).
