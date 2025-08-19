📈 Stock Market Prediction Dashboard
An end-to-end AI-powered stock prediction system built with Flask (Python backend) and a modern Tailwind/Chart.js frontend.
The system uses machine learning models (XGBoost) trained on historical NSE data to forecast stock prices, visualize actual vs predicted trends, and provide model performance metrics.


🚀 Features
🌐 Flask API Backend with endpoints for training, predicting, and fetching available stock symbols
📊 Interactive Dashboard with real-time charts (Chart.js)
⚡ Automatic Model Training if no model exists for a symbol
🎯 Model Metrics: R², RMSE, MAE displayed in the dashboard
🔮 Next-Day Prediction using historical features
💾 Export Options: Download forecasts as CSV or chart snapshots
🎨 Modern UI built with TailwindCSS and glassmorphism design

🏗️ Tech Stack
Backend: Flask, Flask-CORS, scikit-learn, XGBoost, Pandas, NumPy, Joblib
Frontend: HTML5, TailwindCSS, Chart.js, Vanilla JavaScript
Data: Historical NSE (Indian stock market) dataset

📂 Project Structure
├── src                        # source codes like web app and html
    ├── app.py                 # Flask backend (API + model training/prediction logic)
    ├── html                   # html folder
      ├── welcome.html         # Landing page (intro + navigation)
      ├── prediction.html      # Dashboard with charts & metrics
├── notebooks                  # contains all the notebooks used for dataset creation preprocessing etc
├── data                       # data sets (gitignore)
├── models/                    # Trained models & scalers (ignored in .gitignore)
└── requirements.txt           # Python dependencies

📊 DATA
The data set is downloaded from kaggle
We can use the dataset_creation notebook from notebooks folder to make the raw dataset
Using the preprocessing note book we can make our final data set which is the nse_cleaned.csv
https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020

🌐 Pythonanywhere
https://panchamiraveendran.pythonanywhere.com/
