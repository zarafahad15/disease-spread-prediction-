🧬 Disease Spread Prediction (Machine Learning Project)

📖 Overview

This project predicts the next-day spread of a disease (e.g., COVID-19–like outbreaks) using Machine Learning on simulated epidemic data.
It models disease transmission patterns using lag-based time-series features and can run on both:
	•	🔹 Random Forest Regressor (default) — fast & simple
	•	🔹 LSTM Neural Network (optional) — for sequential deep learning

The project is self-contained, easily extendable, and can also train on real-world epidemiological data such as WHO or CDC datasets.

⸻

🚀 Features

✅ Generates synthetic epidemic data using an SIR model (Susceptible–Infected–Recovered)
✅ Builds lag features (1, 2, 3, 7, 14-day history)
✅ Predicts next-day new cases
✅ Automatically chooses model type:
	•	TensorFlow → LSTM
	•	Else → RandomForestRegressor
✅ Produces:
	•	CSV of predictions
	•	Plot comparing actual vs predicted cases
	•	Saved trained model file

⸻

🏗 Tech Stack

Category	Tools
Language	Python 3.9+
Data Science	NumPy, Pandas, Scikit-Learn
Visualization	Matplotlib
Machine Learning	RandomForest, TensorFlow (optional LSTM)
Model Persistence	Joblib / TensorFlow SavedModel


⸻

🧠 How It Works
	1.	Simulate Epidemic Data
	•	Generates S, I, R time series with configurable infection rate (β) and recovery rate (γ).
	2.	Feature Engineering
	•	Creates lag features from new_cases (1, 2, 3, 7, 14 days).
	3.	Train Model
	•	Predicts next-day cases based on historical trends.
	4.	Evaluation
	•	Computes Mean Absolute Error (MAE) and RMSE.
	5.	Visualization & Saving
	•	Plots predicted vs true curves and saves results.

⸻

⚙ Installation & Setup

1. Clone or Download

git clone https://github.com/<your-username>/disease-spread-prediction.git
cd disease-spread-prediction

Or simply download the .zip file and extract it.

2. Install Dependencies

pip install -r requirements.txt

If you don’t have a requirements.txt, install manually:

pip install numpy pandas scikit-learn matplotlib joblib tensorflow

3. Run the Project

python disease_prediction.py


⸻

📁 Project Structure

📂 disease-spread-prediction/
 ├── disease_prediction.py        # Main training script
 ├── disease_predictions.csv      # Output predictions (generated)
 ├── prediction_plot.png          # Visualization of results
 ├── disease_model.joblib         # Trained RandomForest model
 ├── tf_disease_model/ (optional) # Saved LSTM model
 ├── requirements.txt             # Dependencies
 └── README.md                    # Project documentation


⸻

📊 Output Examples

🧾 Console Summary

Summary metrics: {'MAE': 358.3, 'RMSE': 386.7, 'model_used': 'random_forest'}
Saved predictions CSV to: disease_predictions.csv
Saved plot to: prediction_plot.png

📈 Visualization

prediction_plot.png shows the true vs predicted next-day cases trend.

⸻

💾 Outputs Explained

File	Description
disease_predictions.csv	True and predicted next-day cases
prediction_plot.png	Plot comparing actual and predicted spread
disease_model.joblib	Trained model (load via joblib.load())
/tf_disease_model/	TensorFlow SavedModel (if LSTM used)


⸻

🧪 Model Evaluation

Metric	Description
MAE	Mean Absolute Error – average deviation
RMSE	Root Mean Squared Error – penalizes large errors
Model Used	RandomForest or LSTM (auto-selected)


⸻

🧩 How to Load the Model Later

import joblib
model = joblib.load("disease_model.joblib")
pred = model.predict([[200,180,150,90,50]])  # Example input


⸻

🔮 Next Steps / Improvements
	•	Replace synthetic data with real-world datasets
	•	Add R₀ estimation and infection curve forecasting
	•	Deploy with Streamlit or Flask for interactive predictions
	•	Integrate mobility and weather data as features
	•	Add hyperparameter tuning (Optuna / GridSearchCV)
