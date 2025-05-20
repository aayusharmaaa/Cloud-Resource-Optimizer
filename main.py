from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from utils.simulate_data import get_mock_cpu_data
from model.lstm_model import LSTMModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lstm_model = LSTMModel()

@app.get("/")
def root():
    return {"message": "Cloud Optimizer API"}

@app.get("/predict")
def predict():
    cpu_data = get_mock_cpu_data()
    prediction = lstm_model.predict(cpu_data)
    action = get_action(prediction)
    return {
        "cpu_history": cpu_data,
        "predicted_utilization": prediction,
        "recommended_action": action
    }

def get_action(predicted_value):
    if predicted_value > 80:
        return "Scale Up"
    elif predicted_value < 50:
        return "Scale Down"
    return "Maintain"