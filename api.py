from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ml_controller import controller, DataModel
import os
import numpy as np

app = FastAPI()

# Add CORS middleware to allow requests from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the controller at module level
controler = None

@app.on_event("startup")
def startup_event():
    global controler
    try:
        controler = controller()
        print("Controller initialized successfully")
    except Exception as e:
        print(f"Error initializing controller: {str(e)}")



@app.post("/predict/{model}")
def predict(model: str, data: DataModel):
    if model not in ["rf", "dt", "knn", "xgb", "lr"]:
        return {"error": "Invalid model name"}
    
    if controler is None:
        return {"error": "Models not properly loaded. Check server logs."}
    
    try:
        if controler.models.get(model) is None:
            return {"error": f"Model {model} could not be loaded"}
            
        prediction = controler.predict(model, data)
        # Convert numpy types to standard Python types for JSON serialization
        if isinstance(prediction, np.integer):
            prediction = int(prediction)
        elif isinstance(prediction, np.floating):
            prediction = float(prediction)
        elif isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
            
        if(prediction == 1):
            pred = "Potable"
        else:
            pred = "Not Potable"

        return {"prediction": prediction, "label": pred}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Read the HTML file and return it
    with open('index.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# uvicorn api:app --host 0.0.0.0 --port 5000 --reload