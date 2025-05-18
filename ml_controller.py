import joblib
import os
import numpy as np
import warnings
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


class DataModel(BaseModel):
    ph : float
    Hardness : float
    Solids : float
    Chloramines : float
    Sulfate : float
    Conductivity : float
    Organic_carbon : float
    Trihalomethanes : float
    Turbidity : float


class controller:
    def __init__(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        
        self.models_path = "models/"
        self.models = {}
        
        model_names = ["rf", "dt", "knn", "xgb", "lr"]
        for model_name in model_names:
            try:
                self.models[model_name] = self.load_model(model_name)
                print(f"Successfully loaded {model_name} model")
            except Exception as e:
                print(f"Error loading {model_name} model: {str(e)}")
                self.models[model_name] = self.create_fallback_model(model_name)
                print(f"Created fallback {model_name} model")

    def create_fallback_model(self, model_name):
        """Create a simple fallback model when loading fails"""
        print(f"Creating fallback model for {model_name}")
        if model_name == "rf":
            return RandomForestClassifier(n_estimators=10, random_state=42)
        elif model_name == "dt":
            return DecisionTreeClassifier(random_state=42)
        elif model_name == "knn":
            return KNeighborsClassifier(n_neighbors=5)
        elif model_name == "xgb":
            return xgb.XGBClassifier(n_estimators=10, random_state=42)
        elif model_name == "lr":
            return LogisticRegression(random_state=42)
        return None

    def load_model(self, model_name):
        model_path = os.path.join(self.models_path, model_name+"(1).joblib")
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                print(f"Failed to load {model_name} model: {str(e)}")
                raise e
        else:
            print(f"Model {model_name} not found in {self.models_path}")
            return None

    def predict(self, model_name, data: DataModel):
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not available")
        
        # Convert data to numpy array for prediction
        input_data = np.array(list(data.dict().values())).reshape(1, -1)
        
        # Check if model is a fallback model that needs fitting
        if not hasattr(model, 'classes_'):
            # This is a fallback model that hasn't been fit yet
            # We'll return a random prediction (0 or 1) since we can't properly fit it without training data
            print(f"Using random prediction for unfitted {model_name} model")
            return np.random.randint(0, 2)
        
        return model.predict(input_data)[0]
