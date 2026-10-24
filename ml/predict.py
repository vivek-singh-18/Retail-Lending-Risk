"""
Inference & Prediction Pipeline.

Loads preprocessor and trained ML model to generate default probabilities
and risk categorization for arbitrary loan applications.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from ml.features import compute_application_features
from ml.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

class RiskPredictor:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            
        self.models_dir = models_dir
        self.preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        self.model_path = os.path.join(models_dir, "best_model.pkl")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Trained model artifact not found. Please execute `python -m ml.train` first.")
            
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        self.threshold = self.metadata.get("optimal_threshold", 0.40)
        
    def predict_single(self, borrower_dict):
        """
        Predicts default probability and risk level for a single application dictionary.
        """
        df_single = pd.DataFrame([borrower_dict])
        
        # Ensure default required columns exist
        defaults = {
            "age": 35,
            "city_tier": "Tier_1",
            "employment_type": "Salaried_Private",
            "monthly_income": 60000,
            "employment_length": 4.0,
            "credit_score": 720,
            "active_loans": 1,
            "closed_loans": 2,
            "credit_utilization": 0.30,
            "delinquency_count": 0,
            "recent_delinquency": 0,
            "employer_type": "Private_Ltd",
            "income_stability": "Medium",
            "requested_amount": 500000,
            "loan_type": "Personal_Loan"
        }
        for k, v in defaults.items():
            if k not in df_single.columns:
                df_single[k] = v
                
        df_feat = compute_application_features(df_single)
        X_proc = self.preprocessor.transform(df_feat[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
        
        prob_default = float(self.model.predict_proba(X_proc)[0, 1])
        
        if prob_default < 0.15:
            risk_category = "LOW"
        elif prob_default < 0.35:
            risk_category = "MODERATE"
        elif prob_default < 0.60:
            risk_category = "HIGH"
        else:
            risk_category = "VERY_HIGH"
            
        return {
            "default_probability": round(prob_default, 4),
            "risk_category": risk_category,
            "threshold": self.threshold,
            "exceeds_threshold": prob_default >= self.threshold,
            "features": df_feat.iloc[0].to_dict()
        }

def get_default_predictor():
    return RiskPredictor()
