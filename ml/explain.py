"""
SHAP Explainability Layer.

Provides global model feature importance and local instance-level SHAP explanations
for retail lending credit risk model predictions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import shap

from ml.features import compute_application_features
from ml.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

class ModelExplainer:
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            
        self.models_dir = models_dir
        self.preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
        self.model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
        
        # Get feature names after column transformation
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_cols_encoded = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        self.feature_names = NUMERICAL_FEATURES + cat_cols_encoded
        
        # Initialize SHAP explainer
        if hasattr(self.model, "coef_"):
            self.explainer = shap.LinearExplainer(self.model, masker=shap.maskers.Independent(data=np.zeros((1, len(self.feature_names)))))
        else:
            self.explainer = shap.TreeExplainer(self.model)
            
    def explain_local(self, borrower_dict, top_k=5):
        """
        Generates local SHAP explanation for a single borrower.
        Returns top risk-increasing and risk-decreasing features with exact SHAP contributions.
        """
        df_single = pd.DataFrame([borrower_dict])
        
        defaults = {
            "age": 35, "city_tier": "Tier_1", "employment_type": "Salaried_Private",
            "monthly_income": 60000, "employment_length": 4.0, "credit_score": 720,
            "active_loans": 1, "closed_loans": 2, "credit_utilization": 0.30,
            "delinquency_count": 0, "recent_delinquency": 0, "employer_type": "Private_Ltd",
            "income_stability": "Medium", "requested_amount": 500000, "loan_type": "Personal_Loan"
        }
        for k, v in defaults.items():
            if k not in df_single.columns:
                df_single[k] = v
                
        df_feat = compute_application_features(df_single)
        X_proc = self.preprocessor.transform(df_feat[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
        
        shap_values = self.explainer.shap_values(X_proc)
        if isinstance(shap_values, list):
            shap_vec = shap_values[1][0] # class 1 (default)
        elif len(shap_values.shape) == 3:
            shap_vec = shap_values[0, :, 1]
        else:
            shap_vec = shap_values[0]
            
        feature_contributions = []
        for name, val in zip(self.feature_names, shap_vec):
            feature_contributions.append({
                "feature": name,
                "shap_value": round(float(val), 4)
            })
            
        # Sort by SHAP value
        sorted_contribs = sorted(feature_contributions, key=lambda x: x["shap_value"], reverse=True)
        
        risk_increasing = [x for x in sorted_contribs if x["shap_value"] > 0][:top_k]
        risk_decreasing = [x for x in sorted_contribs if x["shap_value"] < 0][-top_k:]
        
        return {
            "risk_increasing_factors": risk_increasing,
            "risk_decreasing_factors": risk_decreasing,
            "disclaimer": "SHAP quantifies model feature contribution to the predicted probability; it does not prove causality."
        }

def get_model_explainer():
    return ModelExplainer()
