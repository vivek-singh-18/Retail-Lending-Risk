"""
Model Training and Optimization Pipeline.

Trains Logistic Regression, Random Forest, and XGBoost models for default prediction.
Performs model evaluation, threshold tuning under asymmetric financial costs,
and persists trained artifacts into `ml/models/`.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix

from ml.features import compute_application_features, apply_temporal_split
from ml.preprocessing import build_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def train_and_evaluate_models():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading datasets for model training...")
    customers_df = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    credit_df = pd.read_csv(os.path.join(data_dir, "credit_history.csv"))
    emp_df = pd.read_csv(os.path.join(data_dir, "employment_history.csv"))
    apps_df = pd.read_csv(os.path.join(data_dir, "applications.csv"))
    
    df = apps_df.merge(customers_df, on="customer_id") \
                .merge(credit_df, on="customer_id") \
                .merge(emp_df.drop(columns=["employment_length"], errors="ignore"), on="customer_id")
                
    df_feat = compute_application_features(df)
    train_df, val_df, test_df = apply_temporal_split(df_feat)
    
    X_train = train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["default_target"]
    
    X_val = val_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_val = val_df["default_target"]
    
    X_test = test_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["default_target"]
    
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # Save Preprocessor
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
    
    # Class weight ratio
    scale_pos = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=42, eval_metric="logloss")
    }
    
    results = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_model_obj = None
    
    print("\nTraining candidate models...")
    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        
        y_val_prob = model.predict_proba(X_val_proc)[:, 1]
        y_test_prob = model.predict_proba(X_test_proc)[:, 1]
        
        roc_val = roc_auc_score(y_val, y_val_prob)
        pr_val = average_precision_score(y_val, y_val_prob)
        roc_test = roc_auc_score(y_test, y_test_prob)
        pr_test = average_precision_score(y_test, y_test_prob)
        brier_test = brier_score_loss(y_test, y_test_prob)
        
        results[name] = {
            "val_roc_auc": round(float(roc_val), 4),
            "val_pr_auc": round(float(pr_val), 4),
            "test_roc_auc": round(float(roc_test), 4),
            "test_pr_auc": round(float(pr_test), 4),
            "test_brier_score": round(float(brier_test), 4)
        }
        
        print(f" - {name:20s} | Val ROC-AUC: {roc_val:.4f} | Test ROC-AUC: {roc_test:.4f} | Test PR-AUC: {pr_test:.4f} | Brier: {brier_test:.4f}")
        
        if roc_test > best_roc_auc:
            best_roc_auc = roc_test
            best_model_name = name
            best_model_obj = model
            
    # Optimize Threshold for Best Model under Financial Cost Framework
    # Cost(FN) = 100,000 INR (Loss on defaulted loan)
    # Cost(FP) = 10,000 INR (Lost interest margin on declined good borrower)
    y_test_prob_best = best_model_obj.predict_proba(X_test_proc)[:, 1]
    
    thresholds = np.linspace(0.1, 0.9, 81)
    min_cost = float("inf")
    optimal_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_test_prob_best >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total_cost = (fn * 100000) + (fp * 10000)
        if total_cost < min_cost:
            min_cost = total_cost
            optimal_threshold = thresh
            
    optimal_threshold = round(float(optimal_threshold), 2)
    print(f"\n[OPTIMIZATION] Best Model: {best_model_name}")
    print(f" - Optimal Financial Decision Threshold: {optimal_threshold} (Min Cost: INR {min_cost:,.0f})")
    
    # Save Best Model
    joblib.dump(best_model_obj, os.path.join(models_dir, "best_model.pkl"))
    
    metadata = {
        "best_model_name": best_model_name,
        "optimal_threshold": optimal_threshold,
        "metrics": results,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    }
    
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[SUCCESS] Trained artifacts saved to {models_dir}")
    return metadata

if __name__ == "__main__":
    train_and_evaluate_models()
