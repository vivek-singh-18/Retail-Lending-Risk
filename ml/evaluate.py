"""
Model Evaluation and Calibration Verification Module.

Runs complete out-of-time evaluation on the test dataset and exports metrics
to `evaluation_results/ml_evaluation.json`.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, precision_recall_curve
)
from sklearn.calibration import calibration_curve

from ml.features import compute_application_features, apply_temporal_split
from ml.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def evaluate_pipeline():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data", "raw")
    models_dir = os.path.join(base_dir, "ml", "models")
    eval_dir = os.path.join(base_dir, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("Executing comprehensive ML evaluation on out-of-time test set...")
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
    model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    with open(os.path.join(models_dir, "model_metadata.json"), "r") as f:
        metadata = json.load(f)
        
    threshold = metadata.get("optimal_threshold", 0.40)
    
    customers_df = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    credit_df = pd.read_csv(os.path.join(data_dir, "credit_history.csv"))
    emp_df = pd.read_csv(os.path.join(data_dir, "employment_history.csv"))
    apps_df = pd.read_csv(os.path.join(data_dir, "applications.csv"))
    
    df = apps_df.merge(customers_df, on="customer_id") \
                .merge(credit_df, on="customer_id") \
                .merge(emp_df.drop(columns=["employment_length"], errors="ignore"), on="customer_id")
                
    df_feat = compute_application_features(df)
    _, _, test_df = apply_temporal_split(df_feat)
    
    X_test = test_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["default_target"]
    
    X_test_proc = preprocessor.transform(X_test)
    y_prob = model.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calibration Curve (binned default rate vs predicted prob)
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
    calibration_data = [
        {"predicted_prob": round(float(p_pred), 4), "actual_default_rate": round(float(p_true), 4)}
        for p_pred, p_true in zip(prob_pred, prob_true)
    ]
    
    eval_report = {
        "model_name": metadata.get("best_model_name"),
        "optimal_threshold": threshold,
        "test_sample_count": len(test_df),
        "test_default_count": int(y_test.sum()),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "precision": round(float(tp / max(tp + fp, 1)), 4),
        "recall": round(float(tp / max(tp + fn, 1)), 4),
        "f1_score": round(float(2*tp / max(2*tp + fp + fn, 1)), 4),
        "calibration_bins": calibration_data
    }
    
    output_path = os.path.join(eval_dir, "ml_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(eval_report, f, indent=2)
        
    print("\n" + "="*60)
    print("ML EVALUATION SUMMARY")
    print("="*60)
    print(f" - Model:              {metadata.get('best_model_name')}")
    print(f" - ROC-AUC:            {roc_auc:.4f}")
    print(f" - PR-AUC:             {pr_auc:.4f}")
    print(f" - Brier Score:        {brier:.4f}")
    print(f" - Optimal Threshold:  {threshold}")
    print(f" - Recall @ Threshold: {tp / max(tp + fn, 1):.4f}")
    print(f" - Precision @ Thresh: {tp / max(tp + fp, 1):.4f}")
    print(f"[SUCCESS] Exported ML metrics to {output_path}")
    return eval_report

if __name__ == "__main__":
    evaluate_pipeline()
