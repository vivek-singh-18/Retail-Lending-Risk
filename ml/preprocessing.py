"""
Preprocessing Pipeline Module.

Constructs ColumnTransformer pipelines for scaling numerical risk metrics
and one-hot encoding categorical attributes.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERICAL_FEATURES = [
    "age", "monthly_income", "employment_length", "credit_score",
    "active_loans", "closed_loans", "credit_utilization", "delinquency_count",
    "recent_delinquency", "requested_amount", "annual_income", "lti",
    "emi_to_income", "dti", "income_stability_num", "loan_burden_index",
    "credit_score_norm", "emp_age_ratio"
]

CATEGORICAL_FEATURES = [
    "city_tier", "employment_type", "loan_type", "employer_type"
]

def build_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor
