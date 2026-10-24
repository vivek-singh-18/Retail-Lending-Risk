"""
Feature Engineering and Data Leakage Prevention Module.

Constructs financially relevant risk metrics strictly using information
available at application time. Prevents lookahead bias and repayment data leakage.
"""

import numpy as np
import pandas as pd

def compute_application_features(df):
    """
    Computes application-time financial features for risk scoring.
    
    Features engineered:
    1. dti (Debt-to-Income): Monthly debt obligations / Monthly income
    2. lti (Loan-to-Income): Requested loan amount / Annual income
    3. emi_to_income: Estimated monthly EMI / Monthly income
    4. credit_utilization: Existing credit card / line utilization ratio
    5. active_loans: Number of currently active credit obligations
    6. delinquency_frequency: Historical delinquency count
    7. recent_delinquency: Binary flag (1 if delinquent in last 6 months)
    8. income_stability_num: Numeric encoding of income stability (Low=0, Medium=1, High=2)
    9. loan_burden_index: Composite ratio of (DTI * LTI)
    10. credit_score_norm: Standardized credit score scaled [0, 1]
    """
    df = df.copy()
    
    # 1. Income & EMI Ratios
    df["annual_income"] = df["monthly_income"] * 12
    df["lti"] = (df["requested_amount"] / df["annual_income"]).round(4)
    
    # Calculate estimated monthly EMI (assumed 12% avg interest rate for standardization)
    r = 0.12 / 12
    tenure_months = np.where(df["loan_type"] == "Housing_Loan", 180, 
                             np.where(df["loan_type"] == "Used_Car_Loan", 48, 36))
    
    emi_est = df["requested_amount"] * r * ((1 + r)**tenure_months) / (((1 + r)**tenure_months) - 1)
    df["emi_to_income"] = (emi_est / df["monthly_income"]).round(4)
    
    # DTI = (Existing Debt Payments + Estimated EMI) / Monthly Income
    existing_debt_payments = df["monthly_income"] * df["credit_utilization"] * 0.15
    df["dti"] = ((existing_debt_payments + emi_est) / df["monthly_income"]).round(4)
    
    # 2. Income Stability Numerical Encoding
    stability_map = {"Low": 0, "Medium": 1, "High": 2}
    df["income_stability_num"] = df["income_stability"].map(stability_map).fillna(1)
    
    # 3. Composite Risk Indices
    df["loan_burden_index"] = (df["dti"] * df["lti"]).round(4)
    df["credit_score_norm"] = ((df["credit_score"] - 300) / 550.0).round(4)
    
    # 4. Employment to Age Ratio
    emp_len_col = "employment_length" if "employment_length" in df.columns else ("employment_length_x" if "employment_length_x" in df.columns else "employment_length_y")
    df["emp_age_ratio"] = (df[emp_len_col] / df["age"]).round(4)
    if emp_len_col != "employment_length":
        df["employment_length"] = df[emp_len_col]
        
    return df

def apply_temporal_split(df, date_col="application_date", train_end="2025-06-30", val_end="2025-12-31"):
    """
    Splits dataset chronologically into Train, Validation, and Test sets
    to emulate realistic out-of-time production validation.
    
    Train:   start <= date <= 2025-06-30
    Val:     2025-07-01 <= date <= 2025-12-31
    Test:    date > 2025-12-31
    """
    df[date_col] = pd.to_datetime(df[date_col])
    
    train_mask = df[date_col] <= pd.to_datetime(train_end)
    val_mask = (df[date_col] > pd.to_datetime(train_end)) & (df[date_col] <= pd.to_datetime(val_end))
    test_mask = df[date_col] > pd.to_datetime(val_end)
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Temporal Split Summary:")
    print(f" - Train Set: {len(train_df):,} samples ({train_df[date_col].min().strftime('%Y-%m-%d')} to {train_df[date_col].max().strftime('%Y-%m-%d')})")
    print(f" - Val Set:   {len(val_df):,} samples ({val_df[date_col].min().strftime('%Y-%m-%d')} to {val_df[date_col].max().strftime('%Y-%m-%d')})")
    print(f" - Test Set:  {len(test_df):,} samples ({test_df[date_col].min().strftime('%Y-%m-%d')} to {test_df[date_col].max().strftime('%Y-%m-%d')})")
    
    return train_df, val_df, test_df
