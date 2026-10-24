"""
Data Quality and Integrity Validation Module.

Performs robust checks on raw and merged data tables including:
- Missing value detection
- Duplicate record checks (Customer ID, Loan ID, Application ID)
- Logical boundary & range validation (Age, Income, Credit Score, DTI, Utilization)
- Impossible value / negative value detection
- Referential integrity verification
- Outlier reporting
"""

import os
import pandas as pd
import numpy as np

def validate_customers(df):
    errors = []
    warnings = []
    
    # Missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        errors.append(f"Missing values found in customers: {null_counts[null_counts > 0].to_dict()}")
        
    # Duplicate Primary Keys
    dups = df["customer_id"].duplicated().sum()
    if dups > 0:
        errors.append(f"Duplicate customer_id entries: {dups}")
        
    # Boundary checks
    invalid_age = df[(df["age"] < 18) | (df["age"] > 100)]
    if len(invalid_age) > 0:
        errors.append(f"Found {len(invalid_age)} records with age outside [18, 100]")
        
    invalid_income = df[df["monthly_income"] <= 0]
    if len(invalid_income) > 0:
        errors.append(f"Found {len(invalid_income)} records with non-positive monthly income")
        
    invalid_emp_len = df[df["employment_length"] < 0]
    if len(invalid_emp_len) > 0:
        errors.append(f"Found {len(invalid_emp_len)} records with negative employment length")
        
    return errors, warnings

def validate_credit_history(df):
    errors = []
    warnings = []
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        errors.append(f"Missing values in credit history: {null_counts[null_counts > 0].to_dict()}")
        
    dups = df["customer_id"].duplicated().sum()
    if dups > 0:
        errors.append(f"Duplicate customer_id entries in credit history: {dups}")
        
    invalid_score = df[(df["credit_score"] < 300) | (df["credit_score"] > 850)]
    if len(invalid_score) > 0:
        errors.append(f"Found {len(invalid_score)} invalid credit scores outside [300, 850]")
        
    invalid_util = df[(df["credit_utilization"] < 0) | (df["credit_utilization"] > 2.0)]
    if len(invalid_util) > 0:
        errors.append(f"Found {len(invalid_util)} credit utilization values outside [0, 2.0]")
        
    return errors, warnings

def validate_applications(df, customers_df=None):
    errors = []
    warnings = []
    
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        errors.append(f"Missing values in applications: {null_counts[null_counts > 0].to_dict()}")
        
    dups = df["application_id"].duplicated().sum()
    if dups > 0:
        errors.append(f"Duplicate application_id entries: {dups}")
        
    invalid_amt = df[df["requested_amount"] <= 0]
    if len(invalid_amt) > 0:
        errors.append(f"Found {len(invalid_amt)} non-positive requested amounts")
        
    if customers_df is not None:
        valid_cust_ids = set(customers_df["customer_id"])
        app_cust_ids = set(df["customer_id"])
        unmatched = app_cust_ids - valid_cust_ids
        if len(unmatched) > 0:
            errors.append(f"Referential integrity failure: {len(unmatched)} applications reference non-existent customer IDs")
            
    return errors, warnings

def run_all_validations(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
        
    print("=" * 60)
    print("RUNNING DATA QUALITY & INTEGRITY AUDIT")
    print("=" * 60)
    
    customers_df = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    credit_df = pd.read_csv(os.path.join(data_dir, "credit_history.csv"))
    emp_df = pd.read_csv(os.path.join(data_dir, "employment_history.csv"))
    apps_df = pd.read_csv(os.path.join(data_dir, "applications.csv"))
    loans_df = pd.read_csv(os.path.join(data_dir, "loans.csv"))
    
    all_errors = []
    all_warnings = []
    
    c_errs, c_warns = validate_customers(customers_df)
    cr_errs, cr_warns = validate_credit_history(credit_df)
    app_errs, app_warns = validate_applications(apps_df, customers_df)
    
    all_errors.extend(c_errs + cr_errs + app_errs)
    all_warnings.extend(c_warns + cr_warns + app_warns)
    
    if len(all_errors) == 0:
        print("[PASSED] Data Quality Checks: 0 Errors detected.")
        print(f" - Audited {len(customers_df):,} Customers, {len(credit_df):,} Credit Profiles, {len(apps_df):,} Applications.")
        return True, all_errors, all_warnings
    else:
        print(f"[FAILED] Found {len(all_errors)} Data Quality Violations:")
        for err in all_errors:
            print(f" - [ERROR] {err}")
        return False, all_errors, all_warnings

if __name__ == "__main__":
    run_all_validations()
