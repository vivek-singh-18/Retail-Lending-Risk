"""
Synthetic Data Generation Pipeline for Retail Lending Risk Intelligence.

Generates realistic relational data for 10,000 customers and 15,000+ loan applications,
loans, repayments, credit history, employment history, and delinquency records.
Enforces realistic statistical dependencies between risk factors and default outcomes.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def generate_customers(num_customers=10000):
    print(f"Generating {num_customers} customer profiles...")
    customer_ids = [f"C{i+1:05d}" for i in range(num_customers)]
    
    ages = np.random.randint(21, 65, size=num_customers)
    city_tiers = np.random.choice(["Tier_1", "Tier_2", "Tier_3"], size=num_customers, p=[0.4, 0.35, 0.25])
    
    emp_types = ["Salaried_MNC", "Salaried_Govt", "Salaried_Private", "Self_Employed", "Business"]
    emp_probs = [0.25, 0.20, 0.35, 0.12, 0.08]
    employment_types = np.random.choice(emp_types, size=num_customers, p=emp_probs)
    
    # Monthly income follows log-normal distribution
    income_base = np.random.lognormal(mean=10.8, sigma=0.6, size=num_customers)
    monthly_incomes = np.clip(income_base, 18000, 450000).round(-2)
    
    employment_lengths = np.round(np.random.uniform(0.5, 30.0, size=num_customers), 1)
    
    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "city_tier": city_tiers,
        "employment_type": employment_types,
        "monthly_income": monthly_incomes,
        "employment_length": employment_lengths
    })
    return customers_df

def generate_credit_and_employment(customers_df):
    print("Generating credit history and employment records...")
    num_customers = len(customers_df)
    
    credit_scores = []
    active_loans_list = []
    closed_loans_list = []
    credit_utils = []
    delinq_counts = []
    recent_delinqs = []
    
    employer_types = []
    income_stabilities = []
    
    for idx, row in customers_df.iterrows():
        emp_type = row["employment_type"]
        income = row["monthly_income"]
        emp_len = row["employment_length"]
        
        # Credit Score base influenced by employment type and income
        base_score = 680 + (10 if emp_type in ["Salaried_MNC", "Salaried_Govt"] else -15)
        base_score += min(60, int((income - 30000) / 3000))
        score_noise = np.random.normal(0, 50)
        c_score = int(np.clip(base_score + score_noise, 320, 850))
        credit_scores.append(c_score)
        
        # Delinquency history correlated inversely with credit score
        delinq_prob = max(0.02, (750 - c_score) / 600)
        d_count = np.random.poisson(delinq_prob * 3)
        recent_d = 1 if (d_count > 0 and np.random.rand() < 0.65) else 0
        
        delinq_counts.append(d_count)
        recent_delinqs.append(recent_d)
        
        # Credit utilization
        util_base = 0.35 + (0.25 if c_score < 620 else -0.1) + (0.15 if recent_d else 0)
        util = round(float(np.clip(np.random.normal(util_base, 0.15), 0.05, 0.98)), 2)
        credit_utils.append(util)
        
        # Active and Closed loans
        act = int(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.2, 0.35, 0.25, 0.12, 0.05, 0.03]))
        cls = int(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], p=[0.15, 0.25, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02]))
        active_loans_list.append(act)
        closed_loans_list.append(cls)
        
        # Employer Details
        if emp_type == "Salaried_MNC":
            emp_t = "MNC"
            stability = "High"
        elif emp_type == "Salaried_Govt":
            emp_t = "Govt"
            stability = "High"
        elif emp_type == "Salaried_Private":
            emp_t = "Private_Ltd"
            stability = "Medium" if emp_len > 2 else "Low"
        elif emp_type == "Self_Employed":
            emp_t = "Proprietary"
            stability = "Medium" if income > 50000 else "Low"
        else:
            emp_t = "Unregistered"
            stability = "Low"
            
        employer_types.append(emp_t)
        income_stabilities.append(stability)
        
    credit_df = pd.DataFrame({
        "customer_id": customers_df["customer_id"],
        "credit_score": credit_scores,
        "active_loans": active_loans_list,
        "closed_loans": closed_loans_list,
        "credit_utilization": credit_utils,
        "delinquency_count": delinq_counts,
        "recent_delinquency": recent_delinqs
    })
    
    emp_df = pd.DataFrame({
        "customer_id": customers_df["customer_id"],
        "employer_type": employer_types,
        "employment_length": customers_df["employment_length"],
        "income_stability": income_stabilities
    })
    
    return credit_df, emp_df

def generate_applications_and_loans(customers_df, credit_df, emp_df, num_applications=16500):
    print(f"Generating {num_applications} loan applications and loan records...")
    
    app_ids = [f"APP{i+1:06d}" for i in range(num_applications)]
    customer_sample_ids = np.random.choice(customers_df["customer_id"], size=num_applications, replace=True)
    
    cust_map = customers_df.set_index("customer_id").to_dict(orient="index")
    cred_map = credit_df.set_index("customer_id").to_dict(orient="index")
    emp_map = emp_df.set_index("customer_id").to_dict(orient="index")
    
    loan_types = ["Personal_Loan", "Housing_Loan", "Used_Car_Loan"]
    type_probs = [0.55, 0.25, 0.20]
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 6, 30)
    total_days = (end_date - start_date).days
    
    app_rows = []
    loan_rows = []
    delinq_rows = []
    repayment_rows = []
    
    loan_counter = 1
    delinq_counter = 1
    payment_counter = 1
    
    for i in range(num_applications):
        app_id = app_ids[i]
        c_id = customer_sample_ids[i]
        
        c_info = cust_map[c_id]
        cred_info = cred_map[c_id]
        emp_info = emp_map[c_id]
        
        l_type = np.random.choice(loan_types, p=type_probs)
        app_days = np.random.randint(0, total_days)
        app_date = start_date + timedelta(days=int(app_days))
        app_date_str = app_date.strftime("%Y-%m-%d")
        
        income = c_info["monthly_income"]
        c_score = cred_info["credit_score"]
        c_util = cred_info["credit_utilization"]
        recent_d = cred_info["recent_delinquency"]
        d_count = cred_info["delinquency_count"]
        stabl = emp_info["income_stability"]
        
        # Requested Amount based on loan type and income
        if l_type == "Housing_Loan":
            req_amount = round(float(income * np.random.uniform(15, 45)), -4)
            tenure = int(np.random.choice([120, 180, 240]))
            base_ir = 8.5
        elif l_type == "Used_Car_Loan":
            req_amount = round(float(income * np.random.uniform(3, 10)), -3)
            tenure = int(np.random.choice([36, 48, 60]))
            base_ir = 11.5
        else: # Personal Loan
            req_amount = round(float(income * np.random.uniform(2, 12)), -3)
            tenure = int(np.random.choice([12, 24, 36, 48, 60]))
            base_ir = 14.0
            
        req_amount = max(50000, min(req_amount, 7500000))
        
        # Calculate DTI & EMI
        annual_rate = base_ir + (200 / max(c_score, 400))
        r = annual_rate / (12 * 100)
        emi = req_amount * r * ((1 + r)**tenure) / (((1 + r)**tenure) - 1)
        dti = round((emi + (income * c_util * 0.2)) / income, 3)
        
        # Logistic formula for default target computation
        logit = (
            -2.8
            - 0.007 * (c_score - 650)
            + 2.8 * (dti - 0.4)
            + 2.2 * (c_util - 0.3)
            + 1.6 * recent_d
            + 0.3 * d_count
            - 0.03 * c_info["employment_length"]
            + (0.6 if stabl == "Low" else -0.3)
            + np.random.normal(0, 0.7)
        )
        prob_default = 1.0 / (1.0 + np.exp(-logit))
        default_target = 1 if prob_default > 0.45 else 0
        
        # Application Status
        if c_score < 550 or dti > 0.65 or recent_d == 1 and prob_default > 0.7:
            app_status = "Declined"
        elif prob_default > 0.55:
            app_status = np.random.choice(["Approved", "Declined"], p=[0.35, 0.65])
        else:
            app_status = "Approved"
            
        app_rows.append({
            "application_id": app_id,
            "customer_id": c_id,
            "application_date": app_date_str,
            "requested_amount": req_amount,
            "loan_type": l_type,
            "application_status": app_status,
            "default_target": default_target
        })
        
        # If application was approved, create active/closed/defaulted loan record
        if app_status == "Approved":
            loan_id = f"L{loan_counter:06d}"
            loan_counter += 1
            
            interest_rate = round(annual_rate, 2)
            
            if default_target == 1:
                loan_status = "Defaulted"
            else:
                loan_status = np.random.choice(["Active", "Closed"], p=[0.6, 0.4])
                
            loan_rows.append({
                "loan_id": loan_id,
                "customer_id": c_id,
                "loan_type": l_type,
                "loan_amount": req_amount,
                "tenure": tenure,
                "interest_rate": interest_rate,
                "application_date": app_date_str,
                "loan_status": loan_status
            })
            
            # Generate repayments and delinquency records
            num_payments = min(12, tenure)
            for m in range(1, num_payments + 1):
                p_id = f"P{payment_counter:07d}"
                payment_counter += 1
                
                p_date = app_date + timedelta(days=m*30)
                due_amt = round(emi, 2)
                
                if loan_status == "Defaulted" and m >= num_payments - 3:
                    dpd = int(np.random.choice([30, 60, 90, 120]))
                    paid_amt = 0.0 if dpd >= 60 else round(due_amt * 0.4, 2)
                    
                    # Record delinquency
                    delinq_rows.append({
                        "delinquency_id": f"D{delinq_counter:06d}",
                        "customer_id": c_id,
                        "loan_id": loan_id,
                        "days_past_due": dpd,
                        "delinquency_date": p_date.strftime("%Y-%m-%d"),
                        "severity": "Critical" if dpd >= 90 else ("Severe" if dpd >= 60 else "Moderate")
                    })
                    delinq_counter += 1
                else:
                    dpd = 0 if np.random.rand() > 0.08 else int(np.random.choice([5, 10, 15]))
                    paid_amt = due_amt
                    
                repayment_rows.append({
                    "payment_id": p_id,
                    "loan_id": loan_id,
                    "payment_date": p_date.strftime("%Y-%m-%d"),
                    "due_amount": due_amt,
                    "paid_amount": paid_amt,
                    "days_past_due": dpd
                })

    apps_df = pd.DataFrame(app_rows)
    loans_df = pd.DataFrame(loan_rows)
    delinq_df = pd.DataFrame(delinq_rows)
    repay_df = pd.DataFrame(repayment_rows)
    
    return apps_df, loans_df, delinq_df, repay_df

def main():
    set_seed(42)
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    customers_df = generate_customers(num_customers=10000)
    credit_df, emp_df = generate_credit_and_employment(customers_df)
    apps_df, loans_df, delinq_df, repay_df = generate_applications_and_loans(customers_df, credit_df, emp_df, num_applications=16000)
    
    customers_df.to_csv(os.path.join(raw_dir, "customers.csv"), index=False)
    credit_df.to_csv(os.path.join(raw_dir, "credit_history.csv"), index=False)
    emp_df.to_csv(os.path.join(raw_dir, "employment_history.csv"), index=False)
    apps_df.to_csv(os.path.join(raw_dir, "applications.csv"), index=False)
    loans_df.to_csv(os.path.join(raw_dir, "loans.csv"), index=False)
    delinq_df.to_csv(os.path.join(raw_dir, "delinquencies.csv"), index=False)
    repay_df.to_csv(os.path.join(raw_dir, "repayments.csv"), index=False)
    
    print("\n[SUCCESS] Synthetic data generation complete!")
    print(f"- Customers: {len(customers_df)}")
    print(f"- Credit History Records: {len(credit_df)}")
    print(f"- Employment History Records: {len(emp_df)}")
    print(f"- Applications: {len(apps_df)} (Defaults: {apps_df['default_target'].sum()}, Rate: {apps_df['default_target'].mean():.2%})")
    print(f"- Approved Loans: {len(loans_df)}")
    print(f"- Delinquencies Recorded: {len(delinq_df)}")
    print(f"- Repayment Schedules: {len(repay_df)}")

if __name__ == "__main__":
    main()
