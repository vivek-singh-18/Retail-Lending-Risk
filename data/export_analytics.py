"""
Portfolio Analytics CSV Export Pipeline for Power BI.

Queries the SQL database (`database/retail_lending.db`) and exports 8 structured
analytic datasets into `data/processed/` ready for Power BI dashboard consumption.
"""

import os
import sqlite3
import pandas as pd

def export_portfolio_analytics():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(base_dir, "database", "retail_lending.db")
    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    print(f"Exporting Power BI analytical CSV datasets to {proc_dir}...")
    
    # 1. Portfolio Summary CSV
    q1 = """
    SELECT 
        COUNT(DISTINCT l.loan_id) AS total_loans_disbursed,
        SUM(l.loan_amount) AS total_disbursed_amount,
        AVG(l.loan_amount) AS avg_loan_amount,
        SUM(CASE WHEN l.loan_status = 'Active' THEN l.loan_amount ELSE 0 END) AS active_exposure,
        SUM(CASE WHEN l.loan_status = 'Defaulted' THEN l.loan_amount ELSE 0 END) AS defaulted_exposure,
        ROUND(CAST(SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(l.loan_id) * 100, 2) AS default_rate_pct
    FROM loans l;
    """
    pd.read_sql_query(q1, conn).to_csv(os.path.join(proc_dir, "portfolio_summary.csv"), index=False)
    
    # 2. Risk by Product CSV
    q2 = """
    SELECT 
        l.loan_type,
        COUNT(l.loan_id) AS total_loans,
        SUM(l.loan_amount) AS total_disbursed_volume,
        ROUND(AVG(l.interest_rate), 2) AS avg_interest_rate,
        SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS default_count,
        ROUND(CAST(SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(l.loan_id) * 100, 2) AS default_rate_pct
    FROM loans l
    GROUP BY l.loan_type;
    """
    pd.read_sql_query(q2, conn).to_csv(os.path.join(proc_dir, "risk_by_product.csv"), index=False)
    
    # 3. Risk by City Tier CSV
    q3 = """
    SELECT 
        c.city_tier,
        COUNT(DISTINCT c.customer_id) AS customer_count,
        COUNT(l.loan_id) AS total_loans,
        SUM(l.loan_amount) AS total_exposure,
        ROUND(AVG(c.monthly_income), 2) AS avg_income,
        ROUND(CAST(SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(l.loan_id) * 100, 2) AS default_rate_pct
    FROM customers c
    JOIN loans l ON c.customer_id = l.customer_id
    GROUP BY c.city_tier;
    """
    pd.read_sql_query(q3, conn).to_csv(os.path.join(proc_dir, "risk_by_city.csv"), index=False)
    
    # 4. Risk by Income Segment CSV
    q4 = """
    WITH IncomeSeg AS (
        SELECT 
            customer_id,
            CASE 
                WHEN monthly_income < 30000 THEN '1. Low Income (<30k)'
                WHEN monthly_income BETWEEN 30000 AND 75000 THEN '2. Middle Income (30k-75k)'
                WHEN monthly_income BETWEEN 75001 AND 150000 THEN '3. Upper Middle (75k-150k)'
                ELSE '4. High Net Worth (>150k)'
            END AS income_segment
        FROM customers
    )
    SELECT 
        s.income_segment,
        COUNT(DISTINCT s.customer_id) AS customer_count,
        COUNT(l.loan_id) AS total_loans,
        SUM(l.loan_amount) AS total_disbursed,
        ROUND(CAST(SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(l.loan_id) * 100, 2) AS default_rate_pct
    FROM IncomeSeg s
    JOIN loans l ON s.customer_id = l.customer_id
    GROUP BY s.income_segment;
    """
    pd.read_sql_query(q4, conn).to_csv(os.path.join(proc_dir, "risk_by_income_segment.csv"), index=False)
    
    # 5. Monthly Default Trend CSV
    q5 = """
    SELECT 
        SUBSTR(application_date, 1, 7) AS app_month,
        COUNT(application_id) AS total_applications,
        SUM(CASE WHEN application_status = 'Approved' THEN 1 ELSE 0 END) AS approved_count,
        SUM(default_target) AS expected_defaults,
        ROUND(CAST(SUM(default_target) AS FLOAT) / COUNT(application_id) * 100, 2) AS default_rate_pct
    FROM applications
    GROUP BY SUBSTR(application_date, 1, 7)
    ORDER BY app_month ASC;
    """
    pd.read_sql_query(q5, conn).to_csv(os.path.join(proc_dir, "monthly_default_trend.csv"), index=False)
    
    # 6. Customer Risk Segments CSV
    q6 = """
    SELECT 
        c.customer_id,
        c.city_tier,
        c.employment_type,
        c.monthly_income,
        ch.credit_score,
        ch.credit_utilization,
        ch.delinquency_count,
        ch.recent_delinquency,
        CASE 
            WHEN ch.credit_score < 600 OR ch.recent_delinquency = 1 THEN 'HIGH_RISK'
            WHEN ch.credit_score BETWEEN 600 AND 720 THEN 'MEDIUM_RISK'
            ELSE 'LOW_RISK'
        END AS risk_segment
    FROM customers c
    JOIN credit_history ch ON c.customer_id = ch.customer_id;
    """
    pd.read_sql_query(q6, conn).to_csv(os.path.join(proc_dir, "customer_risk_segments.csv"), index=False)
    
    # 7. Delinquency Analysis CSV
    q7 = """
    SELECT 
        d.severity,
        COUNT(d.delinquency_id) AS incident_count,
        COUNT(DISTINCT d.customer_id) AS impacted_borrowers,
        SUM(CASE WHEN d.days_past_due BETWEEN 30 AND 59 THEN 1 ELSE 0 END) AS dpd_30_59,
        SUM(CASE WHEN d.days_past_due BETWEEN 60 AND 89 THEN 1 ELSE 0 END) AS dpd_60_89,
        SUM(CASE WHEN d.days_past_due >= 90 THEN 1 ELSE 0 END) AS dpd_90_plus
    FROM delinquencies d
    GROUP BY d.severity;
    """
    pd.read_sql_query(q7, conn).to_csv(os.path.join(proc_dir, "delinquency_analysis.csv"), index=False)
    
    # 8. Exposure Analysis CSV
    q8 = """
    SELECT 
        l.loan_status,
        COUNT(l.loan_id) AS total_loans,
        SUM(l.loan_amount) AS total_exposure,
        AVG(l.loan_amount) AS avg_exposure
    FROM loans l
    GROUP BY l.loan_status;
    """
    pd.read_sql_query(q8, conn).to_csv(os.path.join(proc_dir, "exposure_analysis.csv"), index=False)
    
    conn.close()
    print("[SUCCESS] All 8 Power BI-ready CSV files exported to `data/processed/`!")

if __name__ == "__main__":
    export_portfolio_analytics()
