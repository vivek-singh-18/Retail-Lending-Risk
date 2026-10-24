-- ====================================================================
-- Retail Lending Risk Intelligence: Analytical SQL Queries
-- Demonstrates Advanced SQL: CTEs, Window Functions, LAG, DENSE_RANK, 
-- Aggregations, CASE WHEN, GROUP BY, HAVING, and Date Functions.
-- ====================================================================

-- Query 1: Executive Portfolio Overview Metrics
SELECT 
    COUNT(DISTINCT l.loan_id) AS total_loans_issued,
    SUM(l.loan_amount) AS total_disbursed_amount,
    AVG(l.loan_amount) AS avg_loan_amount,
    SUM(CASE WHEN l.loan_status = 'Active' THEN l.loan_amount ELSE 0 END) AS active_outstanding_exposure,
    SUM(CASE WHEN l.loan_status = 'Defaulted' THEN l.loan_amount ELSE 0 END) AS defaulted_exposure,
    ROUND(CAST(SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(l.loan_id) * 100, 2) AS portfolio_default_rate_pct
FROM loans l;


-- Query 2: Risk Profile by Loan Product
SELECT 
    l.loan_type,
    COUNT(l.loan_id) AS total_loans,
    SUM(l.loan_amount) AS total_volume,
    AVG(l.interest_rate) AS avg_interest_rate,
    SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS default_count,
    ROUND(CAST(SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(l.loan_id) * 100, 2) AS default_rate_pct
FROM loans l
GROUP BY l.loan_type
ORDER BY default_rate_pct DESC;


-- Query 3: Geographical Exposure and Default Risk by City Tier
SELECT 
    c.city_tier,
    COUNT(DISTINCT c.customer_id) AS customer_count,
    COUNT(l.loan_id) AS total_loans,
    SUM(l.loan_amount) AS total_exposure,
    ROUND(AVG(c.monthly_income), 2) AS avg_monthly_income,
    ROUND(CAST(SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(l.loan_id) * 100, 2) AS city_default_rate_pct
FROM customers c
JOIN loans l ON c.customer_id = l.customer_id
GROUP BY c.city_tier
ORDER BY total_exposure DESC;


-- Query 4: Risk Segmentation by Income Bracket
WITH IncomeSegments AS (
    SELECT 
        c.customer_id,
        c.monthly_income,
        CASE 
            WHEN c.monthly_income < 30000 THEN '1. Low Income (<30k)'
            WHEN c.monthly_income BETWEEN 30000 AND 75000 THEN '2. Middle Income (30k-75k)'
            WHEN c.monthly_income BETWEEN 75001 AND 150000 THEN '3. Upper Middle (75k-150k)'
            ELSE '4. High Net Worth (>150k)'
        END AS income_segment
    FROM customers c
)
SELECT 
    s.income_segment,
    COUNT(DISTINCT s.customer_id) AS borrower_count,
    COUNT(l.loan_id) AS total_loans,
    SUM(l.loan_amount) AS total_disbursed,
    ROUND(CAST(SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(l.loan_id) * 100, 2) AS default_rate_pct
FROM IncomeSegments s
JOIN loans l ON s.customer_id = l.customer_id
GROUP BY s.income_segment
ORDER BY s.income_segment;


-- Query 5: Top 20 High-Risk / High-Exposure Borrowers (Ranked by Risk Score & Exposure)
WITH BorrowerRisk AS (
    SELECT 
        c.customer_id,
        c.monthly_income,
        c.employment_type,
        ch.credit_score,
        ch.credit_utilization,
        ch.delinquency_count,
        ch.recent_delinquency,
        SUM(l.loan_amount) AS total_loan_exposure,
        DENSE_RANK() OVER (ORDER BY ch.credit_score ASC, SUM(l.loan_amount) DESC) AS risk_rank
    FROM customers c
    JOIN credit_history ch ON c.customer_id = ch.customer_id
    JOIN loans l ON c.customer_id = l.customer_id
    WHERE l.loan_status = 'Active'
    GROUP BY c.customer_id, c.monthly_income, c.employment_type, ch.credit_score, ch.credit_utilization, ch.delinquency_count, ch.recent_delinquency
)
SELECT *
FROM BorrowerRisk
WHERE risk_rank <= 20
ORDER BY risk_rank;


-- Query 6: Borrowers with Multiple Active Loans and High Utilization
SELECT 
    c.customer_id,
    c.employment_type,
    ch.credit_score,
    ch.active_loans,
    ch.credit_utilization,
    SUM(l.loan_amount) AS aggregate_active_exposure
FROM customers c
JOIN credit_history ch ON c.customer_id = ch.customer_id
JOIN loans l ON c.customer_id = l.customer_id
WHERE ch.active_loans >= 2 AND ch.credit_utilization > 0.60 AND l.loan_status = 'Active'
GROUP BY c.customer_id, c.employment_type, ch.credit_score, ch.active_loans, ch.credit_utilization
HAVING SUM(l.loan_amount) > 300000
ORDER BY aggregate_active_exposure DESC;


-- Query 7: Monthly Application and Default Trend (Temporal Trend)
SELECT 
    SUBSTR(a.application_date, 1, 7) AS app_month,
    COUNT(a.application_id) AS total_applications,
    SUM(CASE WHEN a.application_status = 'Approved' THEN 1 ELSE 0 END) AS approved_count,
    SUM(a.default_target) AS expected_defaults,
    ROUND(CAST(SUM(a.default_target) AS FLOAT) / COUNT(a.application_id) * 100, 2) AS monthly_default_rate_pct
FROM applications a
GROUP BY SUBSTR(a.application_date, 1, 7)
ORDER BY app_month ASC;


-- Query 8: Delinquency Escalation Analysis (Using LAG Window Function)
WITH DelinquencySequence AS (
    SELECT 
        d.customer_id,
        d.loan_id,
        d.delinquency_date,
        d.days_past_due,
        LAG(d.days_past_due, 1, 0) OVER (PARTITION BY d.customer_id ORDER BY d.delinquency_date ASC) AS prev_dpd
    FROM delinquencies d
)
SELECT 
    s.customer_id,
    s.loan_id,
    s.delinquency_date,
    s.prev_dpd,
    s.days_past_due AS current_dpd,
    (s.days_past_due - s.prev_dpd) AS dpd_escalation
FROM DelinquencySequence s
WHERE s.days_past_due > s.prev_dpd AND s.prev_dpd > 0
ORDER BY dpd_escalation DESC;


-- Query 9: Repayment Behavior & Payment Timeliness
SELECT 
    l.loan_type,
    COUNT(r.payment_id) AS total_repayments,
    SUM(CASE WHEN r.days_past_due = 0 THEN 1 ELSE 0 END) AS on_time_payments,
    SUM(CASE WHEN r.days_past_due BETWEEN 1 AND 29 THEN 1 ELSE 0 END) AS dpd_1_29,
    SUM(CASE WHEN r.days_past_due BETWEEN 30 AND 59 THEN 1 ELSE 0 END) AS dpd_30_59,
    SUM(CASE WHEN r.days_past_due >= 60 THEN 1 ELSE 0 END) AS dpd_60_plus,
    ROUND(CAST(SUM(CASE WHEN r.days_past_due > 0 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(r.payment_id) * 100, 2) AS late_payment_rate_pct
FROM repayments r
JOIN loans l ON r.loan_id = l.loan_id
GROUP BY l.loan_type;


-- Query 10: Portfolio Risk Concentration Analysis
SELECT 
    c.employment_type,
    eh.income_stability,
    COUNT(DISTINCT c.customer_id) AS total_borrowers,
    SUM(l.loan_amount) AS total_exposure,
    ROUND(CAST(SUM(CASE WHEN l.loan_status = 'Defaulted' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(l.loan_id) * 100, 2) AS default_rate_pct
FROM customers c
JOIN employment_history eh ON c.customer_id = eh.customer_id
JOIN loans l ON c.customer_id = l.customer_id
GROUP BY c.employment_type, eh.income_stability
ORDER BY total_exposure DESC;
