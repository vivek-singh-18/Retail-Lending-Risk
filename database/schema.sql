-- Retail Lending Risk Intelligence Database Schema
-- Compatible with PostgreSQL and SQLite standard ANSI SQL specs

DROP TABLE IF EXISTS delinquencies;
DROP TABLE IF EXISTS repayments;
DROP TABLE IF EXISTS loans;
DROP TABLE IF EXISTS applications;
DROP TABLE IF EXISTS credit_history;
DROP TABLE IF EXISTS employment_history;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    age INT NOT NULL CHECK (age >= 18 AND age <= 100),
    city_tier VARCHAR(20) NOT NULL,
    employment_type VARCHAR(50) NOT NULL,
    monthly_income NUMERIC(12, 2) NOT NULL CHECK (monthly_income > 0),
    employment_length NUMERIC(4, 1) NOT NULL CHECK (employment_length >= 0)
);

CREATE TABLE credit_history (
    customer_id VARCHAR(20) PRIMARY KEY,
    credit_score INT NOT NULL CHECK (credit_score >= 300 AND credit_score <= 850),
    active_loans INT NOT NULL DEFAULT 0,
    closed_loans INT NOT NULL DEFAULT 0,
    credit_utilization NUMERIC(5, 2) NOT NULL CHECK (credit_utilization >= 0 AND credit_utilization <= 1.5),
    delinquency_count INT NOT NULL DEFAULT 0,
    recent_delinquency INT NOT NULL DEFAULT 0 CHECK (recent_delinquency IN (0, 1)),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE TABLE employment_history (
    customer_id VARCHAR(20) PRIMARY KEY,
    employer_type VARCHAR(50) NOT NULL,
    employment_length NUMERIC(4, 1) NOT NULL,
    income_stability VARCHAR(20) NOT NULL CHECK (income_stability IN ('High', 'Medium', 'Low')),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE TABLE applications (
    application_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    application_date DATE NOT NULL,
    requested_amount NUMERIC(12, 2) NOT NULL CHECK (requested_amount > 0),
    loan_type VARCHAR(50) NOT NULL,
    application_status VARCHAR(20) NOT NULL CHECK (application_status IN ('Approved', 'Declined', 'Cancelled')),
    default_target INT NOT NULL CHECK (default_target IN (0, 1)),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE TABLE loans (
    loan_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    loan_type VARCHAR(50) NOT NULL,
    loan_amount NUMERIC(12, 2) NOT NULL CHECK (loan_amount > 0),
    tenure INT NOT NULL CHECK (tenure > 0),
    interest_rate NUMERIC(5, 2) NOT NULL CHECK (interest_rate > 0),
    application_date DATE NOT NULL,
    loan_status VARCHAR(20) NOT NULL CHECK (loan_status IN ('Active', 'Closed', 'Defaulted', 'Settled')),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

CREATE TABLE repayments (
    payment_id VARCHAR(20) PRIMARY KEY,
    loan_id VARCHAR(20) NOT NULL,
    payment_date DATE NOT NULL,
    due_amount NUMERIC(12, 2) NOT NULL,
    paid_amount NUMERIC(12, 2) NOT NULL,
    days_past_due INT NOT NULL DEFAULT 0 CHECK (days_past_due >= 0),
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
);

CREATE TABLE delinquencies (
    delinquency_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    loan_id VARCHAR(20) NOT NULL,
    days_past_due INT NOT NULL CHECK (days_past_due > 0),
    delinquency_date DATE NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('Minor', 'Moderate', 'Severe', 'Critical')),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE,
    FOREIGN KEY (loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
);

-- Optimization Indexes
CREATE INDEX idx_apps_customer ON applications(customer_id);
CREATE INDEX idx_apps_date ON applications(application_date);
CREATE INDEX idx_loans_customer ON loans(customer_id);
CREATE INDEX idx_loans_status ON loans(loan_status);
CREATE INDEX idx_repay_loan ON repayments(loan_id);
CREATE INDEX idx_delinq_customer ON delinquencies(customer_id);
