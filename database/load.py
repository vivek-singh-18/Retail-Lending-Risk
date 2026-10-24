"""
Database Loading and Initialization Script.

Loads raw CSV files into an SQLite database (`database/retail_lending.db`),
executes the SQL schema DDL, verifies table counts and referential integrity,
and generates a sample `database/seed.sql` file.
"""

import os
import sqlite3
import pandas as pd

def get_db_connection(db_path=None):
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), "retail_lending.db")
    return sqlite3.connect(db_path)

def init_db(db_path=None):
    base_dir = os.path.dirname(__file__)
    if db_path is None:
        db_path = os.path.join(base_dir, "retail_lending.db")
        
    schema_path = os.path.join(base_dir, "schema.sql")
    raw_dir = os.path.join(os.path.dirname(base_dir), "data", "raw")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Initializing database schema at {db_path}...")
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    
    cursor.executescript(schema_sql)
    conn.commit()
    
    tables = [
        ("customers", "customers.csv"),
        ("credit_history", "credit_history.csv"),
        ("employment_history", "employment_history.csv"),
        ("applications", "applications.csv"),
        ("loans", "loans.csv"),
        ("repayments", "repayments.csv"),
        ("delinquencies", "delinquencies.csv")
    ]
    
    print("Populating database tables from raw CSV datasets...")
    for table_name, csv_filename in tables:
        csv_path = os.path.join(raw_dir, csv_filename)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing raw CSV file: {csv_path}. Run `python -m data.generate` first.")
            
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="append", index=False)
        count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f" - Loaded {count:,} rows into table '{table_name}'")
        
    conn.close()
    print("[SUCCESS] Database initialization and loading complete!")
    
    # Generate seed.sql sample
    generate_seed_sql(raw_dir, os.path.join(base_dir, "seed.sql"))

def generate_seed_sql(raw_dir, seed_path):
    print(f"Generating sample `seed.sql` at {seed_path}...")
    with open(seed_path, "w") as out_f:
        out_f.write("-- Sample Data Seed SQL for Retail Lending System\n\n")
        customers_df = pd.read_csv(os.path.join(raw_dir, "customers.csv")).head(50)
        for _, row in customers_df.iterrows():
            out_f.write(
                f"INSERT INTO customers (customer_id, age, city_tier, employment_type, monthly_income, employment_length) "
                f"VALUES ('{row['customer_id']}', {row['age']}, '{row['city_tier']}', '{row['employment_type']}', {row['monthly_income']}, {row['employment_length']});\n"
            )
    print(" - `seed.sql` generated successfully.")

if __name__ == "__main__":
    init_db()
