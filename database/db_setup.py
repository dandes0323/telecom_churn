import sqlite3
import pandas as pd
import os


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "telecom.db")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "telco_churn.csv")


def setup_database():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_csv(CSV_PATH)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Base de datos creada en {DB_PATH}")
    print(f"Cargados {len(df)} registros en la tabla 'customers'")


def get_data_from_db():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT customerID, gender, SeniorCitizen, tenure, PhoneService,
           InternetService, Contract, MonthlyCharges, TotalCharges, Churn
    FROM customers
    WHERE TotalCharges != ' '
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


if __name__ == "__main__":
    setup_database()
