import sqlite3

DATABASE_NAME = 'healthcare.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def log_prediction(patient_name, module_type, prediction_result, confidence, additional_info=""):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patient_predictions (patient_name, module_type, prediction_result, confidence, additional_info)
            VALUES (?, ?, ?, ?, ?)
        ''', (patient_name, module_type, prediction_result, confidence, additional_info))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database logging error: {e}")

def fetch_all_predictions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patient_predictions ORDER BY created_at DESC')
    records = cursor.fetchall()
    conn.close()
    return records