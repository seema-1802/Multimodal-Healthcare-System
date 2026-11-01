import sqlite3

def init_db():
    conn = sqlite3.connect('healthcare.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT DEFAULT 'Guest',
        module_type TEXT NOT NULL,
        prediction_result TEXT NOT NULL,
        confidence REAL,
        additional_info TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print("SQLite Database 'healthcare.db' aur Table 'patient_predictions' create ho chuke hain!")

if __name__ == '__main__':
    init_db()