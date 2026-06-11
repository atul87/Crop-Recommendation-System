import os
import json
import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "crop_recommendation")

def get_connection(include_db=True):
    """Establish connection to MySQL server"""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE if include_db else None,
        cursorclass=pymysql.cursors.DictCursor
    )

def init_db():
    """Initialize database and tables"""
    try:
        # Connect to MySQL server without database first (to create db if missing)
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`")
        conn.close()
        
        # Now connect to the database and create the table
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `predictions` (
                `id` INT AUTO_INCREMENT PRIMARY KEY,
                `prediction_type` VARCHAR(50) NOT NULL,
                `inputs` TEXT NOT NULL,
                `result` VARCHAR(100) NOT NULL,
                `confidence` FLOAT NOT NULL,
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print("Database initialized successfully!")
        return True
    except Exception as e:
        print(f"Database initialization failed: {e}")
        return False

def save_prediction(prediction_type, inputs_dict, result, confidence):
    """Save prediction request into database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        inputs_json = json.dumps(inputs_dict)
        cursor.execute(
            "INSERT INTO `predictions` (`prediction_type`, `inputs`, `result`, `confidence`) VALUES (%s, %s, %s, %s)",
            (prediction_type, inputs_json, result, float(confidence))
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving prediction to database: {e}")
        return False

def get_predictions_history():
    """Retrieve all prediction records"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM `predictions` ORDER BY `created_at` DESC")
        records = cursor.fetchall()
        conn.close()
        
        # Parse inputs JSON back to dicts
        for r in records:
            try:
                r['inputs'] = json.loads(r['inputs'])
            except:
                pass
        return records
    except Exception as e:
        print(f"Error retrieving prediction history: {e}")
        return []
