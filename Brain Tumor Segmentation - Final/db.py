import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def create_user(full_name, email, password):
    """
    Creates a new user in database.
    Returns (True, user_dict) if successful, or (False, error_message) if email exists or fails.
    """
    email_clean = email.strip().lower()
    full_name_clean = full_name.strip()
    
    if not email_clean or not password or not full_name_clean:
        return False, "All fields are required."
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
        
    pwd_hash = generate_password_hash(password)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)",
            (full_name_clean, email_clean, pwd_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        return True, {
            'id': user_id,
            'full_name': full_name_clean,
            'email': email_clean
        }
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except Exception as e:
        return False, f"Database error: {str(e)}"

def verify_user(email, password):
    """
    Verifies user credentials.
    Returns (user_dict, None) if verified, or (None, error_message).
    """
    email_clean = email.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email_clean,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        return None, "No account found with this email."
        
    if check_password_hash(user['password_hash'], password):
        return {
            'id': user['id'],
            'full_name': user['full_name'],
            'email': user['email']
        }, None
    else:
        return None, "Incorrect password. Please try again."

def get_user_by_id(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_name, email, created_at FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return dict(user)
    return None

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully at:", DB_PATH)
