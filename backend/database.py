"""
In-memory database for hackathon demo.
In production, replace with PostgreSQL/Supabase.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# In-memory storage
users_db: Dict[str, dict] = {}
access_logs: List[dict] = []
loyalty_transactions: List[dict] = []

# File path for persistence (optional)
DATA_FILE = "db_data.json"

def generate_id() -> str:
    """Generate a simple unique ID."""
    import uuid
    return str(uuid.uuid4())[:8]

# ============== USER OPERATIONS ==============

def create_user(name: str, email: str, face_encoding: np.ndarray, photo_path: str) -> dict:
    """Register a new user with their face encoding."""
    user_id = generate_id()
    user = {
        "id": user_id,
        "name": name,
        "email": email,
        "face_encoding": face_encoding.tolist(),  # Convert numpy array to list for JSON
        "photo_path": photo_path,
        "loyalty_points": 100,  # Welcome bonus!
        "tier": "Bronze",
        "created_at": datetime.now().isoformat(),
        "total_visits": 0
    }
    users_db[user_id] = user
    save_data()
    return user

def get_user(user_id: str) -> Optional[dict]:
    """Get user by ID."""
    return users_db.get(user_id)

def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email."""
    for user in users_db.values():
        if user["email"] == email:
            return user
    return None

def get_all_users() -> List[dict]:
    """Get all registered users."""
    return list(users_db.values())

def get_all_face_encodings() -> List[tuple]:
    """Get all face encodings with user IDs for matching."""
    encodings = []
    for user_id, user in users_db.items():
        if user.get("face_encoding"):
            encodings.append((user_id, np.array(user["face_encoding"])))
    return encodings

def update_user_points(user_id: str, points: int) -> Optional[dict]:
    """Add or subtract loyalty points."""
    if user_id in users_db:
        users_db[user_id]["loyalty_points"] += points
        # Update tier based on total points
        total = users_db[user_id]["loyalty_points"]
        if total >= 1000:
            users_db[user_id]["tier"] = "Platinum"
        elif total >= 500:
            users_db[user_id]["tier"] = "Gold"
        elif total >= 200:
            users_db[user_id]["tier"] = "Silver"
        save_data()
        return users_db[user_id]
    return None

def increment_visits(user_id: str) -> None:
    """Increment visit count for a user."""
    if user_id in users_db:
        users_db[user_id]["total_visits"] += 1
        save_data()

# ============== ACCESS LOG OPERATIONS ==============

def log_access(user_id: str, user_name: str, access_type: str, location: str = "Main Entrance") -> dict:
    """Log an access event."""
    log_entry = {
        "id": generate_id(),
        "user_id": user_id,
        "user_name": user_name,
        "access_type": access_type,  # "entry", "exit", "loyalty_scan"
        "location": location,
        "timestamp": datetime.now().isoformat(),
        "status": "granted"
    }
    access_logs.append(log_entry)
    increment_visits(user_id)
    save_data()
    return log_entry

def get_recent_logs(limit: int = 50) -> List[dict]:
    """Get recent access logs."""
    return sorted(access_logs, key=lambda x: x["timestamp"], reverse=True)[:limit]

def get_user_logs(user_id: str) -> List[dict]:
    """Get access logs for a specific user."""
    return [log for log in access_logs if log["user_id"] == user_id]

# ============== LOYALTY OPERATIONS ==============

def add_loyalty_transaction(user_id: str, points: int, description: str) -> dict:
    """Record a loyalty transaction."""
    transaction = {
        "id": generate_id(),
        "user_id": user_id,
        "points": points,
        "description": description,
        "timestamp": datetime.now().isoformat()
    }
    loyalty_transactions.append(transaction)
    update_user_points(user_id, points)
    save_data()
    return transaction

def get_loyalty_transactions(user_id: str) -> List[dict]:
    """Get loyalty transactions for a user."""
    return [t for t in loyalty_transactions if t["user_id"] == user_id]

# ============== PERSISTENCE ==============

def save_data():
    """Save data to JSON file."""
    try:
        data = {
            "users": {uid: {**u, "face_encoding": u["face_encoding"]} for uid, u in users_db.items()},
            "access_logs": access_logs,
            "loyalty_transactions": loyalty_transactions
        }
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving data: {e}")

def load_data():
    """Load data from JSON file."""
    global users_db, access_logs, loyalty_transactions
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                users_db = data.get("users", {})
                access_logs = data.get("access_logs", [])
                loyalty_transactions = data.get("loyalty_transactions", [])
                print(f"Loaded {len(users_db)} users from database")
        except Exception as e:
            print(f"Error loading data: {e}")

# Load existing data on module import
load_data()
