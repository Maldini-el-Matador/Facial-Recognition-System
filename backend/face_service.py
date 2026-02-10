"""
Face Recognition Service
Handles face detection, encoding, and matching.
"""
import os
import io
import base64
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image

# Try to import face_recognition, fall back to mock if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("[WARNING] face_recognition not installed. Using mock mode for demo.")

import database as db

# Threshold for face matching (lower = stricter)
FACE_MATCH_TOLERANCE = 0.6

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Convert base64 image string to numpy array."""
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return np.array(image)

def save_user_photo(user_id: str, image_array: np.ndarray) -> str:
    """Save user photo to disk."""
    photo_dir = "known_faces"
    os.makedirs(photo_dir, exist_ok=True)
    
    photo_path = os.path.join(photo_dir, f"{user_id}.jpg")
    image = Image.fromarray(image_array)
    image.save(photo_path, "JPEG", quality=90)
    
    return photo_path

def get_face_encoding(image_array: np.ndarray) -> Optional[np.ndarray]:
    """Extract face encoding from image."""
    if not FACE_RECOGNITION_AVAILABLE:
        # Mock: return random encoding for demo
        return np.random.rand(128)
    
    # Detect face locations
    face_locations = face_recognition.face_locations(image_array)
    
    if not face_locations:
        return None
    
    # Get encoding for the first face found
    encodings = face_recognition.face_encodings(image_array, face_locations)
    
    if encodings:
        return encodings[0]
    return None

def detect_face(image_array: np.ndarray) -> Tuple[bool, dict]:
    """
    Detect if a face is present in the image.
    Returns (success, face_info)
    """
    if not FACE_RECOGNITION_AVAILABLE:
        # Mock: always detect a face for demo
        h, w = image_array.shape[:2]
        return True, {
            "detected": True,
            "location": {"top": h//4, "right": 3*w//4, "bottom": 3*h//4, "left": w//4},
            "confidence": 0.95
        }
    
    face_locations = face_recognition.face_locations(image_array)
    
    if face_locations:
        top, right, bottom, left = face_locations[0]
        return True, {
            "detected": True,
            "location": {"top": top, "right": right, "bottom": bottom, "left": left},
            "confidence": 0.95
        }
    
    return False, {"detected": False, "location": None, "confidence": 0}

def register_face(name: str, email: str, base64_image: str) -> Tuple[bool, dict]:
    """
    Register a new user with their face.
    Returns (success, result)
    """
    # Check if email already exists
    existing = db.get_user_by_email(email)
    if existing:
        return False, {"error": "Email already registered", "code": "EMAIL_EXISTS"}
    
    # Decode image
    try:
        image_array = decode_base64_image(base64_image)
    except Exception as e:
        return False, {"error": f"Invalid image: {str(e)}", "code": "INVALID_IMAGE"}
    
    # Detect face
    face_detected, face_info = detect_face(image_array)
    if not face_detected:
        return False, {"error": "No face detected in image", "code": "NO_FACE"}
    
    # Get face encoding
    encoding = get_face_encoding(image_array)
    if encoding is None:
        return False, {"error": "Could not encode face", "code": "ENCODING_FAILED"}
    
    # Generate user ID and save photo
    import uuid
    user_id = str(uuid.uuid4())[:8]
    photo_path = save_user_photo(user_id, image_array)
    
    # Create user in database
    user = db.create_user(name, email, encoding, photo_path)
    
    # Log the registration as first access
    db.log_access(user["id"], user["name"], "registration", "Registration Kiosk")
    
    # Award welcome bonus
    db.add_loyalty_transaction(user["id"], 0, "Welcome bonus awarded!")  # 100 points already added in create_user
    
    return True, {
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "loyalty_points": user["loyalty_points"],
            "tier": user["tier"]
        },
        "message": "Registration successful! Welcome bonus: 100 points"
    }

def verify_face(base64_image: str) -> Tuple[bool, dict]:
    """
    Verify a face against registered users.
    Returns (success, result)
    """
    # Decode image
    try:
        image_array = decode_base64_image(base64_image)
    except Exception as e:
        return False, {"error": f"Invalid image: {str(e)}", "code": "INVALID_IMAGE"}
    
    # Detect face
    face_detected, face_info = detect_face(image_array)
    if not face_detected:
        return False, {"error": "No face detected", "code": "NO_FACE", "face_info": face_info}
    
    # Get face encoding
    encoding = get_face_encoding(image_array)
    if encoding is None:
        return False, {"error": "Could not encode face", "code": "ENCODING_FAILED"}
    
    # Get all known face encodings
    known_encodings = db.get_all_face_encodings()
    
    if not known_encodings:
        return False, {"error": "No registered users", "code": "NO_USERS"}
    
    # Compare with all known faces
    best_match = None
    best_distance = float('inf')
    
    for user_id, known_encoding in known_encodings:
        if FACE_RECOGNITION_AVAILABLE:
            # Calculate face distance
            distance = face_recognition.face_distance([known_encoding], encoding)[0]
        else:
            # Mock: random distance for demo (simulate 70% match rate)
            distance = np.random.uniform(0.3, 0.7)
        
        if distance < best_distance:
            best_distance = distance
            best_match = user_id
    
    # Check if match is good enough
    if best_distance <= FACE_MATCH_TOLERANCE and best_match:
        user = db.get_user(best_match)
        if user:
            confidence = max(0, min(100, int((1 - best_distance) * 100)))
            return True, {
                "matched": True,
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "loyalty_points": user["loyalty_points"],
                    "tier": user["tier"],
                    "total_visits": user["total_visits"]
                },
                "confidence": confidence,
                "face_info": face_info
            }
    
    return False, {
        "matched": False,
        "error": "Face not recognized",
        "code": "NO_MATCH",
        "confidence": max(0, int((1 - best_distance) * 100)) if best_distance != float('inf') else 0
    }

def process_access(base64_image: str, location: str = "Main Entrance", award_points: int = 10) -> Tuple[bool, dict]:
    """
    Process an access request - verify face and log access.
    Returns (success, result)
    """
    success, result = verify_face(base64_image)
    
    if success and result.get("matched"):
        user = result["user"]
        
        # Log the access
        log = db.log_access(user["id"], user["name"], "entry", location)
        
        # Award loyalty points for visit
        transaction = db.add_loyalty_transaction(user["id"], award_points, f"Visit to {location}")
        
        # Get updated user info
        updated_user = db.get_user(user["id"])
        
        return True, {
            "access_granted": True,
            "user": {
                "id": updated_user["id"],
                "name": updated_user["name"],
                "loyalty_points": updated_user["loyalty_points"],
                "tier": updated_user["tier"],
                "total_visits": updated_user["total_visits"]
            },
            "points_awarded": award_points,
            "confidence": result["confidence"],
            "log_id": log["id"],
            "message": f"Welcome back, {updated_user['name']}! +{award_points} points"
        }
    
    return False, {
        "access_granted": False,
        "error": result.get("error", "Access denied"),
        "code": result.get("code", "DENIED")
    }
