"""
Face Recognition Service
Handles face detection, encoding, and matching.

Uses a custom PyTorch model trained on LFW dataset for face embeddings.
Falls back to random embeddings if model is not yet trained.
"""
import os
import io
import base64
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image

from model_inference import get_model, FaceRecognitionModel

import database as db

# Threshold for face matching (cosine similarity, higher = stricter)
FACE_MATCH_THRESHOLD = 0.55

# Initialize the model
face_model = get_model()
FACE_RECOGNITION_AVAILABLE = face_model.model_loaded


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
    """Extract face encoding (embedding) from image using trained model."""
    # Detect faces first
    faces = face_model.detect_faces(image_array)
    
    if not faces:
        return None
    
    # Get embedding for the first (largest/most confident) face
    face_box = faces[0]
    embedding = face_model.get_embedding(image_array, face_box=face_box)
    
    return embedding


def detect_face(image_array: np.ndarray) -> Tuple[bool, dict]:
    """
    Detect if a face is present in the image.
    Returns (success, face_info)
    """
    faces = face_model.detect_faces(image_array)
    
    if faces:
        face = faces[0]
        return True, {
            "detected": True,
            "location": {
                "top": face["top"],
                "right": face["right"],
                "bottom": face["bottom"],
                "left": face["left"],
            },
            "confidence": face.get("confidence", 0.95),
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
    db.add_loyalty_transaction(user["id"], 0, "Welcome bonus awarded!")
    
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
    Verify a face against registered users using cosine similarity.
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
    
    # Compare with all known faces using cosine similarity
    best_match = None
    best_similarity = -1.0
    
    for user_id, known_encoding in known_encodings:
        similarity = face_model.compare_faces(known_encoding, encoding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = user_id
    
    # Check if match is good enough
    if best_similarity >= FACE_MATCH_THRESHOLD and best_match:
        user = db.get_user(best_match)
        if user:
            confidence = max(0, min(100, int(best_similarity * 100)))
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
        "confidence": max(0, int(best_similarity * 100)) if best_similarity >= 0 else 0
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
