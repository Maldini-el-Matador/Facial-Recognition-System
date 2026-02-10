"""
FaceAccess & Loyalty API
FastAPI backend for facial recognition access control and loyalty program.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import os

import database as db
import face_service

# ============== APP SETUP ==============

app = FastAPI(
    title="FaceAccess & Loyalty API",
    description="Facial recognition for building access and loyalty programs",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve user photos
if os.path.exists("known_faces"):
    app.mount("/photos", StaticFiles(directory="known_faces"), name="photos")

# ============== MODELS ==============

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    image: str  # Base64 encoded image

class VerifyRequest(BaseModel):
    image: str  # Base64 encoded image

class AccessRequest(BaseModel):
    image: str  # Base64 encoded image
    location: Optional[str] = "Main Entrance"
    award_points: Optional[int] = 10

class LoyaltyRedeemRequest(BaseModel):
    user_id: str
    points: int
    description: str

# ============== HEALTH CHECK ==============

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "FaceAccess & Loyalty API",
        "version": "1.0.0",
        "face_recognition_available": face_service.FACE_RECOGNITION_AVAILABLE
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "users_count": len(db.get_all_users())}

# ============== FACE DETECTION ==============

@app.post("/api/detect")
async def detect_face(request: VerifyRequest):
    """Detect if a face is present in the image (for UI feedback)."""
    try:
        image_array = face_service.decode_base64_image(request.image)
        detected, face_info = face_service.detect_face(image_array)
        return {"success": True, **face_info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============== REGISTRATION ==============

@app.post("/api/register")
async def register_user(request: RegisterRequest):
    """Register a new user with their face."""
    success, result = face_service.register_face(
        name=request.name,
        email=request.email,
        base64_image=request.image
    )
    
    if success:
        return {"success": True, **result}
    else:
        raise HTTPException(status_code=400, detail=result)

# ============== VERIFICATION ==============

@app.post("/api/verify")
async def verify_face(request: VerifyRequest):
    """Verify a face against registered users."""
    success, result = face_service.verify_face(request.image)
    return {"success": success, **result}

# ============== ACCESS CONTROL ==============

@app.post("/api/access")
async def process_access(request: AccessRequest):
    """Process an access request - verify face, log access, award points."""
    success, result = face_service.process_access(
        base64_image=request.image,
        location=request.location,
        award_points=request.award_points
    )
    return {"success": success, **result}

# ============== USER MANAGEMENT ==============

@app.get("/api/users")
async def get_all_users():
    """Get all registered users (admin)."""
    users = db.get_all_users()
    # Remove sensitive data
    safe_users = []
    for user in users:
        safe_users.append({
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "loyalty_points": user["loyalty_points"],
            "tier": user["tier"],
            "total_visits": user["total_visits"],
            "created_at": user["created_at"],
            "photo_url": f"/photos/{user['id']}.jpg" if user.get("photo_path") else None
        })
    return {"users": safe_users, "count": len(safe_users)}

@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    """Get a specific user by ID."""
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "loyalty_points": user["loyalty_points"],
        "tier": user["tier"],
        "total_visits": user["total_visits"],
        "created_at": user["created_at"],
        "photo_url": f"/photos/{user['id']}.jpg"
    }

# ============== ACCESS LOGS ==============

@app.get("/api/logs")
async def get_access_logs(limit: int = 50):
    """Get recent access logs (admin)."""
    logs = db.get_recent_logs(limit)
    return {"logs": logs, "count": len(logs)}

@app.get("/api/logs/{user_id}")
async def get_user_logs(user_id: str):
    """Get access logs for a specific user."""
    logs = db.get_user_logs(user_id)
    return {"logs": logs, "count": len(logs)}

# ============== LOYALTY PROGRAM ==============

@app.post("/api/loyalty/redeem")
async def redeem_points(request: LoyaltyRedeemRequest):
    """Redeem loyalty points for a reward."""
    user = db.get_user(request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user["loyalty_points"] < request.points:
        raise HTTPException(status_code=400, detail="Insufficient points")
    
    # Deduct points (negative value)
    transaction = db.add_loyalty_transaction(
        request.user_id, 
        -request.points, 
        f"Redeemed: {request.description}"
    )
    
    updated_user = db.get_user(request.user_id)
    
    return {
        "success": True,
        "transaction": transaction,
        "remaining_points": updated_user["loyalty_points"],
        "message": f"Successfully redeemed {request.points} points!"
    }

@app.get("/api/loyalty/{user_id}")
async def get_loyalty_history(user_id: str):
    """Get loyalty transaction history for a user."""
    transactions = db.get_loyalty_transactions(user_id)
    user = db.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": user_id,
        "current_points": user["loyalty_points"],
        "tier": user["tier"],
        "transactions": transactions
    }

# ============== STATS (ADMIN DASHBOARD) ==============

@app.get("/api/stats")
async def get_stats():
    """Get overall statistics for admin dashboard."""
    users = db.get_all_users()
    logs = db.get_recent_logs(1000)
    
    total_points = sum(u["loyalty_points"] for u in users)
    total_visits = sum(u["total_visits"] for u in users)
    
    tier_counts = {"Bronze": 0, "Silver": 0, "Gold": 0, "Platinum": 0}
    for user in users:
        tier_counts[user.get("tier", "Bronze")] += 1
    
    return {
        "total_users": len(users),
        "total_access_events": len(logs),
        "total_visits": total_visits,
        "total_loyalty_points": total_points,
        "tier_distribution": tier_counts,
        "recent_activity": logs[:10]
    }

# ============== RUN ==============

if __name__ == "__main__":
    import uvicorn
    print("[*] Starting FaceAccess & Loyalty API...")
    print("[*] API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
