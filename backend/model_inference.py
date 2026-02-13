"""
Face Recognition Inference Module
Loads the trained model and provides face detection + embedding extraction.

This module replaces the dlib-based face_recognition library with a custom
PyTorch model trained on LFW.
"""

import os
import json
import numpy as np
from typing import Optional, Tuple, List

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ======================== CONFIG ========================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition_model.pth")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# OpenCV DNN face detector model files
OPENCV_DNN_PROTOTXT = os.path.join(MODEL_DIR, "deploy.prototxt")
OPENCV_DNN_CAFFEMODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Face detection confidence threshold
FACE_DETECTION_CONFIDENCE = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== MODEL ARCHITECTURE ========================

class FaceEmbeddingNet(nn.Module):
    """Same architecture as training - needed to load weights."""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        resnet = models.resnet18(weights=None)  # No pretrained weights needed for inference
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
    
    def get_embedding(self, x):
        """Extract face embedding."""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
    
    def forward(self, x):
        return self.get_embedding(x)


# ======================== INFERENCE CLASS ========================

class FaceRecognitionModel:
    """
    Face Recognition inference engine.
    Provides face detection and embedding extraction using the trained model.
    """
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.face_detector = None
        self.detector_type = None  # "dnn" or "haar"
        self.embedding_dim = 128
        self.metadata = {}
        
        # Image preprocessing (must match training transforms)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self._load_model()
        self._load_face_detector()
    
    def _load_model(self):
        """Load the trained face embedding model."""
        if not os.path.exists(MODEL_PATH):
            print(f"[WARNING] Model not found at {MODEL_PATH}. Run train_model.py first.")
            return
        
        try:
            # Load metadata
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, "r") as f:
                    self.metadata = json.load(f)
                self.embedding_dim = self.metadata.get("embedding_dim", 128)
            
            # Load model
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            
            self.model = FaceEmbeddingNet(embedding_dim=self.embedding_dim).to(DEVICE)
            self.model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
            self.model.embedding.load_state_dict(checkpoint["embedding_state_dict"])
            self.model.eval()
            
            self.model_loaded = True
            print(f"[+] Face recognition model loaded successfully!")
            print(f"    Embedding dim: {self.embedding_dim}")
            print(f"    Val accuracy: {checkpoint.get('best_val_acc', 'N/A')}%")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model_loaded = False
    
    def _load_face_detector(self):
        """Load face detector (OpenCV DNN preferred, Haar cascade fallback)."""
        # Try OpenCV DNN face detector first
        if os.path.exists(OPENCV_DNN_PROTOTXT) and os.path.exists(OPENCV_DNN_CAFFEMODEL):
            try:
                self.face_detector = cv2.dnn.readNetFromCaffe(OPENCV_DNN_PROTOTXT, OPENCV_DNN_CAFFEMODEL)
                self.detector_type = "dnn"
                print("[+] Face detector: OpenCV DNN (SSD)")
                return
            except Exception as e:
                print(f"[WARNING] DNN detector failed: {e}")
        
        # Fallback to Haar cascade (ships with OpenCV)
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(haar_path):
            self.face_detector = cv2.CascadeClassifier(haar_path)
            self.detector_type = "haar"
            print("[+] Face detector: OpenCV Haar Cascade")
        else:
            print("[WARNING] No face detector available!")
            self.detector_type = None
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in an image.
        
        Args:
            image: numpy array (RGB format, HxWxC)
            
        Returns:
            List of face dictionaries with keys: top, right, bottom, left, confidence
        """
        if self.face_detector is None:
            # No detector - assume the entire image is a face
            h, w = image.shape[:2]
            return [{"top": 0, "right": w, "bottom": h, "left": 0, "confidence": 0.5}]
        
        faces = []
        
        if self.detector_type == "dnn":
            h, w = image.shape[:2]
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            blob = cv2.dnn.blobFromImage(bgr_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > FACE_DETECTION_CONFIDENCE:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    left, top, right, bottom = box.astype(int)
                    # Clamp to image bounds
                    left = max(0, left)
                    top = max(0, top)
                    right = min(w, right)
                    bottom = min(h, bottom)
                    faces.append({
                        "top": int(top),
                        "right": int(right),
                        "bottom": int(bottom),
                        "left": int(left),
                        "confidence": float(confidence),
                    })
        
        elif self.detector_type == "haar":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            detections = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            for (x, y, w_box, h_box) in detections:
                faces.append({
                    "top": int(y),
                    "right": int(x + w_box),
                    "bottom": int(y + h_box),
                    "left": int(x),
                    "confidence": 0.9,  # Haar doesn't provide confidence
                })
        
        return faces
    
    def get_embedding(self, image: np.ndarray, face_box: Optional[dict] = None) -> Optional[np.ndarray]:
        """
        Extract a 128-dim face embedding from an image.
        
        Args:
            image: numpy array (RGB, HxWxC)
            face_box: optional dict with top/right/bottom/left keys to crop face
            
        Returns:
            128-dim numpy array or None if model not loaded
        """
        if not self.model_loaded:
            # Fallback: return random embedding (should not happen in production)
            print("[WARNING] Model not loaded, returning random embedding")
            return np.random.rand(self.embedding_dim).astype(np.float32)
        
        # Crop face region if box provided
        if face_box:
            top = max(0, face_box["top"])
            bottom = min(image.shape[0], face_box["bottom"])
            left = max(0, face_box["left"])
            right = min(image.shape[1], face_box["right"])
            
            # Add some margin around the face (20%)
            h = bottom - top
            w = right - left
            margin_h = int(h * 0.2)
            margin_w = int(w * 0.2)
            
            top = max(0, top - margin_h)
            bottom = min(image.shape[0], bottom + margin_h)
            left = max(0, left - margin_w)
            right = min(image.shape[1], right + margin_w)
            
            face_image = image[top:bottom, left:right]
        else:
            face_image = image
        
        # Check if face crop is valid
        if face_image.size == 0 or face_image.shape[0] < 10 or face_image.shape[1] < 10:
            return None
        
        # Convert to PIL Image for transforms
        pil_image = Image.fromarray(face_image)
        
        # Apply transforms and add batch dimension
        tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(tensor)
        
        return embedding.cpu().numpy().flatten()
    
    @staticmethod
    def compare_faces(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity.
        
        Returns:
            Similarity score between 0 and 1 (higher = more similar)
        """
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Clamp to [0, 1]
        return float(max(0.0, min(1.0, similarity)))
    
    @staticmethod
    def similarity_to_distance(similarity: float) -> float:
        """Convert cosine similarity to a distance metric (compatible with face_recognition API)."""
        return 1.0 - similarity


# ======================== SINGLETON ========================

# Global model instance (loaded once at module import)
_model_instance = None


def get_model() -> FaceRecognitionModel:
    """Get or create the global model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = FaceRecognitionModel()
    return _model_instance


if __name__ == "__main__":
    # Quick test
    print("\n--- Testing Face Recognition Model ---\n")
    model = get_model()
    
    print(f"\nModel loaded: {model.model_loaded}")
    print(f"Detector type: {model.detector_type}")
    print(f"Embedding dim: {model.embedding_dim}")
    
    if model.model_loaded:
        # Test with a dummy image
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        emb = model.get_embedding(dummy)
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding norm: {np.linalg.norm(emb):.4f}")
        
        # Test similarity
        emb2 = model.get_embedding(dummy)
        sim = model.compare_faces(emb, emb2)
        print(f"Self-similarity: {sim:.4f}")
    
    print("\n--- Test Complete ---")
