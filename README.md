# FaceAccess Pro 🔐

> Cutting-edge facial recognition for seamless building access and loyalty programs.

![FaceAccess Pro](https://img.shields.io/badge/Hackathon-2026-purple)
![Status](https://img.shields.io/badge/Status-Demo%20Ready-green)

## Features

- ✨ **Face Scan Access** - Instant building entry with real-time recognition
- 👤 **User Registration** - Quick enrollment with welcome bonus
- 📊 **Admin Dashboard** - Monitor access logs and analytics
- ⭐ **Loyalty Program** - Earn points with every visit
- 🎨 **Premium UI** - Stunning animations and glass morphism design

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Framer Motion |
| Backend | Python, FastAPI |
| Face Recognition | face_recognition library (dlib) |
| Database | In-memory + JSON persistence |

## Quick Start

### 1. Start the Backend

```bash
# Navigate to backend
cd backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

The API will be available at `http://localhost:8000`  
📖 API Docs: `http://localhost:8000/docs`

### 2. Start the Frontend

```bash
# Navigate to frontend (in new terminal)
cd frontend

# Install dependencies (if not done)
npm install

# Run dev server
npm run dev
```

The app will be available at `http://localhost:3000`

## Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Landing page with feature cards |
| Face Scan | `/scan` | Scan face for building access |
| Register | `/register` | Register new user with face |
| Dashboard | `/dashboard` | Admin panel with stats |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/register` | Register new user |
| POST | `/api/verify` | Verify face identity |
| POST | `/api/access` | Access request (verify + log + points) |
| GET | `/api/users` | List all users |
| GET | `/api/logs` | Access logs |
| GET | `/api/stats` | Dashboard statistics |

## Demo Flow

1. **Register** - Go to `/register`, enter name/email, take photo
2. **Scan** - Go to `/scan`, look at camera, click "Scan Face"
3. **Dashboard** - Go to `/dashboard` to see stats and users

## Notes for Hackathon Demo

- The app works without `face_recognition` library installed (uses mock mode)
- Real face matching requires `dlib` and `face_recognition` (can be tricky to install on Windows)
- For demo purposes, the mock mode simulates ~70% match rate

## Troubleshooting

**Camera not working?**
- Allow camera permissions in browser
- Use HTTPS or localhost (required for webcam access)

**Backend connection error?**
- Make sure backend is running on port 8000
- Check CORS settings if using different ports

**face_recognition install fails?**
- The app works in mock mode without it
- For real face matching, see: https://github.com/ageitgey/face_recognition

---

Built with ❤️ for Hackathon 2026
