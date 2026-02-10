"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    UserPlus,
    Camera,
    CheckCircle2,
    RefreshCw,
    ArrowLeft,
    Mail,
    User,
    Star,
    Sparkles,
} from "lucide-react";
import Link from "next/link";
import { registerUser, User as UserType } from "@/lib/api";

type RegisterState = "form" | "capture" | "processing" | "success" | "error";

export default function RegisterPage() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const [registerState, setRegisterState] = useState<RegisterState>("form");
    const [name, setName] = useState("");
    const [email, setEmail] = useState("");
    const [capturedImage, setCapturedImage] = useState<string | null>(null);
    const [registeredUser, setRegisteredUser] = useState<UserType | null>(null);
    const [errorMessage, setErrorMessage] = useState("");
    const [isCameraReady, setIsCameraReady] = useState(false);

    // Initialize camera
    const startCamera = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "user", width: 640, height: 480 },
            });
            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    setIsCameraReady(true);
                };
            }
        } catch (err) {
            console.error("Camera error:", err);
            setErrorMessage("Could not access camera");
        }
    }, []);

    // Stop camera
    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
    }, []);

    // Proceed to capture
    const proceedToCapture = () => {
        if (!name.trim() || !email.trim()) {
            setErrorMessage("Please fill in all fields");
            return;
        }
        if (!email.includes("@")) {
            setErrorMessage("Please enter a valid email");
            return;
        }
        setErrorMessage("");
        setRegisterState("capture");
        startCamera();
    };

    // Capture photo
    const capturePhoto = useCallback(() => {
        if (!videoRef.current || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const video = videoRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(video, 0, 0);
        const imageBase64 = canvas.toDataURL("image/jpeg", 0.9);
        setCapturedImage(imageBase64);
    }, []);

    // Submit registration
    const submitRegistration = useCallback(async () => {
        if (!capturedImage) return;

        setRegisterState("processing");

        try {
            const result = await registerUser(name, email, capturedImage);

            if (result.success && result.user) {
                setRegisteredUser(result.user);
                setRegisterState("success");
                stopCamera();
            } else {
                setErrorMessage(result.error || "Registration failed");
                setRegisterState("error");
            }
        } catch (err) {
            console.error("Registration error:", err);
            setErrorMessage("Connection error. Is the backend running?");
            setRegisterState("error");
        }
    }, [capturedImage, name, email, stopCamera]);

    // Retake photo
    const retakePhoto = () => {
        setCapturedImage(null);
    };

    // Reset form
    const resetForm = () => {
        setRegisterState("form");
        setName("");
        setEmail("");
        setCapturedImage(null);
        setRegisteredUser(null);
        setErrorMessage("");
        stopCamera();
    };

    // Cleanup camera on unmount
    useEffect(() => {
        return () => stopCamera();
    }, [stopCamera]);

    return (
        <main className="min-h-screen relative">
            {/* Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <motion.div
                    className="absolute top-1/3 left-1/4 w-96 h-96 bg-emerald-600/20 rounded-full blur-3xl"
                    animate={{ scale: [1, 1.2, 1], opacity: [0.2, 0.4, 0.2] }}
                    transition={{ duration: 5, repeat: Infinity }}
                />
            </div>

            <div className="relative z-10 max-w-xl mx-auto px-6 py-8">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <Link
                        href="/"
                        className="inline-flex items-center gap-2 text-white/60 hover:text-white transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span>Back</span>
                    </Link>
                    <h1 className="text-2xl font-bold text-white">Register</h1>
                    <div className="w-20" />
                </div>

                <AnimatePresence mode="wait">
                    {/* Form State */}
                    {registerState === "form" && (
                        <motion.div
                            key="form"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: 20 }}
                        >
                            <div className="glass-card p-8">
                                <div className="flex items-center justify-center mb-8">
                                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center">
                                        <UserPlus className="w-8 h-8 text-white" />
                                    </div>
                                </div>

                                <h2 className="text-2xl font-bold text-center text-white mb-2">
                                    Join FaceAccess
                                </h2>
                                <p className="text-center text-white/50 mb-8">
                                    Register your face for instant access
                                </p>

                                {/* Form Fields */}
                                <div className="space-y-4 mb-6">
                                    <div>
                                        <label className="block text-sm font-medium text-white/70 mb-2">
                                            Full Name
                                        </label>
                                        <div className="relative">
                                            <User className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40" />
                                            <input
                                                type="text"
                                                value={name}
                                                onChange={(e) => setName(e.target.value)}
                                                placeholder="Enter your name"
                                                className="input-premium pl-12"
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <label className="block text-sm font-medium text-white/70 mb-2">
                                            Email Address
                                        </label>
                                        <div className="relative">
                                            <Mail className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40" />
                                            <input
                                                type="email"
                                                value={email}
                                                onChange={(e) => setEmail(e.target.value)}
                                                placeholder="Enter your email"
                                                className="input-premium pl-12"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {errorMessage && (
                                    <motion.p
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        className="text-red-400 text-sm text-center mb-4"
                                    >
                                        {errorMessage}
                                    </motion.p>
                                )}

                                <motion.button
                                    className="btn-premium w-full flex items-center justify-center gap-3 bg-gradient-to-r from-emerald-500 to-teal-500"
                                    onClick={proceedToCapture}
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                >
                                    <Camera className="w-5 h-5" />
                                    Continue to Photo
                                </motion.button>

                                {/* Welcome Bonus */}
                                <div className="mt-6 flex items-center justify-center gap-2 text-sm text-white/50">
                                    <Star className="w-4 h-4 text-yellow-400" />
                                    <span>Get 100 bonus points on registration!</span>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {/* Capture State */}
                    {registerState === "capture" && (
                        <motion.div
                            key="capture"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                        >
                            <div className="glass-card p-6">
                                <h2 className="text-xl font-bold text-center text-white mb-6">
                                    Take Your Photo
                                </h2>

                                {/* Camera/Photo View */}
                                <div className="camera-frame relative bg-black/50 rounded-2xl overflow-hidden mb-6">
                                    {!capturedImage ? (
                                        <>
                                            <video
                                                ref={videoRef}
                                                autoPlay
                                                playsInline
                                                muted
                                                className="w-full h-full object-cover"
                                            />
                                            {/* Face Guide */}
                                            <div className="absolute inset-0 flex items-center justify-center">
                                                <motion.div
                                                    className="w-48 h-60 border-2 border-dashed border-emerald-400/50 rounded-[60px]"
                                                    animate={{ scale: [1, 1.02, 1] }}
                                                    transition={{ duration: 2, repeat: Infinity }}
                                                />
                                            </div>
                                        </>
                                    ) : (
                                        <img
                                            src={capturedImage}
                                            alt="Captured"
                                            className="w-full h-full object-cover"
                                        />
                                    )}
                                </div>

                                <canvas ref={canvasRef} className="hidden" />

                                {/* Action Buttons */}
                                {!capturedImage ? (
                                    <motion.button
                                        className="btn-premium w-full flex items-center justify-center gap-3 bg-gradient-to-r from-emerald-500 to-teal-500"
                                        onClick={capturePhoto}
                                        disabled={!isCameraReady}
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                    >
                                        <Camera className="w-5 h-5" />
                                        Capture Photo
                                    </motion.button>
                                ) : (
                                    <div className="grid grid-cols-2 gap-4">
                                        <button
                                            onClick={retakePhoto}
                                            className="btn-premium bg-white/10 hover:bg-white/20 flex items-center justify-center gap-2"
                                        >
                                            <RefreshCw className="w-4 h-4" />
                                            Retake
                                        </button>
                                        <button
                                            onClick={submitRegistration}
                                            className="btn-premium bg-gradient-to-r from-emerald-500 to-teal-500 flex items-center justify-center gap-2"
                                        >
                                            <CheckCircle2 className="w-4 h-4" />
                                            Confirm
                                        </button>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {/* Processing State */}
                    {registerState === "processing" && (
                        <motion.div
                            key="processing"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                        >
                            <div className="glass-card p-8 text-center glow-effect">
                                <motion.div
                                    className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-purple-500 to-indigo-500 flex items-center justify-center"
                                    animate={{ rotate: 360 }}
                                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                >
                                    <RefreshCw className="w-10 h-10 text-white" />
                                </motion.div>
                                <h2 className="text-2xl font-bold text-white mb-2">
                                    Processing...
                                </h2>
                                <p className="text-white/50">Encoding your face data</p>
                            </div>
                        </motion.div>
                    )}

                    {/* Success State */}
                    {registerState === "success" && registeredUser && (
                        <motion.div
                            key="success"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                        >
                            <div className="glass-card success-card p-8">
                                {/* Success Icon */}
                                <motion.div
                                    className="flex justify-center mb-6"
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    transition={{ type: "spring", delay: 0.2 }}
                                >
                                    <div className="w-24 h-24 rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center">
                                        <CheckCircle2 className="w-14 h-14 text-white" />
                                    </div>
                                </motion.div>

                                <motion.h2
                                    className="text-3xl font-bold text-center text-emerald-400 mb-2"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.3 }}
                                >
                                    Welcome!
                                </motion.h2>

                                <motion.p
                                    className="text-center text-white/60 mb-8"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.4 }}
                                >
                                    Registration successful, {registeredUser.name}!
                                </motion.p>

                                {/* Welcome Bonus */}
                                <motion.div
                                    className="flex items-center justify-center gap-3 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 rounded-xl p-4 mb-6"
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: 0.5 }}
                                >
                                    <Sparkles className="w-6 h-6 text-yellow-400" />
                                    <span className="text-lg font-semibold text-yellow-400">
                                        +100 Welcome Bonus Points!
                                    </span>
                                </motion.div>

                                {/* User Card */}
                                <motion.div
                                    className="bg-white/5 rounded-2xl p-6 mb-6"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.6 }}
                                >
                                    <div className="flex items-center gap-4">
                                        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-2xl font-bold text-white">
                                            {registeredUser.name.charAt(0).toUpperCase()}
                                        </div>
                                        <div>
                                            <h3 className="text-lg font-semibold text-white">
                                                {registeredUser.name}
                                            </h3>
                                            <p className="text-white/50">{registeredUser.email}</p>
                                            <div className="flex items-center gap-2 mt-1">
                                                <span className="px-3 py-1 rounded-full tier-bronze text-white text-sm font-medium">
                                                    {registeredUser.tier}
                                                </span>
                                                <span className="text-yellow-400 font-medium">
                                                    {registeredUser.loyalty_points} pts
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>

                                {/* Action Buttons */}
                                <div className="grid grid-cols-2 gap-4">
                                    <Link
                                        href="/scan"
                                        className="btn-premium bg-gradient-to-r from-purple-500 to-indigo-500 text-center"
                                    >
                                        Try Face Scan
                                    </Link>
                                    <Link
                                        href="/dashboard"
                                        className="btn-premium bg-white/10 hover:bg-white/20 text-center"
                                    >
                                        Dashboard
                                    </Link>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {/* Error State */}
                    {registerState === "error" && (
                        <motion.div
                            key="error"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                        >
                            <div className="glass-card denied-card p-8 text-center">
                                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-red-400 to-pink-500 flex items-center justify-center">
                                    <UserPlus className="w-10 h-10 text-white" />
                                </div>
                                <h2 className="text-2xl font-bold text-red-400 mb-4">
                                    Registration Failed
                                </h2>
                                <p className="text-white/60 mb-6">{errorMessage}</p>
                                <button
                                    onClick={resetForm}
                                    className="btn-premium bg-white/10 hover:bg-white/20"
                                >
                                    Try Again
                                </button>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </main>
    );
}
