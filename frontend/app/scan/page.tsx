"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Scan,
    CheckCircle2,
    XCircle,
    ArrowLeft,
    Camera,
    RefreshCw,
    Star,
    Trophy,
    Zap,
} from "lucide-react";
import Link from "next/link";
import { requestAccess, User } from "@/lib/api";

type ScanState = "idle" | "scanning" | "success" | "denied" | "error";

export default function ScanPage() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const [scanState, setScanState] = useState<ScanState>("idle");
    const [matchedUser, setMatchedUser] = useState<User | null>(null);
    const [pointsAwarded, setPointsAwarded] = useState<number>(0);
    const [confidence, setConfidence] = useState<number>(0);
    const [errorMessage, setErrorMessage] = useState<string>("");
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
            setErrorMessage("Could not access camera. Please allow camera permissions.");
            setScanState("error");
        }
    }, []);

    // Stop camera
    const stopCamera = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
        }
    }, []);

    // Capture frame and send to API
    const captureAndScan = useCallback(async () => {
        if (!videoRef.current || !canvasRef.current) return;

        setScanState("scanning");

        const canvas = canvasRef.current;
        const video = videoRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(video, 0, 0);
        const imageBase64 = canvas.toDataURL("image/jpeg", 0.8);

        try {
            const result = await requestAccess(imageBase64, "Main Entrance", 10);

            if (result.success && result.access_granted && result.user) {
                setMatchedUser(result.user);
                setPointsAwarded(result.points_awarded || 10);
                setConfidence(result.confidence || 95);
                setScanState("success");
            } else {
                setErrorMessage(result.error || "Face not recognized");
                setScanState("denied");
            }
        } catch (err) {
            console.error("Scan error:", err);
            setErrorMessage("Connection error. Is the backend running?");
            setScanState("error");
        }
    }, []);

    // Reset scan
    const resetScan = useCallback(() => {
        setScanState("idle");
        setMatchedUser(null);
        setPointsAwarded(0);
        setConfidence(0);
        setErrorMessage("");
    }, []);

    // Initialize camera on mount
    useEffect(() => {
        startCamera();
        return () => stopCamera();
    }, [startCamera, stopCamera]);

    // Tier colors
    const tierColor = {
        Bronze: "tier-bronze",
        Silver: "tier-silver",
        Gold: "tier-gold",
        Platinum: "tier-platinum",
    };

    return (
        <main className="min-h-screen relative">
            {/* Background */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <motion.div
                    className="absolute top-1/4 right-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl"
                    animate={{ scale: [1, 1.1, 1], opacity: [0.2, 0.3, 0.2] }}
                    transition={{ duration: 4, repeat: Infinity }}
                />
            </div>

            <div className="relative z-10 max-w-2xl mx-auto px-6 py-8">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <Link
                        href="/"
                        className="inline-flex items-center gap-2 text-white/60 hover:text-white transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span>Back</span>
                    </Link>
                    <h1 className="text-2xl font-bold text-white">Face Scan</h1>
                    <div className="w-20" />
                </div>

                {/* Main Content */}
                <AnimatePresence mode="wait">
                    {/* Camera View */}
                    {(scanState === "idle" || scanState === "scanning") && (
                        <motion.div
                            key="camera"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                        >
                            <div
                                className={`glass-card p-6 ${scanState === "scanning" ? "glow-effect" : ""
                                    }`}
                            >
                                {/* Camera Frame */}
                                <div className="camera-frame relative bg-black/50 rounded-2xl overflow-hidden mb-6">
                                    <video
                                        ref={videoRef}
                                        autoPlay
                                        playsInline
                                        muted
                                        className="w-full h-full object-cover"
                                    />

                                    {/* Scanning Animation */}
                                    {scanState === "scanning" && (
                                        <motion.div
                                            className="absolute inset-0 bg-gradient-to-b from-purple-500/20 via-transparent to-purple-500/20"
                                            animate={{ opacity: [0.5, 1, 0.5] }}
                                            transition={{ duration: 1, repeat: Infinity }}
                                        >
                                            <div className="scan-line" />
                                        </motion.div>
                                    )}

                                    {/* Face Guide */}
                                    {scanState === "idle" && (
                                        <div className="absolute inset-0 flex items-center justify-center">
                                            <motion.div
                                                className="w-48 h-60 border-2 border-dashed border-purple-400/50 rounded-[60px]"
                                                animate={{ scale: [1, 1.02, 1] }}
                                                transition={{ duration: 2, repeat: Infinity }}
                                            />
                                        </div>
                                    )}
                                </div>

                                {/* Hidden Canvas for capture */}
                                <canvas ref={canvasRef} className="hidden" />

                                {/* Status Text */}
                                <div className="text-center mb-6">
                                    {scanState === "idle" && (
                                        <p className="text-white/60">
                                            Position your face within the frame and press scan
                                        </p>
                                    )}
                                    {scanState === "scanning" && (
                                        <motion.p
                                            className="text-purple-300 font-medium"
                                            animate={{ opacity: [1, 0.5, 1] }}
                                            transition={{ duration: 1, repeat: Infinity }}
                                        >
                                            Scanning face...
                                        </motion.p>
                                    )}
                                </div>

                                {/* Scan Button */}
                                <motion.button
                                    className="btn-premium w-full flex items-center justify-center gap-3 text-lg"
                                    onClick={captureAndScan}
                                    disabled={!isCameraReady || scanState === "scanning"}
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                >
                                    {scanState === "scanning" ? (
                                        <>
                                            <motion.div
                                                animate={{ rotate: 360 }}
                                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                            >
                                                <RefreshCw className="w-6 h-6" />
                                            </motion.div>
                                            Processing...
                                        </>
                                    ) : (
                                        <>
                                            <Scan className="w-6 h-6" />
                                            Scan Face
                                        </>
                                    )}
                                </motion.button>
                            </div>
                        </motion.div>
                    )}

                    {/* Success State */}
                    {scanState === "success" && matchedUser && (
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

                                {/* Access Granted */}
                                <motion.h2
                                    className="text-3xl font-bold text-center text-emerald-400 mb-2"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.3 }}
                                >
                                    Access Granted
                                </motion.h2>

                                <motion.p
                                    className="text-center text-white/60 mb-8"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ delay: 0.4 }}
                                >
                                    Welcome back, {matchedUser.name}!
                                </motion.p>

                                {/* User Info Card */}
                                <motion.div
                                    className="bg-white/5 rounded-2xl p-6 mb-6"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.5 }}
                                >
                                    <div className="flex items-center justify-between mb-4">
                                        <div>
                                            <h3 className="text-xl font-semibold text-white">
                                                {matchedUser.name}
                                            </h3>
                                            <p className="text-white/50">{matchedUser.email}</p>
                                        </div>
                                        <div
                                            className={`px-4 py-2 rounded-full text-white font-medium ${tierColor[matchedUser.tier]
                                                }`}
                                        >
                                            {matchedUser.tier}
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="bg-white/5 rounded-xl p-4 text-center">
                                            <Star className="w-6 h-6 text-yellow-400 mx-auto mb-2" />
                                            <div className="text-2xl font-bold text-white">
                                                {matchedUser.loyalty_points}
                                            </div>
                                            <div className="text-sm text-white/50">Points</div>
                                        </div>
                                        <div className="bg-white/5 rounded-xl p-4 text-center">
                                            <Trophy className="w-6 h-6 text-purple-400 mx-auto mb-2" />
                                            <div className="text-2xl font-bold text-white">
                                                {matchedUser.total_visits}
                                            </div>
                                            <div className="text-sm text-white/50">Total Visits</div>
                                        </div>
                                    </div>
                                </motion.div>

                                {/* Points Earned */}
                                <motion.div
                                    className="flex items-center justify-center gap-3 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 rounded-xl p-4 mb-6"
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: 0.6 }}
                                >
                                    <Zap className="w-6 h-6 text-yellow-400" />
                                    <span className="text-lg font-semibold text-yellow-400">
                                        +{pointsAwarded} Points Earned!
                                    </span>
                                </motion.div>

                                {/* Confidence */}
                                <div className="text-center text-white/40 text-sm mb-6">
                                    Match confidence: {confidence}%
                                </div>

                                {/* Action Buttons */}
                                <div className="grid grid-cols-2 gap-4">
                                    <button
                                        onClick={resetScan}
                                        className="btn-premium bg-white/10 hover:bg-white/20"
                                    >
                                        Scan Again
                                    </button>
                                    <Link href="/dashboard" className="btn-premium text-center">
                                        Dashboard
                                    </Link>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {/* Denied State */}
                    {(scanState === "denied" || scanState === "error") && (
                        <motion.div
                            key="denied"
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                        >
                            <div className="glass-card denied-card p-8">
                                {/* Denied Icon */}
                                <motion.div
                                    className="flex justify-center mb-6"
                                    initial={{ scale: 0 }}
                                    animate={{ scale: 1 }}
                                    transition={{ type: "spring", delay: 0.2 }}
                                >
                                    <div className="w-24 h-24 rounded-full bg-gradient-to-br from-red-400 to-pink-500 flex items-center justify-center">
                                        <XCircle className="w-14 h-14 text-white" />
                                    </div>
                                </motion.div>

                                <h2 className="text-3xl font-bold text-center text-red-400 mb-4">
                                    {scanState === "error" ? "Error" : "Access Denied"}
                                </h2>

                                <p className="text-center text-white/60 mb-8">{errorMessage}</p>

                                <div className="grid grid-cols-2 gap-4">
                                    <button
                                        onClick={resetScan}
                                        className="btn-premium bg-white/10 hover:bg-white/20"
                                    >
                                        Try Again
                                    </button>
                                    <Link href="/register" className="btn-premium text-center">
                                        Register
                                    </Link>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </main>
    );
}
