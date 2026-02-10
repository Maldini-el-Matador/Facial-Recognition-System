// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface User {
    id: string;
    name: string;
    email: string;
    loyalty_points: number;
    tier: "Bronze" | "Silver" | "Gold" | "Platinum";
    total_visits: number;
    created_at?: string;
    photo_url?: string;
}

export interface AccessLog {
    id: string;
    user_id: string;
    user_name: string;
    access_type: string;
    location: string;
    timestamp: string;
    status: string;
}

export interface ApiResponse<T> {
    success: boolean;
    error?: string;
    code?: string;
    [key: string]: T | boolean | string | undefined;
}

// Helper function for API calls
async function fetchAPI<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        ...options,
        headers: {
            "Content-Type": "application/json",
            ...options.headers,
        },
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(error.detail || "API request failed");
    }

    return response.json();
}

// Face Detection
export async function detectFace(imageBase64: string) {
    return fetchAPI<{ success: boolean; detected: boolean; location?: object }>("/api/detect", {
        method: "POST",
        body: JSON.stringify({ image: imageBase64 }),
    });
}

// User Registration
export async function registerUser(name: string, email: string, imageBase64: string) {
    return fetchAPI<{
        success: boolean;
        user?: User;
        message?: string;
        error?: string;
    }>("/api/register", {
        method: "POST",
        body: JSON.stringify({ name, email, image: imageBase64 }),
    });
}

// Face Verification
export async function verifyFace(imageBase64: string) {
    return fetchAPI<{
        success: boolean;
        matched: boolean;
        user?: User;
        confidence?: number;
        error?: string;
    }>("/api/verify", {
        method: "POST",
        body: JSON.stringify({ image: imageBase64 }),
    });
}

// Access Request (verify + log + award points)
export async function requestAccess(
    imageBase64: string,
    location: string = "Main Entrance",
    awardPoints: number = 10
) {
    return fetchAPI<{
        success: boolean;
        access_granted: boolean;
        user?: User;
        points_awarded?: number;
        confidence?: number;
        message?: string;
        error?: string;
    }>("/api/access", {
        method: "POST",
        body: JSON.stringify({
            image: imageBase64,
            location,
            award_points: awardPoints,
        }),
    });
}

// Get All Users
export async function getUsers() {
    return fetchAPI<{ users: User[]; count: number }>("/api/users");
}

// Get User by ID
export async function getUser(userId: string) {
    return fetchAPI<User>(`/api/users/${userId}`);
}

// Get Access Logs
export async function getAccessLogs(limit: number = 50) {
    return fetchAPI<{ logs: AccessLog[]; count: number }>(`/api/logs?limit=${limit}`);
}

// Get Stats
export async function getStats() {
    return fetchAPI<{
        total_users: number;
        total_access_events: number;
        total_visits: number;
        total_loyalty_points: number;
        tier_distribution: { Bronze: number; Silver: number; Gold: number; Platinum: number };
        recent_activity: AccessLog[];
    }>("/api/stats");
}

// Redeem Loyalty Points
export async function redeemPoints(userId: string, points: number, description: string) {
    return fetchAPI<{
        success: boolean;
        remaining_points: number;
        message: string;
    }>("/api/loyalty/redeem", {
        method: "POST",
        body: JSON.stringify({ user_id: userId, points, description }),
    });
}

// Health Check
export async function healthCheck() {
    return fetchAPI<{ status: string; users_count: number }>("/health");
}
