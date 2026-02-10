"use client";

import { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import {
    LayoutDashboard,
    Users,
    Activity,
    Star,
    Trophy,
    Clock,
    ArrowLeft,
    RefreshCw,
    TrendingUp,
    Shield,
    Zap,
} from "lucide-react";
import Link from "next/link";
import { getStats, getUsers, getAccessLogs, User, AccessLog } from "@/lib/api";

export default function DashboardPage() {
    const [stats, setStats] = useState<{
        total_users: number;
        total_access_events: number;
        total_visits: number;
        total_loyalty_points: number;
        tier_distribution: { Bronze: number; Silver: number; Gold: number; Platinum: number };
        recent_activity: AccessLog[];
    } | null>(null);
    const [users, setUsers] = useState<User[]>([]);
    const [logs, setLogs] = useState<AccessLog[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<"overview" | "users" | "logs">("overview");

    const fetchData = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const [statsData, usersData, logsData] = await Promise.all([
                getStats(),
                getUsers(),
                getAccessLogs(50),
            ]);
            setStats(statsData);
            setUsers(usersData.users);
            setLogs(logsData.logs);
        } catch (err) {
            console.error("Failed to fetch data:", err);
            setError("Failed to connect to backend. Is the server running?");
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData();
        // Auto-refresh every 10 seconds
        const interval = setInterval(fetchData, 10000);
        return () => clearInterval(interval);
    }, [fetchData]);

    const formatTime = (timestamp: string) => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
        });
    };

    const formatDate = (timestamp: string) => {
        const date = new Date(timestamp);
        return date.toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
        });
    };

    const tierColors: Record<string, string> = {
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
                    className="absolute top-0 right-0 w-96 h-96 bg-orange-600/10 rounded-full blur-3xl"
                    animate={{ scale: [1, 1.1, 1], opacity: [0.1, 0.2, 0.1] }}
                    transition={{ duration: 6, repeat: Infinity }}
                />
                <motion.div
                    className="absolute bottom-0 left-0 w-96 h-96 bg-pink-600/10 rounded-full blur-3xl"
                    animate={{ scale: [1.1, 1, 1.1], opacity: [0.2, 0.1, 0.2] }}
                    transition={{ duration: 6, repeat: Infinity }}
                />
            </div>

            <div className="relative z-10 max-w-6xl mx-auto px-6 py-8">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <Link
                        href="/"
                        className="inline-flex items-center gap-2 text-white/60 hover:text-white transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span>Back</span>
                    </Link>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <LayoutDashboard className="w-7 h-7 text-orange-400" />
                        Admin Dashboard
                    </h1>
                    <button
                        onClick={fetchData}
                        className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                        disabled={isLoading}
                    >
                        <RefreshCw className={`w-5 h-5 text-white ${isLoading ? "animate-spin" : ""}`} />
                    </button>
                </div>

                {/* Error State */}
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass-card denied-card p-4 mb-6 text-center"
                    >
                        <p className="text-red-400">{error}</p>
                        <p className="text-white/50 text-sm mt-2">
                            Run: <code className="bg-white/10 px-2 py-1 rounded">cd backend && python main.py</code>
                        </p>
                    </motion.div>
                )}

                {/* Loading State */}
                {isLoading && !stats && (
                    <div className="glass-card p-12 text-center">
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                            className="inline-block"
                        >
                            <RefreshCw className="w-10 h-10 text-purple-400" />
                        </motion.div>
                        <p className="text-white/60 mt-4">Loading dashboard...</p>
                    </div>
                )}

                {/* Dashboard Content */}
                {stats && (
                    <>
                        {/* Tab Navigation */}
                        <div className="flex gap-2 mb-6">
                            {[
                                { id: "overview", label: "Overview", icon: TrendingUp },
                                { id: "users", label: "Users", icon: Users },
                                { id: "logs", label: "Access Logs", icon: Activity },
                            ].map((tab) => (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id as typeof activeTab)}
                                    className={`nav-link flex items-center gap-2 ${activeTab === tab.id ? "active" : ""
                                        }`}
                                >
                                    <tab.icon className="w-4 h-4" />
                                    {tab.label}
                                </button>
                            ))}
                        </div>

                        {/* Overview Tab */}
                        {activeTab === "overview" && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ duration: 0.3 }}
                            >
                                {/* Stats Grid */}
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                                    {[
                                        {
                                            label: "Total Users",
                                            value: stats.total_users,
                                            icon: Users,
                                            color: "from-purple-500 to-indigo-500",
                                        },
                                        {
                                            label: "Access Events",
                                            value: stats.total_access_events,
                                            icon: Shield,
                                            color: "from-emerald-500 to-teal-500",
                                        },
                                        {
                                            label: "Total Visits",
                                            value: stats.total_visits,
                                            icon: Activity,
                                            color: "from-orange-500 to-pink-500",
                                        },
                                        {
                                            label: "Loyalty Points",
                                            value: stats.total_loyalty_points.toLocaleString(),
                                            icon: Star,
                                            color: "from-yellow-500 to-orange-500",
                                        },
                                    ].map((stat, index) => (
                                        <motion.div
                                            key={stat.label}
                                            className="stat-card"
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: index * 0.1 }}
                                        >
                                            <div
                                                className={`w-12 h-12 rounded-xl bg-gradient-to-br ${stat.color} flex items-center justify-center mb-4`}
                                            >
                                                <stat.icon className="w-6 h-6 text-white" />
                                            </div>
                                            <div className="text-3xl font-bold text-white mb-1">
                                                {stat.value}
                                            </div>
                                            <div className="text-sm text-white/50">{stat.label}</div>
                                        </motion.div>
                                    ))}
                                </div>

                                {/* Tier Distribution & Recent Activity */}
                                <div className="grid md:grid-cols-2 gap-6">
                                    {/* Tier Distribution */}
                                    <motion.div
                                        className="glass-card p-6"
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: 0.4 }}
                                    >
                                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                            <Trophy className="w-5 h-5 text-yellow-400" />
                                            Tier Distribution
                                        </h3>
                                        <div className="space-y-3">
                                            {Object.entries(stats.tier_distribution).map(([tier, count]) => (
                                                <div key={tier} className="flex items-center gap-3">
                                                    <span
                                                        className={`px-3 py-1 rounded-full text-white text-sm font-medium ${tierColors[tier]}`}
                                                    >
                                                        {tier}
                                                    </span>
                                                    <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
                                                        <motion.div
                                                            className={`h-full ${tierColors[tier]}`}
                                                            initial={{ width: 0 }}
                                                            animate={{
                                                                width: `${stats.total_users > 0
                                                                        ? (count / stats.total_users) * 100
                                                                        : 0
                                                                    }%`,
                                                            }}
                                                            transition={{ delay: 0.5, duration: 0.5 }}
                                                        />
                                                    </div>
                                                    <span className="text-white/60 text-sm w-8">{count}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </motion.div>

                                    {/* Recent Activity */}
                                    <motion.div
                                        className="glass-card p-6"
                                        initial={{ opacity: 0, x: 20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: 0.4 }}
                                    >
                                        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                            <Zap className="w-5 h-5 text-emerald-400" />
                                            Recent Activity
                                        </h3>
                                        <div className="space-y-3 max-h-64 overflow-y-auto">
                                            {stats.recent_activity.length === 0 ? (
                                                <p className="text-white/40 text-center py-8">
                                                    No activity yet
                                                </p>
                                            ) : (
                                                stats.recent_activity.slice(0, 5).map((log) => (
                                                    <div
                                                        key={log.id}
                                                        className="flex items-center gap-3 p-3 bg-white/5 rounded-xl"
                                                    >
                                                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white font-bold">
                                                            {log.user_name.charAt(0)}
                                                        </div>
                                                        <div className="flex-1">
                                                            <p className="text-white font-medium">{log.user_name}</p>
                                                            <p className="text-white/50 text-sm">{log.location}</p>
                                                        </div>
                                                        <div className="text-right">
                                                            <p className="text-white/60 text-sm">{formatTime(log.timestamp)}</p>
                                                            <p className="text-white/40 text-xs">{formatDate(log.timestamp)}</p>
                                                        </div>
                                                    </div>
                                                ))
                                            )}
                                        </div>
                                    </motion.div>
                                </div>
                            </motion.div>
                        )}

                        {/* Users Tab */}
                        {activeTab === "users" && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ duration: 0.3 }}
                                className="glass-card p-6"
                            >
                                <h3 className="text-lg font-semibold text-white mb-4">
                                    Registered Users ({users.length})
                                </h3>
                                {users.length === 0 ? (
                                    <p className="text-white/40 text-center py-12">
                                        No users registered yet
                                    </p>
                                ) : (
                                    <table className="table-premium">
                                        <thead>
                                            <tr>
                                                <th>User</th>
                                                <th>Email</th>
                                                <th>Tier</th>
                                                <th>Points</th>
                                                <th>Visits</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {users.map((user) => (
                                                <tr key={user.id}>
                                                    <td>
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white font-bold">
                                                                {user.name.charAt(0)}
                                                            </div>
                                                            <span className="text-white font-medium">{user.name}</span>
                                                        </div>
                                                    </td>
                                                    <td className="text-white/60">{user.email}</td>
                                                    <td>
                                                        <span
                                                            className={`px-3 py-1 rounded-full text-white text-sm font-medium ${tierColors[user.tier]}`}
                                                        >
                                                            {user.tier}
                                                        </span>
                                                    </td>
                                                    <td className="text-yellow-400 font-medium">
                                                        {user.loyalty_points}
                                                    </td>
                                                    <td className="text-white/60">{user.total_visits}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                )}
                            </motion.div>
                        )}

                        {/* Logs Tab */}
                        {activeTab === "logs" && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ duration: 0.3 }}
                                className="glass-card p-6"
                            >
                                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                    <Clock className="w-5 h-5" />
                                    Access Logs ({logs.length})
                                </h3>
                                {logs.length === 0 ? (
                                    <p className="text-white/40 text-center py-12">
                                        No access logs yet
                                    </p>
                                ) : (
                                    <div className="space-y-3 max-h-96 overflow-y-auto">
                                        {logs.map((log) => (
                                            <div
                                                key={log.id}
                                                className="flex items-center gap-4 p-4 bg-white/5 rounded-xl"
                                            >
                                                <div
                                                    className={`w-3 h-3 rounded-full ${log.status === "granted" ? "bg-emerald-400" : "bg-red-400"
                                                        }`}
                                                />
                                                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white font-bold">
                                                    {log.user_name.charAt(0)}
                                                </div>
                                                <div className="flex-1">
                                                    <p className="text-white font-medium">{log.user_name}</p>
                                                    <p className="text-white/50 text-sm">
                                                        {log.access_type} at {log.location}
                                                    </p>
                                                </div>
                                                <div className="text-right">
                                                    <p className="text-white/60">{formatTime(log.timestamp)}</p>
                                                    <p className="text-white/40 text-sm">{formatDate(log.timestamp)}</p>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </>
                )}
            </div>
        </main>
    );
}
