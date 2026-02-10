"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Scan,
  UserPlus,
  LayoutDashboard,
  Shield,
  Star,
  ChevronRight
} from "lucide-react";
import Link from "next/link";

export default function Home() {
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

  const features = [
    {
      icon: Scan,
      title: "Face Scan Access",
      description: "Instant building access with real-time facial recognition",
      href: "/scan",
      gradient: "from-purple-500 to-indigo-600",
    },
    {
      icon: UserPlus,
      title: "Register Face",
      description: "Quick enrollment with instant loyalty bonus",
      href: "/register",
      gradient: "from-emerald-500 to-teal-600",
    },
    {
      icon: LayoutDashboard,
      title: "Admin Dashboard",
      description: "Monitor access logs and manage users",
      href: "/dashboard",
      gradient: "from-orange-500 to-pink-600",
    },
  ];

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute -top-40 -right-40 w-80 h-80 bg-purple-600/30 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3]
          }}
          transition={{ duration: 5, repeat: Infinity }}
        />
        <motion.div
          className="absolute -bottom-40 -left-40 w-80 h-80 bg-indigo-600/30 rounded-full blur-3xl"
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.5, 0.3, 0.5]
          }}
          transition={{ duration: 5, repeat: Infinity }}
        />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-12">
        {/* Header */}
        <motion.header
          className="text-center mb-16"
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.div
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-6"
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.3 }}
          >
            <Shield className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-purple-300">Secure Biometric Access</span>
          </motion.div>

          <h1 className="text-5xl md:text-7xl font-bold mb-6">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-purple-400 via-pink-400 to-indigo-400">
              FaceAccess
            </span>
            <span className="text-white/90"> Pro</span>
          </h1>

          <p className="text-xl text-white/60 max-w-2xl mx-auto leading-relaxed">
            Cutting-edge facial recognition for seamless building access
            and loyalty rewards. Just look and go.
          </p>
        </motion.header>

        {/* Feature Cards */}
        <motion.div
          className="grid md:grid-cols-3 gap-6 mb-16"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          {features.map((feature, index) => (
            <Link href={feature.href} key={index}>
              <motion.div
                className="glass-card p-8 cursor-pointer relative overflow-hidden group"
                onMouseEnter={() => setHoveredCard(index)}
                onMouseLeave={() => setHoveredCard(null)}
                whileHover={{ scale: 1.02, y: -5 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                {/* Gradient Overlay */}
                <motion.div
                  className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-300`}
                />

                {/* Icon */}
                <motion.div
                  className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-6`}
                  animate={hoveredCard === index ? { rotate: [0, -5, 5, 0] } : {}}
                  transition={{ duration: 0.5 }}
                >
                  <feature.icon className="w-7 h-7 text-white" />
                </motion.div>

                {/* Content */}
                <h3 className="text-xl font-semibold mb-3 text-white group-hover:text-purple-300 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-white/50 mb-6 leading-relaxed">
                  {feature.description}
                </p>

                {/* Arrow */}
                <div className="flex items-center text-purple-400 font-medium">
                  <span className="mr-2">Get Started</span>
                  <motion.div
                    animate={hoveredCard === index ? { x: [0, 5, 0] } : {}}
                    transition={{ duration: 0.6, repeat: hoveredCard === index ? Infinity : 0 }}
                  >
                    <ChevronRight className="w-5 h-5" />
                  </motion.div>
                </div>
              </motion.div>
            </Link>
          ))}
        </motion.div>

        {/* Stats */}
        <motion.div
          className="glass-card p-8"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { value: "<0.5s", label: "Recognition Time" },
              { value: "99.9%", label: "Accuracy Rate" },
              { value: "256-bit", label: "Encryption" },
              { value: "24/7", label: "Availability" },
            ].map((stat, index) => (
              <motion.div
                key={index}
                className="text-center"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.8 + index * 0.1 }}
              >
                <div className="text-3xl md:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400 mb-2">
                  {stat.value}
                </div>
                <div className="text-sm text-white/50">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Loyalty Section */}
        <motion.div
          className="mt-12 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
        >
          <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full glass-card">
            <Star className="w-5 h-5 text-yellow-400" />
            <span className="text-white/80">
              Earn <span className="text-yellow-400 font-semibold">10 loyalty points</span> with every entry
            </span>
          </div>
        </motion.div>
      </div>
    </main>
  );
}
