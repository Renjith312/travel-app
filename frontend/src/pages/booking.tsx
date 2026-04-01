import { useState } from "react";
import { useNavigate } from "react-router-dom";

const bookingServices = [
  {
    id: "ksrtc",
    name: "KSRTC",
    tagline: "Kerala State Road Transport",
    description: "Book state bus tickets across Kerala and neighbouring states with ease.",
    url: "https://www.keralartc.com/",
    icon: "🚌",
    category: "Bus",
    color: "#ff6b6b",
    glow: "rgba(255,107,107,0.35)",
    badge: "State Bus",
  },
  {
    id: "redbus",
    name: "RedBus",
    tagline: "India's Largest Bus Network",
    description: "Book inter-city bus tickets from 2000+ operators across India.",
    url: "https://www.redbus.in/",
    icon: "🚍",
    category: "Bus",
    color: "#ff4757",
    glow: "rgba(255,71,87,0.35)",
    badge: "Popular",
  },
  {
    id: "irctc",
    name: "IRCTC",
    tagline: "Indian Railways Official",
    description: "Official portal for train ticket booking across all Indian Railways.",
    url: "https://www.irctc.co.in/",
    icon: "🚂",
    category: "Train",
    color: "#48dbfb",
    glow: "rgba(72,219,251,0.3)",
    badge: "Official",
  },
  {
    id: "booking",
    name: "Booking.com",
    tagline: "Hotels & Stays Worldwide",
    description: "Find and book hotels, resorts, homestays and apartments worldwide.",
    url: "https://www.booking.com/",
    icon: "🏨",
    category: "Hotel",
    color: "#a29bfe",
    glow: "rgba(162,155,254,0.35)",
    badge: "Best Rates",
  },
];

export default function BookingPage() {
  const navigate = useNavigate();
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  return (
    <div style={styles.page}>

      {/* Ambient background orbs */}
      <div style={{ ...styles.orb, top: "-120px", left: "-100px", background: "radial-gradient(circle, rgba(72,219,251,0.12) 0%, transparent 70%)", width: "500px", height: "500px" }} />
      <div style={{ ...styles.orb, bottom: "-80px", right: "-80px", background: "radial-gradient(circle, rgba(162,155,254,0.1) 0%, transparent 70%)", width: "400px", height: "400px" }} />
      <div style={{ ...styles.orb, top: "40%", left: "50%", transform: "translate(-50%,-50%)", background: "radial-gradient(circle, rgba(255,107,107,0.06) 0%, transparent 70%)", width: "600px", height: "600px" }} />

      {/* Back button */}
      <div style={styles.backRow}>
        <button onClick={() => navigate('/')} style={styles.backBtn}>
          ← Home
        </button>
      </div>

      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerInner}>
          <div style={styles.pill}>
            <span style={styles.pillDot} />
            TRAVEL BOOKINGS
          </div>
          <h1 style={styles.title}>
            Book Your
            <span style={styles.titleAccent}> Journey</span>
          </h1>
          <p style={styles.subtitle}>
            Direct links to official booking portals —{" "}
            <span style={{ color: "#ff6b6b" }}>buses</span>,{" "}
            <span style={{ color: "#48dbfb" }}>trains</span> &amp;{" "}
            <span style={{ color: "#a29bfe" }}>hotels</span>
          </p>
        </div>
      </div>

      {/* Cards Grid — all services, no filter */}
      <div style={styles.grid}>
        {bookingServices.map((service) => {
          const isHovered = hoveredId === service.id;
          return (
            <a
              key={service.id}
              href={service.url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                ...styles.card,
                ...(isHovered
                  ? {
                      ...styles.cardHovered,
                      borderColor: service.color + "55",
                      boxShadow: `0 0 0 1px ${service.color}33, 0 24px 48px rgba(0,0,0,0.5), 0 0 60px ${service.glow}`,
                    }
                  : {}),
              }}
              onMouseEnter={() => setHoveredId(service.id)}
              onMouseLeave={() => setHoveredId(null)}
            >
              {/* Top shimmer line */}
              <div
                style={{
                  ...styles.shimmerLine,
                  background: `linear-gradient(90deg, transparent, ${service.color}, transparent)`,
                  opacity: isHovered ? 1 : 0,
                }}
              />

              {/* Card header row */}
              <div style={styles.cardHeader}>
                <div
                  style={{
                    ...styles.iconBubble,
                    background: isHovered
                      ? `radial-gradient(circle at 30% 30%, ${service.color}33, ${service.color}11)`
                      : "rgba(255,255,255,0.04)",
                    borderColor: isHovered ? service.color + "44" : "rgba(255,255,255,0.08)",
                    boxShadow: isHovered ? `0 0 20px ${service.glow}` : "none",
                  }}
                >
                  <span style={styles.icon}>{service.icon}</span>
                </div>

                <div
                  style={{
                    ...styles.badge,
                    color: service.color,
                    background: service.color + "18",
                    border: `1px solid ${service.color}30`,
                  }}
                >
                  {service.badge}
                </div>
              </div>

              {/* Category label */}
              <div style={styles.categoryLabel}>
                <span
                  style={{
                    ...styles.categoryBar,
                    background: service.color,
                    boxShadow: `0 0 6px ${service.color}`,
                  }}
                />
                <span style={{
                  color: service.color + "cc",
                  fontSize: "11px",
                  fontWeight: 600,
                  letterSpacing: "1.5px",
                  textTransform: "uppercase" as const,
                }}>
                  {service.category}
                </span>
              </div>

              {/* Name & description */}
              <div style={styles.cardBody}>
                <h3
                  style={{
                    ...styles.cardName,
                    color: isHovered ? service.color : "#f1f3f5",
                    textShadow: isHovered ? `0 0 20px ${service.glow}` : "none",
                  }}
                >
                  {service.name}
                </h3>
                <p style={styles.cardTagline}>{service.tagline}</p>
                <p style={styles.cardDesc}>{service.description}</p>
              </div>

              {/* CTA */}
              <div
                style={{
                  ...styles.cta,
                  background: isHovered
                    ? `linear-gradient(135deg, ${service.color}22, ${service.color}11)`
                    : "rgba(255,255,255,0.04)",
                  borderColor: isHovered ? service.color + "55" : "rgba(255,255,255,0.08)",
                  color: isHovered ? service.color : "#868e96",
                }}
              >
                <span style={{ fontWeight: 700, letterSpacing: "0.5px" }}>Book Now</span>
                <span
                  style={{
                    fontSize: "18px",
                    display: "inline-block",
                    transition: "transform 0.2s ease",
                    transform: isHovered ? "translateX(4px)" : "none",
                  }}
                >
                  →
                </span>
              </div>

              {/* Corner glow */}
              {isHovered && (
                <div
                  style={{
                    position: "absolute",
                    bottom: "-30px",
                    right: "-30px",
                    width: "120px",
                    height: "120px",
                    borderRadius: "50%",
                    background: `radial-gradient(circle, ${service.glow}, transparent 70%)`,
                    pointerEvents: "none",
                  }}
                />
              )}
            </a>
          );
        })}
      </div>

      {/* Stats row */}
      <div style={styles.statsRow}>
        {[
          { label: "Bus Operators",    value: "2000+",   color: "#ff6b6b" },
          { label: "Train Routes",     value: "10,000+", color: "#48dbfb" },
          { label: "Hotels Worldwide", value: "500K+",   color: "#a29bfe" },
        ].map((stat) => (
          <div key={stat.label} style={styles.statCard}>
            <span style={{ ...styles.statValue, color: stat.color }}>{stat.value}</span>
            <span style={styles.statLabel}>{stat.label}</span>
          </div>
        ))}
      </div>

      <p style={styles.disclaimer}>
        All links redirect to official third-party booking portals. Travel Copilot is not affiliated with these services.
      </p>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "#0d1117",
    fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
    padding: "0 0 80px",
    position: "relative",
    overflowX: "hidden",
    color: "#f1f3f5",
  },
  orb: {
    position: "fixed",
    borderRadius: "50%",
    pointerEvents: "none",
    zIndex: 0,
  },
  header: {
    position: "relative",
    zIndex: 1,
    padding: "64px 24px 52px",
    textAlign: "center",
    borderBottom: "1px solid rgba(255,255,255,0.05)",
  },
  headerInner: {
    maxWidth: "640px",
    margin: "0 auto",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "16px",
  },
  pill: {
    display: "inline-flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "11px",
    fontWeight: 700,
    letterSpacing: "3px",
    color: "#74b9ff",
    background: "rgba(116,185,255,0.1)",
    border: "1px solid rgba(116,185,255,0.2)",
    borderRadius: "20px",
    padding: "5px 16px",
  },
  pillDot: {
    width: "6px",
    height: "6px",
    borderRadius: "50%",
    background: "#74b9ff",
    boxShadow: "0 0 8px #74b9ff",
    display: "inline-block",
  },
  title: {
    fontSize: "clamp(32px, 6vw, 52px)",
    fontWeight: 900,
    margin: 0,
    letterSpacing: "-1.5px",
    lineHeight: 1.1,
    color: "#ffffff",
  },
  titleAccent: {
    background: "linear-gradient(135deg, #74b9ff, #a29bfe)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
  },
  subtitle: {
    fontSize: "15px",
    color: "rgba(255,255,255,0.4)",
    margin: 0,
    lineHeight: 1.7,
  },
  grid: {
    position: "relative",
    zIndex: 1,
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
    gap: "18px",
    maxWidth: "1100px",
    margin: "48px auto 0",
    padding: "0 24px",
  },
  card: {
    position: "relative",
    background: "rgba(255,255,255,0.03)",
    backdropFilter: "blur(16px)",
    borderRadius: "18px",
    padding: "26px 22px 20px",
    display: "flex",
    flexDirection: "column",
    gap: "14px",
    textDecoration: "none",
    color: "inherit",
    border: "1px solid rgba(255,255,255,0.08)",
    boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
    transition: "all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)",
    overflow: "hidden",
    cursor: "pointer",
  },
  cardHovered: {
    transform: "translateY(-8px) scale(1.015)",
    background: "rgba(255,255,255,0.055)",
  },
  shimmerLine: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    height: "1.5px",
    transition: "opacity 0.3s ease",
  },
  cardHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  iconBubble: {
    width: "56px",
    height: "56px",
    borderRadius: "14px",
    border: "1px solid",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 0.3s ease",
  },
  icon: {
    fontSize: "26px",
    lineHeight: 1,
  },
  badge: {
    fontSize: "10px",
    fontWeight: 700,
    letterSpacing: "1px",
    textTransform: "uppercase",
    padding: "4px 12px",
    borderRadius: "20px",
  },
  categoryLabel: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  categoryBar: {
    width: "3px",
    height: "12px",
    borderRadius: "2px",
    flexShrink: 0,
  },
  cardBody: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
    flex: 1,
  },
  cardName: {
    fontSize: "24px",
    fontWeight: 800,
    margin: 0,
    letterSpacing: "-0.5px",
    transition: "color 0.25s ease, text-shadow 0.25s ease",
  },
  cardTagline: {
    fontSize: "12px",
    color: "rgba(255,255,255,0.3)",
    fontWeight: 500,
    margin: 0,
    letterSpacing: "0.2px",
  },
  cardDesc: {
    fontSize: "13px",
    color: "rgba(255,255,255,0.45)",
    lineHeight: 1.6,
    margin: "6px 0 0",
  },
  cta: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "11px 16px",
    borderRadius: "10px",
    fontSize: "13px",
    border: "1px solid",
    transition: "all 0.25s ease",
    marginTop: "auto",
  },
  statsRow: {
    position: "relative",
    zIndex: 1,
    display: "flex",
    justifyContent: "center",
    gap: "16px",
    flexWrap: "wrap",
    maxWidth: "700px",
    margin: "48px auto 0",
    padding: "0 24px",
  },
  statCard: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "4px",
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: "14px",
    padding: "18px 32px",
    backdropFilter: "blur(10px)",
    flex: "1 1 160px",
  },
  statValue: {
    fontSize: "26px",
    fontWeight: 800,
    letterSpacing: "-0.5px",
  },
  statLabel: {
    fontSize: "12px",
    color: "rgba(255,255,255,0.35)",
    fontWeight: 500,
    letterSpacing: "0.3px",
  },
  backRow: {
    position: "relative",
    zIndex: 1,
    padding: "20px 24px 0",
  },
  backBtn: {
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
    padding: "8px 18px",
    borderRadius: "10px",
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(255,255,255,0.05)",
    color: "rgba(255,255,255,0.6)",
    fontSize: "13px",
    fontWeight: 600,
    cursor: "pointer",
    transition: "all 0.2s ease",
  },
  disclaimer: {
    position: "relative",
    zIndex: 1,
    textAlign: "center",
    fontSize: "12px",
    color: "rgba(255,255,255,0.2)",
    marginTop: "40px",
    padding: "0 24px",
    lineHeight: 1.6,
  },
};