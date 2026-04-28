import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    email: "",
    password: "",
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await fetch("http://localhost:8000/api/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(form),
      });

      const data = await res.json();

      if (res.ok) {
        localStorage.setItem("token", data.token);
        localStorage.setItem("role", data.user.role);
        window.dispatchEvent(new Event("authchange"));

        document.documentElement.requestFullscreen().catch(console.error);

        navigate("/");
      } else {
        setError(data.message || "Login failed");
      }
    } catch {
      setError("Server error");
    } finally {
      setLoading(false);
    }
  };

  return (
  <div style={styles.container}>

    {/* LEFT SIDE */}
    <div style={styles.left}>
      <img src="/assets/logo.png" style={styles.logo} alt="logo" />
      <h1 style={styles.brandTitle}>FORENSIGHT</h1>
      <p style={styles.tagline}>
        Login to access Criminal Face Generation and Recognition
      </p>
    </div>

    {/* RIGHT SIDE */}
    <div style={styles.right}>
      <div style={styles.card}>
        <h2 style={styles.title}>Welcome Back </h2>
        <p style={styles.subtitle}>Login to continue</p>

        <form onSubmit={handleSubmit} style={styles.form}>
          <input
            type="email"
            name="email"
            placeholder="Email"
            required
            onChange={handleChange}
            style={styles.input}
          />

          <input
            type="password"
            name="password"
            placeholder="Password"
            required
            onChange={handleChange}
            style={styles.input}
          />

          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        {error && <p style={styles.error}>{error}</p>}

        <p style={styles.footer}>
          Don’t have an account?{" "}
          <span onClick={() => navigate("/signup")} style={styles.link}>
            Signup
          </span>
        </p>
      </div>
    </div>

  </div>
);
}
const styles = {
  container: {
    height: "100vh",
    display: "flex",
    background: "linear-gradient(to right, #020617, #0f172a)",
  },

  /* LEFT SIDE */
  left: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "flex-start",
    paddingLeft: "80px",
    color: "#fff",
    background: "radial-gradient(circle at left, #1e293b, #020617)",
  },

  logo: {
    width: "220px",
    marginBottom: "20px",
    filter: "drop-shadow(0 0 20px rgba(99,102,241,0.4))",
  },

  brandTitle: {
    fontSize: "32px",
    fontWeight: "700",
    fontFamily: "'Lucida Sans', 'Lucida Sans Regular', 'Lucida Grande', 'Lucida Sans Unicode', Geneva, Verdana, sans-serif",
  },

  tagline: {
    marginTop: "10px",
    color: "#94a3b8",
    maxWidth: "300px",
  },

  /* RIGHT SIDE */
  right: {
    flex: 1,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },

  card: {
    background: "rgba(30,41,59,0.85)",
    backdropFilter: "blur(12px)",
    marginTop: "-75px",
    marginLeft: "40px",
    padding: "40px",
    borderRadius: "20px",
    width: "360px",
    boxShadow: "0 20px 50px rgba(0,0,0,0.6)",
    textAlign: "center",
  },

  title: {
    color: "#fff",
    marginBottom: "5px",
  },

  subtitle: {
    color: "#94a3b8",
    marginBottom: "25px",
  },

  form: {
    display: "flex",
    flexDirection: "column",
    gap: "14px",
  },

  input: {
    padding: "14px",
    borderRadius: "10px",
    border: "none",
    background: "#334155",
    color: "#fff",
    outline: "none",
  },

  button: {
    padding: "14px",
    borderRadius: "10px",
    border: "none",
    background: "linear-gradient(135deg,#6366f1,#3b82f6)",
    color: "#fff",
    fontWeight: "bold",
    cursor: "pointer",
  },

  error: {
    marginTop: "10px",
    color: "#ef4444",
    fontSize: "14px",
    textAlign: "center",
  },

  footer: {
    marginTop: "20px",
    color: "#94a3b8",
  },

  link: {
    color: "#818cf8",
    cursor: "pointer",
    fontWeight: "bold",
  },
};