import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Signup() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    role: "NORMAL",
    adminSecret: "",
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
      const payload =
        form.role === "ADMIN" ? form : { ...form, adminSecret: undefined };

      const res = await fetch("http://localhost:8000/api/auth/signup", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (res.ok) {
        navigate("/login");
      } else {
        setError(data.message || "Signup failed");
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
          Create account to access Criminal Face Generation and recognition
        </p>
      </div>

      {/* RIGHT SIDE */}
      <div style={styles.right}>
        <div style={styles.card}>
          <h2 style={styles.title}>Create Account </h2>
          <p style={styles.subtitle}>Join the system</p>

          <form onSubmit={handleSubmit} style={styles.form}>
            <input
              name="name"
              placeholder="Full Name"
              required
              onChange={handleChange}
              style={styles.input}
            />

            <input
              name="email"
              type="email"
              placeholder="Email"
              required
              onChange={handleChange}
              style={styles.input}
            />

            <input
              name="password"
              type="password"
              placeholder="Password"
              required
              onChange={handleChange}
              style={styles.input}
            />

            <select
              name="role"
              onChange={handleChange}
              style={styles.input}
            >
              <option value="NORMAL">Normal User</option>
              <option value="ADMIN">Admin</option>
            </select>

            {form.role === "ADMIN" && (
              <input
                name="adminSecret"
                placeholder="Admin Secret Key"
                required
                onChange={handleChange}
                style={styles.input}
              />
            )}

            <button type="submit" style={styles.button} disabled={loading}>
              {loading ? "Creating..." : "Signup"}
            </button>
          </form>

          {error && <p style={styles.error}>{error}</p>}

          <p style={styles.footer}>
            Already have an account?{" "}
            <span onClick={() => navigate("/login")} style={styles.link}>
              Login
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
  
    // background-image: "linear-gradient(135deg, rgba(30,41,59,0.85), rgba(30,41,59,0.6))",
    backdropFilter: "blur(12px)",
    marginTop: "-80px",
    marginLeft: "50px",
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