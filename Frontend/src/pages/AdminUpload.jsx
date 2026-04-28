import { useState, useEffect } from "react";
import styles from "./AdminUpload.module.css";

export default function AdminUpload() {
  const [form, setForm] = useState({
    name: "",
    age: "",
    sex: "",
    address: "",
    height: "",
    weight: "",
    crime: "",
    status: "ARRESTED",
  });

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (key, value) => {
    setForm({ ...form, [key]: value });
  };

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleUpload = async () => {
    if (!file) return alert("Upload image");

    const formData = new FormData();
    Object.entries(form).forEach(([k, v]) => formData.append(k, v));
    formData.append("image", file);

    const token = localStorage.getItem("token");

    try {
      setLoading(true);

      const res = await fetch("http://localhost:8000/api/enroll", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) return alert(data.message);

      alert("✅ Criminal Added");

      setForm({
        name: "",
        age: "",
        sex: "",
        address: "",
        height: "",
        weight: "",
        crime: "",
        status: "ARRESTED",
      });

      setFile(null);
      setPreview(null);
    } catch {
      alert("Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <div className={styles.leftPanel}>
          {preview ? (
            <img src={preview} alt="preview" className={styles.previewImgLarge} />
          ) : (
            <div className={styles.previewPlaceholder}>
              📸 Upload Image to Preview
            </div>
          )}
        </div>

        <div className={styles.rightPanel}>
          <h2 className={styles.title}>🚨 Add Criminal</h2>

          <div className={styles.grid}>
            <input className={styles.input} placeholder="Name" value={form.name} onChange={(e) => handleChange("name", e.target.value)} />
            <input className={styles.input} type="number" placeholder="Age" value={form.age} onChange={(e) => handleChange("age", e.target.value)} />

            <select className={styles.select} value={form.sex} onChange={(e) => handleChange("sex", e.target.value)}>
              <option value="">Gender</option>
              <option>Male</option>
              <option>Female</option>
              <option>Other</option>

            </select>

            <input className={styles.input} placeholder="Address" value={form.address} onChange={(e) => handleChange("address", e.target.value)} />

            <input className={styles.input} type="number" placeholder="Height (cm)" value={form.height} onChange={(e) => handleChange("height", e.target.value)} />
            <input className={styles.input} type="number" placeholder="Weight (kg)" value={form.weight} onChange={(e) => handleChange("weight", e.target.value)} />

            <input className={styles.input} placeholder="Crime" value={form.crime} onChange={(e) => handleChange("crime", e.target.value)} />
          </div>

          <div className={styles.status}>
            <button
              className={form.status === "ARRESTED" ? styles.activeBtn : styles.btn}
              onClick={() => handleChange("status", "ARRESTED")}
            >
              Arrested
            </button>

            <button
              className={form.status === "NOT ARRESTED" ? styles.activeBtn : styles.btn}
              onClick={() => handleChange("status", "NOT ARRESTED")}
            >
              Not Arrested
            </button>
          </div>

          <input
            className={styles.fileInput}
            type="file"
            accept=".jpg,.png"
            onChange={(e) => {
              const f = e.target.files[0];
              if (!f) return;
              document.documentElement.requestFullscreen().catch(console.error);
              setFile(f);
              setPreview(URL.createObjectURL(f));
            }}
          />

          <button className={styles.submit} onClick={handleUpload}>
            {loading ? "Adding..." : "➕ Add Criminal"}
          </button>
        </div>
      </div>
    </div>
  );
}
