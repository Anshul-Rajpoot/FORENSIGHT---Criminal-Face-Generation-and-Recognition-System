import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import styles from "./Members.module.css";

export default function Members() {
  const API_BASE = useMemo(() => "http://localhost:8000", []);
  const [params] = useSearchParams();

  const name = (params.get("name") || "").trim();
  const sex = (params.get("sex") || "").trim();

  const [loading, setLoading] = useState(false);
  const [members, setMembers] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    if (!name) {
      setMembers([]);
      setError("Type a name in the header search.");
      return;
    }

    const run = async () => {
      try {
        setLoading(true);
        setError("");

        const url = new URL(`${API_BASE}/api/members`);
        url.searchParams.set("name", name);
        if (sex) url.searchParams.set("sex", sex);

        const res = await fetch(url.toString(), {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        let data = {};
        try {
          data = await res.json();
        } catch {
          data = {};
        }
        if (!res.ok) {
          setMembers([]);
          setError(data.message || `Search failed (HTTP ${res.status})`);
          return;
        }

        setMembers(Array.isArray(data.members) ? data.members : []);
      } catch {
        setMembers([]);
        setError("Server error");
      } finally {
        setLoading(false);
      }
    };

    run();
  }, [API_BASE, name, sex]);

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        <div className={styles.headerRow}>
          <h2 className={styles.title}>Members</h2>
          <div className={styles.subtitle}>
            Showing results for: <span className={styles.pill}>{name || "-"}</span>
            {sex ? <span className={styles.pill}>{sex}</span> : null}
          </div>
        </div>

        {loading && <p className={styles.muted}>Searching...</p>}
        {!loading && error && <p className={styles.muted}>{error}</p>}

        {!loading && !error && members.length === 0 && (
          <p className={styles.muted}>No members found.</p>
        )}

        {members.length > 0 && (
          <div className={styles.resultsScroll}>
            <div className={styles.grid}>
              {members.map((m, i) => (
                <div key={`${m.name}-${i}`} className={styles.cardItem}>
                  {m.imageURL && (
                    <img
                      className={styles.img}
                      src={m.imageURL}
                      alt={m.name || "member"}
                    />
                  )}
                  <div className={styles.name}>{m.name}</div>
                  {m.age != null && <div className={styles.meta}>Age: {m.age}</div>}
                  {m.sex && <div className={styles.meta}>Sex: {m.sex}</div>}
                  {m.crime && <div className={styles.meta}>Crime: {m.crime}</div>}
                  {m.status && <div className={styles.meta}>Status: {m.status}</div>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
