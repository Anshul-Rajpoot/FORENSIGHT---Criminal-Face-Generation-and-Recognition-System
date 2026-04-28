import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import styles from "./AppHeader.module.css";

export default function AppHeader() {
  const navigate = useNavigate();
  const [token, setToken] = useState(() => localStorage.getItem("token"));
  const [role, setRole] = useState(() => localStorage.getItem("role"));
  const uploadMatchRef = useRef(null);

  const [searchName, setSearchName] = useState("");
  const [searchSex, setSearchSex] = useState("Any");

  useEffect(() => {
    const sync = () => {
      setToken(localStorage.getItem("token"));
      setRole(localStorage.getItem("role"));
    };
    window.addEventListener("authchange", sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("authchange", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("role");
    window.dispatchEvent(new Event("authchange"));
    navigate("/login");
  };

  const goGenerate = () => {
    navigate(token ? "/app" : "/login");
  };

  const runMemberSearch = (e) => {
    e?.preventDefault?.();

    if (!token) {
      alert("Please login first");
      return;
    }

    const name = searchName.trim();
    if (!name) {
      return;
    }

    const params = new URLSearchParams();
    params.set("name", name);
    if (searchSex && searchSex !== "Any") params.set("sex", searchSex);
    navigate(`/members?${params.toString()}`);
  };

  const startUploadMatch = () => {
    if (!token) {
      navigate("/login");
      return;
    }
    uploadMatchRef.current?.click?.();
  };

  const onUploadMatchSelected = (e) => {
    const file = e.target.files?.[0];
    // Allow selecting the same file twice
    e.target.value = "";

    if (!file) return;
    if (!token) {
      navigate("/login");
      return;
    }

    document.documentElement.requestFullscreen().catch(console.error);

    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      if (!dataUrl.startsWith("data:")) {
        alert("Invalid image file");
        return;
      }
      sessionStorage.setItem("generatedImageUrl", dataUrl);
      sessionStorage.setItem("generatedGender", "Any");
      navigate("/results", { state: { generatedImageUrl: dataUrl, gender: "Any" } });
    };
    reader.onerror = () => alert("Failed to read file");
    reader.readAsDataURL(file);
  };

  const handleDoubleClick = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(console.error);
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  };

  return (
    <header className={styles.header} onDoubleClick={handleDoubleClick}>
      <div className={styles.inner}>
        <Link
          to="/"
          className={`${styles.left} ${styles.brandLink}`}
          aria-label="Go to home"
        >
          <img
            src="/assets/logo.png"
            className={styles.logo}
            alt="FORENSIGHT Logo"
          />
             <h1 className={styles.title}>FORENSIGHT</h1>
        </Link>

        {token ? (
          <form className={styles.search} onSubmit={runMemberSearch}>
            <input
              className={styles.searchInput}
              placeholder="Search member by name"
              value={searchName}
              onChange={(e) => setSearchName(e.target.value)}
            />
            <select
              className={styles.searchSelect}
              value={searchSex}
              onChange={(e) => setSearchSex(e.target.value)}
              aria-label="Sex"
            >
              <option value="Any">Any</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
            <button
              className={styles.searchBtn}
              type="submit"
              disabled={!searchName.trim()}
            >
              Search
            </button>
          </form>
        ) : (
          <div />
        )}

        <div className={styles.actions}>
          <input
            ref={uploadMatchRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={onUploadMatchSelected}
          />

          <button
            type="button"
            className={styles.actionBtn}
            onClick={goGenerate}
          >
            Generate Image
          </button>
          {token ? (
            <>
              <button
                type="button"
                className={styles.actionBtn}
                onClick={startUploadMatch}
              >
                Search Image
              </button>

              {role === "ADMIN" && (
                <button
                  type="button"
                  className={styles.actionBtn}
                  onClick={() => navigate("/upload")}
                >
                  Upload Image
                </button>
              )}

              <button type="button" className={styles.actionBtn} onClick={logout}>
                Logout
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className={styles.actionBtn}
                onClick={() => navigate("/login")}
              >
                Login
              </button>
              <button
                type="button"
                className={styles.actionBtn}
                onClick={() => navigate("/signup")}
              >
                Signup
              </button>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
