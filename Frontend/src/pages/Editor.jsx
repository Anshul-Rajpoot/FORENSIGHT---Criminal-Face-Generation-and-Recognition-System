import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import AssetPanel from "../components/AssetPanel.jsx";
import CanvasArea from "../components/CanvasArea.jsx";
import LayersPanel from "../components/LayersPanel.jsx";
import Toast from "../components/Toast.jsx";
import { useCanvas } from "../hooks/useCanvas.js";
import { useToast } from "../hooks/useToast.js";
import styles from "./Editor.module.css";

export default function Editor() {
  const { toasts, showToast } = useToast();
  const navigate = useNavigate();

  const API_BASE = useMemo(() => "http://localhost:8000", []);
  const token = localStorage.getItem("token");

  const [generatedImageUrl, setGeneratedImageUrl] = useState(null);
  const [gender, setGender] = useState(() => {
    // Force an explicit choice at least once per session
    return sessionStorage.getItem("generatedGender") ?? "";
  });
  const [matches, setMatches] = useState([]);
  const [matchLoading, setMatchLoading] = useState(false);
  const [searchAttempted, setSearchAttempted] = useState(false);

  const {
    canvasRef,
    elements,
    selectedElement,
    selectedId,
    addElement,
    deleteSelected,
    bringForward,
    sendBackward,
    setRotation,
    setPositionX,
    setPositionY,
    selectById,
    downloadImage,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    CANVAS_W,
    CANVAS_H,
  } = useCanvas(showToast);

  // Welcome toast on mount
  useEffect(() => {
    showToast("Welcome to FORENSIGHT! 🎨", "info");
  }, []); // eslint-disable-line

  const dataUrlToBlob = (dataUrl) => {
    const [meta, b64] = dataUrl.split(",");
    const mime = meta.match(/data:(.*);base64/)?.[1] || "image/png";
    const bin = atob(b64);
    const buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    return new Blob([buf], { type: mime });
  };

  const uploadImageForMatch = async (file) => {
    if (!file) return;
    if (!token) {
      showToast("Please login first", "warning");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    try {
      setMatchLoading(true);
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        showToast(data.error || data.message || "Upload failed", "warning");
        return;
      }

      const sorted = (data.matches || []).sort((a, b) => b.score - a.score);
      setMatches(sorted);
      setSearchAttempted(true);
      showToast(
        sorted.length ? `Found ${sorted.length} match(es)` : "No matches",
        sorted.length ? "success" : "info",
      );
    } catch (e) {
      showToast("Server error", "warning");
    } finally {
      setMatchLoading(false);
    }
  };

  const generatePreview = () => {
    if (!canvasRef.current) return;
    if (elements.length === 0) {
      showToast("Add some elements first!", "warning");
      return;
    }
    if (!gender) {
      showToast("Please select a gender.", "warning");
      return;
    }
    const url = canvasRef.current.toDataURL("image/png");
    setGeneratedImageUrl(url);
    setSearchAttempted(false);
    setMatches([]);

    try {
      sessionStorage.setItem("generatedImageUrl", url);
      sessionStorage.setItem("generatedGender", gender);
    } catch {
      // ignore storage failures
    }

    navigate("/results", { state: { generatedImageUrl: url, gender } });
  };

  const onGenderChange = (nextGender) => {
    setGender(nextGender);
    try {
      sessionStorage.setItem("generatedGender", nextGender);
    } catch {
      // ignore storage failures
    }
  };

  const uploadGeneratedForMatch = async () => {
    if (!generatedImageUrl) {
      showToast("Generate the image first", "warning");
      return;
    }
    const blob = dataUrlToBlob(generatedImageUrl);
    const file = new File([blob], `generated-${Date.now()}.png`, {
      type: "image/png",
    });
    await uploadImageForMatch(file);
  };

  const goToUploadForm = () => {
    navigate("/upload");
  };

  return (
    <div className={styles.appWrapper}>
      <div className={styles.layout}>
        <AssetPanel onSelect={addElement} />

        <CanvasArea
          canvasRef={canvasRef}
          selectedElement={selectedElement}
          gender={gender}
          onGenderChange={onGenderChange}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onDelete={deleteSelected}
          onBringForward={bringForward}
          onSendBackward={sendBackward}
          onDownload={downloadImage}
          onGenerate={generatePreview}
          onSearch={uploadGeneratedForMatch}
          onUploadToDatabase={goToUploadForm}
          generatedImageUrl={generatedImageUrl}
          matches={matches}
          searchLoading={matchLoading}
          searchAttempted={searchAttempted}
          onRotate={setRotation}
          onMoveX={setPositionX}
          onMoveY={setPositionY}
          canvasW={CANVAS_W}
          canvasH={CANVAS_H}
        />

        <LayersPanel
          elements={elements}
          selectedId={selectedId}
          onSelect={selectById}
        />
      </div>

      <Toast toasts={toasts} />
    </div>
  );
}
