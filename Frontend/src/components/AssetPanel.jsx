import { useState } from "react";
import { ASSET_CATALOGUE } from "../utils/canvasUtils.js";
import styles from "./AssetPanel.module.css";

export default function AssetPanel({ onSelect }) {
  const [failedImages, setFailedImages] = useState(new Set());

  const handleError = (src) => {
    setFailedImages((prev) => new Set([...prev, src]));
  };

  return (
    <aside className={styles.panel}>
      <h3 className={styles.heading}>Assets</h3>

      {ASSET_CATALOGUE.map(({ folder, count, label }) => {
        const assets = Array.from({ length: count }, (_, i) => {
          const num = String(i + 1).padStart(2, "0");
          return `/assets/${folder}/${num}.png`;
        }).filter((src) => !failedImages.has(src));

        if (assets.length === 0) return null;

        return (
          <section key={folder} className={styles.section}>
            <h4 className={styles.subheading}>{label}</h4>
            <div className={styles.grid}>
              {assets.map((src) => (
                <img
                  key={src}
                  src={src}
                  alt={label}
                  className={styles.assetThumb}
                  loading="lazy"
                  onClick={() => onSelect(src)}
                  onError={() => handleError(src)}
                />
              ))}
            </div>
          </section>
        );
      })}
    </aside>
  );
}
