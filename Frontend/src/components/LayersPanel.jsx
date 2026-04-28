import { ASSET_LABELS } from "../utils/canvasUtils.js";
import styles from "./LayersPanel.module.css";

export default function LayersPanel({ elements, selectedId, onSelect }) {
  return (
    <aside className={styles.panel}>
      <h3 className={styles.heading}>Canvas Layers</h3>

      {elements.length === 0 && (
        <p className={styles.empty}>
          No layers yet.
          <br />
          Click an asset to add.
        </p>
      )}

      {[...elements].reverse().map((el) => (
        <div
          key={el.id}
          className={`${styles.layerItem} ${el.id === selectedId ? styles.active : ""}`}
          onClick={() => onSelect(el.id)}
        >
          <img src={el.img.src} className={styles.preview} alt={el.type} />
          <span className={styles.layerLabel}>
            {ASSET_LABELS[el.type] ?? el.type}
          </span>
        </div>
      ))}
    </aside>
  );
}
