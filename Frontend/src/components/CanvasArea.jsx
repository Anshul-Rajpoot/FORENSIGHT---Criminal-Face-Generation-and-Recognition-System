import styles from "./CanvasArea.module.css";

export default function CanvasArea({
  canvasRef,
  selectedElement,
  gender,
  onGenderChange,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  onDelete,
  onBringForward,
  onSendBackward,
  onDownload,
  onGenerate,
  onSearch,
  onUploadToDatabase,
  generatedImageUrl,
  matches,
  searchLoading,
  searchAttempted,
  onRotate,
  onMoveX,
  onMoveY,
  canvasW,
  canvasH,
}) {
  const relPos = selectedElement
    ? selectedElement.getRelativePosition(canvasW, canvasH)
    : { x: 0, y: 0 };

  return (
    <main className={styles.area}>
      <canvas
        ref={canvasRef}
        width={canvasW}
        height={canvasH}
        className={styles.canvas}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      />

     <div className={styles.controls}>

  {/* Row 1 → Gender */}
 <div className={`${styles.row}`} >

    <button className={styles.btn}  style={{ height: "50px" }}>
      <label className={styles.genderLabel} >Gender :{"   "} </label>
      <select
        className={styles.genderSelect  }
        value={gender}
        onChange={(e) => onGenderChange?.(e.target.value)}
        
      >
        <option value="Any">Any</option>
        <option value="Male">Male</option>
        <option value="Female">Female</option>
        <option value="Other">Other</option>
        
      </select>
    </button>

    <button className={styles.btn} type="button" onClick={onDownload}  style={{ height: "50px" }}>
      Download
    </button>
  </div>


  {/* Row 2 → Delete + Layer controls */}
  <div className={styles.row}>
    <button className={styles.btn} onClick={onDelete}>
      Delete
    </button>

    

     <button className={styles.btn} onClick={onGenerate}>
      Generate
    </button>

    </div>
    
  

  {/* Row 3 → Generate */}
  <div className={styles.row}>
    <button className={styles.btn} onClick={onBringForward}>
      Layer Up
    </button>
    <button className={styles.btn} onClick={onSendBackward}>
      Layer Down
    </button>
   
  </div>

  {/* Row 4 → Download */}

  

  </div>



      {generatedImageUrl && (
        <div className={styles.generated}>
          <p className={styles.generatedTitle}>Generated Image</p>
          <img
            className={styles.generatedImg}
            src={generatedImageUrl}
            alt="Generated"
          />

          <div className={styles.generatedActions}>
            <button
              className={styles.actionBtn}
              type="button"
              onClick={onSearch}
              disabled={searchLoading}
            >
              {searchLoading ? "Searching..." : "Search"}
            </button>
          </div>

          {searchAttempted && Array.isArray(matches) && matches.length === 0 && (
            <button
              className={styles.uploadBtn}
              type="button"
              onClick={onUploadToDatabase}
            >
              Upload
            </button>
          )}
        </div>
      )}

      {/* <div className={styles.transform}>
        <label className={styles.label}> Rotate </label>
        <input
          type="range"
          min="-3.14"
          max="3.14"
          step="0.01"
          value={selectedElement ? selectedElement.rotation : 0}
          className={styles.slider}
          onChange={(e) => onRotate(parseFloat(e.target.value))}
        />

        <label className={styles.label}> X </label>
        <input
          type="range"
          min="-200"
          max="200"
          value={Math.round(relPos.x)}
          className={styles.slider}
          onChange={(e) => onMoveX(parseFloat(e.target.value))}
        />

        <label className={styles.label}> Y </label>
        <input
          type="range"
          min="-200"
          max="200"
          value={Math.round(relPos.y)}
          className={styles.slider}
          onChange={(e) => onMoveY(parseFloat(e.target.value))}
        />
      </div> */}

      {Array.isArray(matches) && matches.length > 0 && (
        <div className={styles.results}>
          <p className={styles.resultsTitle}>Matches</p>
          <div className={styles.resultsGrid}>
            {matches.slice(0, 6).map((m, i) => (
              <div
                key={`${m.name}-${i}`}
                className={
                  i === 0 ? styles.resultCardBest : styles.resultCard
                }
              >
                {m.imageURL && (
                  <img
                    className={styles.resultImg}
                    src={m.imageURL}
                    alt={m.name || "match"}
                  />
                )}

                <p className={styles.resultScore}>
                  Match: {typeof m.score === "number"
                    ? `${(m.score * 100).toFixed(1)}%`
                    : "-"}
                </p>
                <p className={styles.resultName}>{m.name}</p>
                {m.age != null && <p className={styles.resultMeta}>Age: {m.age}</p>}
                {m.sex && <p className={styles.resultMeta}>Sex: {m.sex}</p>}
                {m.crime && <p className={styles.resultMeta}>Crime: {m.crime}</p>}
                {m.status && (
                  <p className={styles.resultMeta}>Status: {m.status}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}