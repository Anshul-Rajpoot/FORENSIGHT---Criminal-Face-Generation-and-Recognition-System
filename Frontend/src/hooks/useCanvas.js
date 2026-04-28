import { useRef, useState, useCallback, useEffect } from "react";
import {
  CanvasElement,
  getSmartPosition,
  LAYER_PRIORITY,
} from "../utils/canvasUtils.js";

const CANVAS_W = 350;
const CANVAS_H = 440;

export function useCanvas(showToast) {
  const canvasRef = useRef(null);

  // All canvas elements (ordered = z-order)
  const [elements, setElements] = useState([]);
  const [selectedId, setSelectedId] = useState(null);

  // Interaction state stored in refs to avoid stale closures in event handlers
  const stateRef = useRef({
    isDragging: false,
    isResizing: false,
    resizeHandle: null,
    offsetX: 0,
    offsetY: 0,
    startX: 0,
    startY: 0,
    startWidth: 0,
    startHeight: 0,
    startElementX: 0,
    startElementY: 0,
    originalLayerIdx: null,
  });

  // Live element list ref so mouse handlers always see current state
  const elementsRef = useRef(elements);
  const selectedIdRef = useRef(selectedId);
  useEffect(() => {
    elementsRef.current = elements;
  }, [elements]);
  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  // ─── Redraw ──────────────────────────────────────────────────────────────
  const redraw = useCallback((elems, selId) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    elems.forEach((el) => el.draw(ctx, el.id === selId));
  }, []);

  // Redraw whenever state changes
  useEffect(() => {
    redraw(elements, selectedId);
  }, [elements, selectedId, redraw]);

  // ─── Selected element (derived) ─────────────────────────────────────────
  const selectedElement = elements.find((e) => e.id === selectedId) ?? null;

  // ─── Add element from asset src ─────────────────────────────────────────
  const addElement = useCallback(
    (src) => {
      const img = new Image();
      img.src = src;
      img.crossOrigin = "anonymous";

      img.onload = () => {
        // Determine type from folder path
        const folder = src.split("/").slice(-2, -1)[0];
        const typeMap = {
          face: "face",
          eyes: "eyes",
          eyebrows: "eyebrows",
          nose: "nose",
          lips: "lips",
          hair: "hair",
          beard: "beard",
          moustache: "moustache",
          left_ears: "left ear",
          right_ears: "right ear",
        };
        const type = typeMap[folder] ?? "element";

        setElements((prev) => {
          // Remove same type (except ears)
          const filtered =
            type === "left ear" || type === "right ear"
              ? [...prev]
              : prev.filter((el) => el.type !== type);

          const face = filtered.find((el) => el.type === "face");
          const { x, y } = getSmartPosition(
            type,
            img,
            face,
            CANVAS_W,
            CANVAS_H,
          );
          const el = new CanvasElement(img, x, y, type);

          // Insert by layer priority
          const newPriority = LAYER_PRIORITY[type] ?? 100;
          let inserted = false;
          const next = [...filtered];
          for (let i = 0; i < next.length; i++) {
            if ((LAYER_PRIORITY[next[i].type] ?? 100) > newPriority) {
              next.splice(i, 0, el);
              inserted = true;
              break;
            }
          }
          if (!inserted) next.push(el);

          setSelectedId(el.id);
          return next;
        });

        showToast(`Added!`, "success");
      };

      img.onerror = () => showToast("Failed to load asset", "warning");
    },
    [showToast],
  );

  // ─── Mouse helpers ───────────────────────────────────────────────────────
  const getCanvasPos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      mx: (e.clientX - rect.left) * scaleX,
      my: (e.clientY - rect.top) * scaleY,
    };
  };

  const onMouseDown = useCallback((e) => {
    const { mx, my } = getCanvasPos(e);
    const elems = elementsRef.current;
    const selId = selectedIdRef.current;
    const selEl = elems.find((el) => el.id === selId);
    const s = stateRef.current;

    // Check resize handles first
    if (selEl) {
      const handle = selEl.getHitHandle(mx, my);
      if (handle) {
        s.isResizing = true;
        s.resizeHandle = handle.type;
        s.startX = mx;
        s.startY = my;
        s.startWidth = selEl.width;
        s.startHeight = selEl.height;
        s.startElementX = selEl.x;
        s.startElementY = selEl.y;
        canvasRef.current.style.cursor = handle.cursor;
        return;
      }
    }

    // Check click on element (top-down)
    for (let i = elems.length - 1; i >= 0; i--) {
      if (elems[i].isInside(mx, my)) {
        s.isDragging = true;
        s.offsetX = mx - elems[i].x;
        s.offsetY = my - elems[i].y;
        s.originalLayerIdx = i;
        setSelectedId(elems[i].id);
        return;
      }
    }

    // Clicked empty canvas
    setSelectedId(null);
  }, []);

  const onMouseMove = useCallback((e) => {
    const { mx, my } = getCanvasPos(e);
    const s = stateRef.current;
    const selId = selectedIdRef.current;

    // Cursor hint
    const selEl = elementsRef.current.find((el) => el.id === selId);
    if (selEl) {
      const handle = selEl.getHitHandle(mx, my);
      canvasRef.current.style.cursor = handle ? handle.cursor : "move";
    } else {
      canvasRef.current.style.cursor = "default";
    }

    if (!selId) return;

    if (s.isDragging) {
      setElements((prev) => {
        const next = prev.map((el) => {
          if (el.id !== selId) return el;
          el.x = mx - s.offsetX;
          el.y = my - s.offsetY;
          return el;
        });
        return [...next];
      });
    }

    if (s.isResizing && s.resizeHandle) {
      const dx = mx - s.startX;
      const dy = my - s.startY;
      setElements((prev) => {
        const next = prev.map((el) => {
          if (el.id !== selId) return el;
          el.applyResize(
            s.resizeHandle,
            dx,
            dy,
            s.startWidth,
            s.startHeight,
            s.startElementX,
            s.startElementY,
          );
          return el;
        });
        return [...next];
      });
    }
  }, []);

  const onMouseUp = useCallback(() => {
    const s = stateRef.current;
    s.isDragging = false;
    s.isResizing = false;
    s.resizeHandle = null;
  }, []);

  // ─── Keyboard shortcuts ──────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (!selectedIdRef.current) return;
      const keys = [
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "Delete",
        "Backspace",
      ];
      if (keys.includes(e.key)) e.preventDefault();

      switch (e.key) {
        case "Delete":
        case "Backspace":
          deleteSelected();
          break;
        case "ArrowUp":
          if (e.shiftKey) {
            bringForward();
          } else {
            setElements((prev) =>
              prev
                .map((el) =>
                  el.id === selectedIdRef.current ? ((el.y -= 5), el) : el,
                )
                .map((el) => ({ ...el })),
            );
          }
          break;
        case "ArrowDown":
          if (e.shiftKey) {
            sendBackward();
          } else {
            setElements((prev) =>
              prev
                .map((el) =>
                  el.id === selectedIdRef.current ? ((el.y += 5), el) : el,
                )
                .map((el) => ({ ...el })),
            );
          }
          break;
        case "ArrowLeft":
          setElements((prev) =>
            prev
              .map((el) =>
                el.id === selectedIdRef.current ? ((el.x -= 5), el) : el,
              )
              .map((el) => ({ ...el })),
          );
          break;
        case "ArrowRight":
          setElements((prev) =>
            prev
              .map((el) =>
                el.id === selectedIdRef.current ? ((el.x += 5), el) : el,
              )
              .map((el) => ({ ...el })),
          );
          break;
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []); // eslint-disable-line

  // ─── Layer controls ──────────────────────────────────────────────────────
  const deleteSelected = useCallback(() => {
    if (!selectedIdRef.current) {
      showToast("No element selected", "warning");
      return;
    }
    setElements((prev) => prev.filter((el) => el.id !== selectedIdRef.current));
    setSelectedId(null);
    showToast("Element deleted", "success");
  }, [showToast]);

  const bringForward = useCallback(() => {
    setElements((prev) => {
      const i = prev.findIndex((el) => el.id === selectedIdRef.current);
      if (i < 0 || i >= prev.length - 1) {
        showToast("Already on top", "info");
        return prev;
      }
      const next = [...prev];
      [next[i], next[i + 1]] = [next[i + 1], next[i]];
      showToast("Layer moved up", "info");
      return next;
    });
  }, [showToast]);

  const sendBackward = useCallback(() => {
    setElements((prev) => {
      const i = prev.findIndex((el) => el.id === selectedIdRef.current);
      if (i <= 0) {
        showToast("Already at bottom", "info");
        return prev;
      }
      const next = [...prev];
      [next[i], next[i - 1]] = [next[i - 1], next[i]];
      showToast("Layer moved down", "info");
      return next;
    });
  }, [showToast]);

  // ─── Transform sliders ───────────────────────────────────────────────────
  const setRotation = useCallback((val) => {
    setElements((prev) =>
      prev
        .map((el) => {
          if (el.id !== selectedIdRef.current) return el;
          el.rotation = val;
          return el;
        })
        .map((el) => ({ ...el })),
    );
  }, []);

  const setPositionX = useCallback((relX) => {
    setElements((prev) =>
      prev
        .map((el) => {
          if (el.id !== selectedIdRef.current) return el;
          const relY = el.getRelativePosition(CANVAS_W, CANVAS_H).y;
          el.setRelativePosition(relX, relY, CANVAS_W, CANVAS_H);
          return el;
        })
        .map((el) => ({ ...el })),
    );
  }, []);

  const setPositionY = useCallback((relY) => {
    setElements((prev) =>
      prev
        .map((el) => {
          if (el.id !== selectedIdRef.current) return el;
          const relX = el.getRelativePosition(CANVAS_W, CANVAS_H).x;
          el.setRelativePosition(relX, relY, CANVAS_W, CANVAS_H);
          return el;
        })
        .map((el) => ({ ...el })),
    );
  }, []);

  // ─── Select from layer panel ─────────────────────────────────────────────
  const selectById = useCallback((id) => {
    setSelectedId(id);
  }, []);

  // ─── Download ────────────────────────────────────────────────────────────
  const downloadImage = useCallback(() => {
    if (elementsRef.current.length === 0) {
      showToast("Add some elements first!", "warning");
      return;
    }
    const link = document.createElement("a");
    link.download = `face-${Date.now()}.png`;
    link.href = canvasRef.current.toDataURL("image/png");
    link.click();
    showToast("Image downloaded!", "success");
  }, [showToast]);

  return {
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
  };
}