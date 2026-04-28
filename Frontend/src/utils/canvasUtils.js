// ─── Layer priority for auto-sorting ────────────────────────────────────────
export const LAYER_PRIORITY = {
  face: 0,
  hair: 1,
  "left ear": 2,
  "right ear": 2,
  eyebrows: 3,
  eyes: 4,
  nose: 5,
  moustache: 6,
  lips: 7,
  beard: 8,
};

// ─── Human-readable labels ───────────────────────────────────────────────────
export const ASSET_LABELS = {
  face: "Face",
  eyes: "Eyes",
  eyebrows: "Eyebrows",
  nose: "Nose",
  lips: "Lips",
  hair: "Hair",
  beard: "Beard",
  moustache: "Moustache",
  "left ear": "Left Ear",
  "right ear": "Right Ear",
};

// ─── Asset catalogue ─────────────────────────────────────────────────────────
// Each entry: { id, folder, count, type, label }
export const ASSET_CATALOGUE = [
  { id: "faces", folder: "face", count: 10, type: "face", label: "Face" },
  { id: "eyes", folder: "eyes", count: 12, type: "eyes", label: "Eyes" },
  {
    id: "eyebrows",
    folder: "eyebrows",
    count: 12,
    type: "eyebrows",
    label: "Eyebrows",
  },
  { id: "noses", folder: "nose", count: 12, type: "nose", label: "Nose" },
  { id: "lips", folder: "lips", count: 12, type: "lips", label: "Lips" },
  { id: "hairs", folder: "hair", count: 12, type: "hair", label: "Hair" },
  {
    id: "moustaches",
    folder: "moustache",
    count: 12,
    type: "moustache",
    label: "Moustache",
  },
  { id: "beards", folder: "beard", count: 12, type: "beard", label: "Beard" },
  {
    id: "left_ears",
    folder: "left_ears",
    count: 4,
    type: "left ear",
    label: "Left Ear",
  },
  {
    id: "right_ears",
    folder: "right_ears",
    count: 4,
    type: "right ear",
    label: "Right Ear",
  },
];

// ─── CanvasElement class ──────────────────────────────────────────────────────
export class CanvasElement {
  constructor(img, x, y, type) {
    this.img = img;
    this.x = x;
    this.y = y;
    this.width = img.width * 0.5;
    this.height = img.height * 0.5;
    this.rotation = 0;
    this.type = type;
    this.id = Date.now() + Math.random();
  }

  draw(ctx, isSelected) {
    ctx.save();
    ctx.translate(this.x + this.width / 2, this.y + this.height / 2);
    ctx.rotate(this.rotation);
    ctx.drawImage(
      this.img,
      -this.width / 2,
      -this.height / 2,
      this.width,
      this.height,
    );
    ctx.restore();

    if (isSelected) {
      ctx.save();
      ctx.strokeStyle = "#6c63ff";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(this.x, this.y, this.width, this.height);
      ctx.restore();
      this._drawHandles(ctx);
    }
  }

  _drawHandles(ctx) {
    const { x, y, width: w, height: h } = this;
    const pts = [
      [x, y],
      [x + w / 2, y],
      [x + w, y],
      [x + w, y + h / 2],
      [x + w, y + h],
      [x + w / 2, y + h],
      [x, y + h],
      [x, y + h / 2],
    ];
    pts.forEach(([px, py]) => {
      ctx.fillStyle = "white";
      ctx.strokeStyle = "#6c63ff";
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.rect(px - 4, py - 4, 8, 8);
      ctx.fill();
      ctx.stroke();
    });
  }

  isInside(mx, my) {
    return (
      mx > this.x &&
      mx < this.x + this.width &&
      my > this.y &&
      my < this.y + this.height
    );
  }

  getRelativePosition(canvasW, canvasH) {
    return {
      x: this.x - (canvasW / 2 - this.width / 2),
      y: this.y - (canvasH / 2 - this.height / 2),
    };
  }

  setRelativePosition(relX, relY, canvasW, canvasH) {
    this.x = canvasW / 2 - this.width / 2 + relX;
    this.y = canvasH / 2 - this.height / 2 + relY;
  }

  getHandles() {
    const { x, y, width: w, height: h } = this;
    return [
      { x, y, type: "nw", cursor: "nw-resize" },
      { x: x + w / 2, y, type: "n", cursor: "n-resize" },
      { x: x + w, y, type: "ne", cursor: "ne-resize" },
      { x: x + w, y: y + h / 2, type: "e", cursor: "e-resize" },
      { x: x + w, y: y + h, type: "se", cursor: "se-resize" },
      { x: x + w / 2, y: y + h, type: "s", cursor: "s-resize" },
      { x, y: y + h, type: "sw", cursor: "sw-resize" },
      { x, y: y + h / 2, type: "w", cursor: "w-resize" },
    ];
  }

  getHitHandle(mx, my) {
    for (const h of this.getHandles()) {
      if (mx > h.x - 6 && mx < h.x + 6 && my > h.y - 6 && my < h.y + 6)
        return h;
    }
    return null;
  }

  applyResize(handle, dx, dy, startW, startH, startX, startY) {
    const MIN = 20;
    switch (handle) {
      case "se":
        this.width = Math.max(MIN, startW + dx);
        this.height = Math.max(MIN, startH + dy);
        break;
      case "nw":
        this.width = Math.max(MIN, startW - dx);
        this.height = Math.max(MIN, startH - dy);
        this.x = startX + dx;
        this.y = startY + dy;
        break;
      case "ne":
        this.width = Math.max(MIN, startW + dx);
        this.height = Math.max(MIN, startH - dy);
        this.y = startY + dy;
        break;
      case "sw":
        this.width = Math.max(MIN, startW - dx);
        this.height = Math.max(MIN, startH + dy);
        this.x = startX + dx;
        break;
      case "n":
        this.height = Math.max(MIN, startH - dy);
        this.y = startY + dy;
        break;
      case "s":
        this.height = Math.max(MIN, startH + dy);
        break;
      case "e":
        this.width = Math.max(MIN, startW + dx);
        break;
      case "w":
        this.width = Math.max(MIN, startW - dx);
        this.x = startX + dx;
        break;
    }
  }
}

// ─── Smart positioning when a face is on canvas ──────────────────────────────
export function getSmartPosition(type, img, face, canvasW, canvasH) {
  let x = canvasW / 2 - img.width / 4;
  let y = canvasH / 2 - img.height / 4;

  if (face && type !== "face") {
    const { width: fw, height: fh, x: fx, y: fy } = face;
    const cx = fx + fw / 2;
    const map = {
      eyebrows: { x: cx - img.width * 0.25, y: fy + fh * 0.25 },
      eyes: { x: cx - img.width * 0.25, y: fy + fh * 0.35 },
      nose: { x: cx - img.width * 0.25, y: fy + fh * 0.48 },
      moustache: { x: cx - img.width * 0.25, y: fy + fh * 0.6 },
      lips: { x: cx - img.width * 0.25, y: fy + fh * 0.7 },
      beard: { x: cx - img.width * 0.35, y: fy + fh * 0.8 },
      hair: { x: cx - img.width * 0.5, y: fy - img.height * 0.35 },
      "left ear": { x: fx - img.width * 0.3, y: fy + fh * 0.4 },
      "right ear": { x: fx + fw - img.width * 0.7, y: fy + fh * 0.4 },
    };
    if (map[type]) {
      x = map[type].x;
      y = map[type].y;
    }
  }
  return { x, y };
}
