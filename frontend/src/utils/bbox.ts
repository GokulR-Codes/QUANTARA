// Pure geometry helpers for operator participant selection.
//
// All box coordinates handled here are in SOURCE-FRAME PIXELS (the resolution of the decoded
// video frame), never in CSS pixels.  The preview <img> may be displayed scaled, so every
// pointer position is converted with `clientToSource` before it becomes part of a box.

export interface Box { x: number; y: number; w: number; h: number }
export interface Size { width: number; height: number }
export interface Point { x: number; y: number }
export interface DisplayRect { left: number; top: number; width: number; height: number }

/** Smallest accepted box side in source pixels (the backend setting MIN_BOX_PX is authoritative). */
export const MIN_BOX_PX = 8;

const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);

/** Convert a pointer position (viewport CSS px) over the displayed frame to source-frame pixels. */
export function clientToSource(clientX: number, clientY: number, rect: DisplayRect, frame: Size): Point {
  if (rect.width <= 0 || rect.height <= 0 || frame.width <= 0 || frame.height <= 0) {
    throw new Error('Cannot map pointer: frame or display size is zero');
  }
  const x = ((clientX - rect.left) / rect.width) * frame.width;
  const y = ((clientY - rect.top) / rect.height) * frame.height;
  return { x: clamp(x, 0, frame.width), y: clamp(y, 0, frame.height) };
}

/** Integer box spanning two source-space points (any drag direction), kept inside the frame. */
export function boxFromPoints(a: Point, b: Point, frame: Size): Box {
  const x1 = clamp(Math.round(Math.min(a.x, b.x)), 0, frame.width);
  const y1 = clamp(Math.round(Math.min(a.y, b.y)), 0, frame.height);
  const x2 = clamp(Math.round(Math.max(a.x, b.x)), 0, frame.width);
  const y2 = clamp(Math.round(Math.max(a.y, b.y)), 0, frame.height);
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

/** Mirrors the backend rules; returns a human-readable error, or null if the box is acceptable. */
export function validateBox(box: Box | null, frame: Size | null, minSide: number = MIN_BOX_PX): string | null {
  if (!box) return 'Draw a box around the participant first.';
  if (!frame || frame.width <= 0 || frame.height <= 0) return 'Frame size is not known yet.';
  for (const [name, v] of Object.entries(box)) {
    if (!Number.isInteger(v)) return `Box ${name} must be a whole number of pixels.`;
  }
  if (box.x < 0 || box.y < 0) return 'Box starts outside the frame.';
  if (box.w < minSide || box.h < minSide) return `Box is too small; each side must be at least ${minSide} px.`;
  if (box.x + box.w > frame.width) return `Box exceeds the frame width (${box.x + box.w} > ${frame.width}).`;
  if (box.y + box.h > frame.height) return `Box exceeds the frame height (${box.y + box.h} > ${frame.height}).`;
  return null;
}

/** Source-space box -> CSS px relative to the top-left of the displayed frame (for the overlay). */
export function sourceToDisplay(box: Box, rect: DisplayRect, frame: Size): Box {
  const sx = rect.width / frame.width;
  const sy = rect.height / frame.height;
  return { x: box.x * sx, y: box.y * sy, w: box.w * sx, h: box.h * sy };
}

/** Parse the numeric input fields; returns null while any field is empty or not an integer. */
export function boxFromFields(fields: { x: string; y: string; w: string; h: string }): Box | null {
  const parsed = [fields.x, fields.y, fields.w, fields.h].map(s => (s.trim() === '' ? NaN : Number(s)));
  if (parsed.some(v => !Number.isInteger(v))) return null;
  return { x: parsed[0], y: parsed[1], w: parsed[2], h: parsed[3] };
}
