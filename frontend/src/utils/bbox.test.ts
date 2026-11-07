// Run with:  npm test      (Node's built-in test runner; no browser or extra dependency)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  boxFromFields, boxFromPoints, clientToSource, sourceToDisplay, validateBox, MIN_BOX_PX,
} from './bbox.ts';

const FRAME = { width: 1920, height: 1080 };
// The 1920x1080 frame is displayed at 640x360 CSS px, positioned at (100, 50) in the viewport.
const RECT = { left: 100, top: 50, width: 640, height: 360 };

test('pointer positions are scaled from display pixels to SOURCE pixels', () => {
  assert.deepEqual(clientToSource(100, 50, RECT, FRAME), { x: 0, y: 0 });
  assert.deepEqual(clientToSource(740, 410, RECT, FRAME), { x: 1920, y: 1080 });
  assert.deepEqual(clientToSource(420, 230, RECT, FRAME), { x: 960, y: 540 });   // centre
  const p = clientToSource(200, 100, RECT, FRAME);                              // 100px right, 50px down
  assert.equal(p.x, 300); assert.equal(p.y, 150);                               // x3 scale
});

test('pointer outside the displayed frame is clamped to the frame edge', () => {
  assert.deepEqual(clientToSource(-500, -500, RECT, FRAME), { x: 0, y: 0 });
  assert.deepEqual(clientToSource(5000, 5000, RECT, FRAME), { x: 1920, y: 1080 });
});

test('mapping with a zero-sized element is an error, not NaN coordinates', () => {
  assert.throws(() => clientToSource(10, 10, { ...RECT, width: 0 }, FRAME));
  assert.throws(() => clientToSource(10, 10, RECT, { width: 0, height: 10 }));
});

test('drag in any direction yields the same normalised integer box', () => {
  const a = { x: 100.4, y: 200.6 };
  const b = { x: 400.2, y: 500.5 };
  const expected = { x: 100, y: 201, w: 300, h: 300 };
  assert.deepEqual(boxFromPoints(a, b, FRAME), expected);
  assert.deepEqual(boxFromPoints(b, a, FRAME), expected);
  assert.deepEqual(boxFromPoints({ x: a.x, y: b.y }, { x: b.x, y: a.y }, FRAME), expected);
});

test('boxes from points never leave the frame', () => {
  const box = boxFromPoints({ x: -20, y: -20 }, { x: 5000, y: 5000 }, FRAME);
  assert.deepEqual(box, { x: 0, y: 0, w: 1920, h: 1080 });
  assert.equal(validateBox(box, FRAME), null);
});

test('display -> source -> display round trip is stable to within 1 source pixel', () => {
  const start = { x: 120, y: 80 }, end = { x: 360, y: 300 };                    // display-relative px
  const a = clientToSource(RECT.left + start.x, RECT.top + start.y, RECT, FRAME);
  const b = clientToSource(RECT.left + end.x, RECT.top + end.y, RECT, FRAME);
  const box = boxFromPoints(a, b, FRAME);
  assert.deepEqual(box, { x: 360, y: 240, w: 720, h: 660 });
  const back = sourceToDisplay(box, RECT, FRAME);
  assert.ok(Math.abs(back.x - start.x) <= 1 / 3 && Math.abs(back.w - (end.x - start.x)) <= 1 / 3);
});

test('validateBox mirrors the backend rules', () => {
  const ok = { x: 10, y: 10, w: 100, h: 100 };
  assert.equal(validateBox(ok, FRAME), null);
  assert.match(validateBox(null, FRAME)!, /Draw a box/);
  assert.match(validateBox(ok, null)!, /Frame size/);
  assert.match(validateBox({ ...ok, x: -1 }, FRAME)!, /outside the frame/);
  assert.match(validateBox({ ...ok, w: MIN_BOX_PX - 1 }, FRAME)!, /too small/);
  assert.match(validateBox({ ...ok, h: 0 }, FRAME)!, /too small/);
  assert.match(validateBox({ x: 1900, y: 10, w: 100, h: 50 }, FRAME)!, /exceeds the frame width/);
  assert.match(validateBox({ x: 10, y: 1050, w: 50, h: 50 }, FRAME)!, /exceeds the frame height/);
  assert.match(validateBox({ ...ok, x: 1.5 }, FRAME)!, /whole number/);
  assert.equal(validateBox({ x: 1820, y: 980, w: 100, h: 100 }, FRAME), null);   // touching the edge is fine
});

test('numeric field parsing rejects blanks and non-integers', () => {
  assert.deepEqual(boxFromFields({ x: '10', y: '20', w: '30', h: '40' }), { x: 10, y: 20, w: 30, h: 40 });
  assert.equal(boxFromFields({ x: '', y: '20', w: '30', h: '40' }), null);
  assert.equal(boxFromFields({ x: '1.5', y: '20', w: '30', h: '40' }), null);
  assert.equal(boxFromFields({ x: 'abc', y: '20', w: '30', h: '40' }), null);
});
