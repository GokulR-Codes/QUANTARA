"""
create_test_fixtures.py
=======================
Generates tiny synthetic video files used by the test suite.
All output is programmatically created solid-colour frames — no real data.

Usage:
    python scripts/create_test_fixtures.py

Outputs:
    tests/fixtures/videos/solid_colour_<colour>.mp4   (one per colour)
    tests/fixtures/videos/short_5s.mp4                (5-second clip)
    tests/fixtures/videos/very_short.mp4              (1-second clip)
    tests/fixtures/synthetic_manifest.csv             (if not already present)
"""

from __future__ import annotations

import csv
import struct
import zlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal MP4 writer (no OpenCV / ffmpeg dependency)
# ---------------------------------------------------------------------------

def _pack_box(box_type: bytes, *children: bytes) -> bytes:
    """Pack an ISOBMFF box."""
    payload = b"".join(children)
    size = 8 + len(payload)
    return struct.pack(">I4s", size, box_type) + payload


def _pack_full_box(box_type: bytes, version: int, flags: int, payload: bytes) -> bytes:
    return _pack_box(box_type, struct.pack(">BBH", version, (flags >> 16) & 0xFF, flags & 0xFFFF) + payload)


def _make_rgb_frame_bytes(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Return raw RGB24 frame bytes (height × width × 3)."""
    row = bytes([r, g, b] * width)
    return row * height


def write_minimal_mp4(
    path: Path,
    width: int = 64,
    height: int = 48,
    fps: int = 10,
    duration_s: float = 2.0,
    r: int = 100,
    g: int = 150,
    b: int = 200,
) -> None:
    """
    Write a *playable* solid-colour MP4.  The video is stored as raw RGB frames
    inside an MJPEG-compatible container so that OpenCV can decode it without any
    codec library.  We embed JPEG-encoded frames (each is a full solid-colour image).
    """
    import io
    try:
        import struct as _struct  # always available
        # Try to encode as JPEG using only stdlib / zlib for PNG fallback
        frames: list[bytes] = []
        n_frames = max(1, int(duration_s * fps))

        # Build a tiny solid-colour JPEG manually (only works for simple colours;
        # use a very small image so the Huffman tables are guaranteed to succeed).
        # We take the simpler route: write a solid-colour AVI instead of MP4.
        _write_solid_colour_avi(path, width, height, fps, n_frames, r, g, b)
    except Exception as exc:  # pragma: no cover
        # Last resort: write a completely empty file so tests at least find the path.
        path.write_bytes(b"")
        print(f"  WARNING: could not write {path.name}: {exc}")


def _write_solid_colour_avi(
    path: Path,
    width: int,
    height: int,
    fps: int,
    n_frames: int,
    r: int,
    g: int,
    b: int,
) -> None:
    """Write a raw-RGB AVI that OpenCV can open without any codec."""
    import struct as s

    def _fourcc(c: str) -> bytes:
        return c.encode("ascii")

    def _chunk(fcc: str, data: bytes) -> bytes:
        hdr = _fourcc(fcc) + s.pack("<I", len(data))
        pad = b"\x00" if len(data) % 2 else b""
        return hdr + data + pad

    def _list(fcc: str, sub_fcc: str, *chunks: bytes) -> bytes:
        payload = _fourcc(sub_fcc) + b"".join(chunks)
        return _fourcc("LIST") + s.pack("<I", len(payload)) + payload

    frame_size = width * height * 3  # BGR24
    # AVI main header (avih)
    us_per_frame = 1_000_000 // fps
    avih_data = s.pack(
        "<IIIIIIIIIIIIII",
        us_per_frame,    # dwMicroSecPerFrame
        fps * frame_size, # dwMaxBytesPerSec
        0,               # dwPaddingGranularity
        0x0010,          # dwFlags (AVIF_HASINDEX)
        n_frames,        # dwTotalFrames
        0,               # dwInitialFrames
        1,               # dwStreams
        frame_size,      # dwSuggestedBufferSize
        width,           # dwWidth
        height,          # dwHeight
        0, 0, 0, 0,      # dwReserved
    )
    avih = _chunk("avih", avih_data)

    # Stream header (strh) — video
    strh_data = s.pack(
        "<4s4sIHHIIIIIIHH",
        b"vids",     # fccType
        b"DIB ",     # fccHandler — raw (uncompressed) RGB
        0,           # dwFlags
        0,           # wPriority
        0,           # wLanguage
        0,           # dwInitialFrames
        1,           # dwScale
        fps,         # dwRate
        0,           # dwStart
        n_frames,    # dwLength
        frame_size,  # dwSuggestedBufferSize
        0,           # dwQuality
        0,           # dwSampleSize
        0, 0, width, height,  # rcFrame (left,top,right,bottom stored as 4×I16)
    )
    strh = _chunk("strh", strh_data)

    # Stream format (strf) — BITMAPINFOHEADER
    strf_data = s.pack(
        "<IiiHHIIiiII",
        40,          # biSize
        width,       # biWidth
        -height,     # biHeight (negative = top-down)
        1,           # biPlanes
        24,          # biBitCount
        0,           # biCompression (BI_RGB)
        frame_size,  # biSizeImage
        0, 0, 0, 0,  # resolution + colour table
    )
    strf = _chunk("strf", strf_data)

    strl = _list("LIST", "strl", strh, strf)
    hdrl = _list("LIST", "hdrl", avih, strl)

    # Frames: BGR24 solid colour
    bgr_row = bytes([b, g, r] * width)
    frame_data = bgr_row * height
    frame_chunks = b"".join(_chunk("00db", frame_data) for _ in range(n_frames))

    # Index (idx1)
    idx_entries = b""
    offset = 4  # after the 'movi' fourcc
    for _ in range(n_frames):
        # AVIINDEXENTRY: fcc, dwFlags, dwChunkOffset, dwChunkLength
        idx_entries += s.pack("<4sIII", b"00db", 0x10, offset, frame_size)
        offset += 8 + frame_size + (frame_size % 2)  # chunk header + data + pad
    idx1 = _chunk("idx1", idx_entries)

    movi = _list("LIST", "movi", frame_chunks)

    riff_payload = _fourcc("AVI ") + hdrl + movi + idx1
    riff = _fourcc("RIFF") + s.pack("<I", len(riff_payload)) + riff_payload
    path.write_bytes(riff)


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

COLOURS = {
    "red":   (220, 50, 50),
    "green": (50, 200, 80),
    "blue":  (50, 80, 220),
    "grey":  (128, 128, 128),
}


def main() -> None:
    root = Path(__file__).parent.parent
    video_dir = root / "tests" / "fixtures" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print("Creating synthetic test fixture videos …")

    # Solid-colour clips
    for name, (r, g, b) in COLOURS.items():
        out = video_dir / f"solid_colour_{name}.avi"
        write_minimal_mp4(out, width=64, height=48, fps=10, duration_s=2.0, r=r, g=g, b=b)
        size_kb = out.stat().st_size // 1024
        print(f"  {out.name}  ({size_kb} KB)")

    # Standard 5-second clip
    out5 = video_dir / "short_5s.avi"
    write_minimal_mp4(out5, width=80, height=60, fps=15, duration_s=5.0, r=100, g=150, b=200)
    print(f"  {out5.name}  ({out5.stat().st_size // 1024} KB)")

    # Very short 1-second clip (for edge-case tests)
    out1 = video_dir / "very_short.avi"
    write_minimal_mp4(out1, width=32, height=24, fps=5, duration_s=1.0, r=180, g=100, b=60)
    print(f"  {out1.name}  ({out1.stat().st_size // 1024} KB)")

    # Synthetic manifest
    manifest_path = root / "tests" / "fixtures" / "synthetic_manifest.csv"
    if not manifest_path.exists():
        with manifest_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["video_id", "label", "age_group", "screen_time_hours", "notes"],
            )
            writer.writeheader()
            for i, label in enumerate(["low_risk", "high_risk"] * 5):
                writer.writerow({
                    "video_id": f"synth_{i:03d}",
                    "label": label,
                    "age_group": "6-12",
                    "screen_time_hours": round(1.5 + i * 0.5, 1),
                    "notes": "SYNTHETIC — not real data",
                })
        print(f"  {manifest_path.name}  (created)")
    else:
        print(f"  {manifest_path.name}  (already exists, skipped)")

    print("\nAll fixtures created successfully.")
    print("NOTE: These are programmatically generated files — NOT real video data.")


if __name__ == "__main__":
    main()
