"""Video upload validation (extension, container signature, size)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.app.core.errors import VideoValidationError

_ISOBMFF_ATOMS = (b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip", b"pnot")
_EXT_FAMILY = {".mp4": "isobmff", ".mov": "isobmff", ".webm": "ebml", ".mkv": "ebml", ".avi": "riff"}


def sniff_container(head: bytes) -> Optional[str]:
    """Identify the container family from the first bytes, or None if unrecognised."""
    if len(head) >= 8 and head[4:8] in _ISOBMFF_ATOMS:
        return "isobmff"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "ebml"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "riff"
    return None


def validate_head(filename: str, head: bytes, accepted_formats: list[str]) -> None:
    """Validate extension and that the file content matches a known video container."""
    ext = Path(filename).suffix.lower()
    if ext not in accepted_formats:
        raise VideoValidationError(
            f"File format '{ext}' not accepted. Accepted formats: {', '.join(accepted_formats)}"
        )
    if len(head) < 12:
        raise VideoValidationError("File too small to be a valid video")
    family = sniff_container(head)
    if family is None:
        raise VideoValidationError(
            "File content is not a recognised video container (MP4/MOV, WebM/MKV or AVI)"
        )
    expected = _EXT_FAMILY.get(ext)
    if expected and family != expected:
        raise VideoValidationError(
            f"File content ({family}) does not match its extension '{ext}'"
        )


def validate_upload_file(
    filename: str, content: bytes, max_bytes: int, accepted_formats: list[str],
) -> None:
    """Validate an in-memory upload (kept for callers/tests that hold the whole file)."""
    if not content:
        raise VideoValidationError("Uploaded file is empty")
    if len(content) > max_bytes:
        raise VideoValidationError(
            f"File size {len(content) // (1024 * 1024)} MB exceeds limit of {max_bytes // (1024 * 1024)} MB"
        )
    validate_head(filename, content[:16], accepted_formats)


async def save_upload_streaming(upload, dest: Path, filename: str, max_bytes: int,
                                accepted_formats: list[str]) -> int:
    """Stream an UploadFile to ``dest`` in chunks (never holding the whole file in memory).

    The first chunk is validated before anything else is written; on any failure the
    partial file is removed.  Returns the number of bytes written.
    """
    total = 0
    first = True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                if first:
                    validate_head(filename, chunk[:16], accepted_formats)
                    first = False
                total += len(chunk)
                if total > max_bytes:
                    raise VideoValidationError(
                        f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit"
                    )
                out.write(chunk)
        if total == 0:
            raise VideoValidationError("Uploaded file is empty")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return total
