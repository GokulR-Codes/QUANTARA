"""Package the project into a clean, verified ZIP (no secrets, no regenerable files).

Usage:  python scripts/package_delivery.py [--out DIR] [--name NAME.zip]

Excludes: .git, virtualenvs, node_modules, caches, build output, .env (any real env file), databases,
logs, uploaded videos / extracted frames / generated checkpoints.  Directory skeletons are kept with
.gitkeep files.  After writing, the archive is re-opened and checked: integrity, that every file that
should be inside is inside with an identical SHA-256, that nothing excluded slipped in, and that no
secret (values from your local .env, or common key patterns) appears in any member.
Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.name

EXCLUDE_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
                ".ruff_cache", ".idea", ".vscode", "dist", "build", ".tmp", ".eggs"}
EXCLUDE_DIR_SUFFIX = (".egg-info",)
EXCLUDE_FILE_NAMES = {".env", ".DS_Store", "Thumbs.db"}
EXCLUDE_FILE_SUFFIX = (".pyc", ".pyo", ".db", ".db-journal", ".db-wal", ".db-shm", ".sqlite", ".sqlite3", ".log",
                       ".tsbuildinfo", ".pem", ".key")
# generated runtime data (content excluded, directory skeleton kept)
RUNTIME_DIRS = ["data/uploads", "data/frames", "artifacts/checkpoints", "artifacts/experiments", "artifacts/models", "logs"]
SECRET_PATTERNS = [
    re.compile(rb"sk-[A-Za-z0-9]{20,}"), re.compile(rb"AKIA[0-9A-Z]{16}"), re.compile(rb"ghp_[A-Za-z0-9]{30,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"), re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
PLACEHOLDER_SECRET = "change-this-to-a-random-secret-key-before-deployment"


def is_excluded(rel: Path) -> bool:
    parts = rel.parts
    if any(p in EXCLUDE_DIRS or p.endswith(EXCLUDE_DIR_SUFFIX) for p in parts[:-1]):
        return True
    if rel.name in EXCLUDE_FILE_NAMES and rel.name != ".env.example":
        return True
    if rel.name.startswith(".env") and rel.name != ".env.example":
        return True
    if rel.name.endswith(EXCLUDE_FILE_SUFFIX):
        return True
    rel_posix = rel.as_posix()
    for d in RUNTIME_DIRS:
        if rel_posix.startswith(d + "/") and rel.name != ".gitkeep":
            return True
    return False


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_files() -> list[Path]:
    for d in RUNTIME_DIRS:                       # keep the skeleton
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        (ROOT / d / ".gitkeep").touch()
    out = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not is_excluded(p.relative_to(ROOT)):
            out.append(p)
    return out


def local_secret_values() -> list[bytes]:
    """Values of secret-looking keys in a local .env (never printed)."""
    vals = []
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if re.search(r"(KEY|TOKEN|PASSWORD|SECRET|PASSWD)", k, re.I) and len(v) >= 8 and v != PLACEHOLDER_SECRET:
                    vals.append(v.encode())
    return vals


def build(out_dir: Path, name: str) -> Path:
    files = collect_files()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, f"{PROJECT}/{p.relative_to(ROOT).as_posix()}")
    return target


def verify(archive: Path) -> list[str]:
    problems: list[str] = []
    expected = {f"{PROJECT}/{p.relative_to(ROOT).as_posix()}": p for p in collect_files()}
    secrets = local_secret_values()
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            problems.append(f"corrupt member: {bad}")
        names = {i.filename for i in z.infolist() if not i.is_dir()}
        for missing in sorted(set(expected) - names):
            problems.append(f"missing from archive: {missing}")
        for extra in sorted(names - set(expected)):
            problems.append(f"unexpected in archive: {extra}")
        for n in sorted(names & set(expected)):
            data = z.read(n)
            if sha(data) != sha(expected[n].read_bytes()):
                problems.append(f"content differs from working tree: {n}")
            if n.endswith("/.env") or Path(n).name == ".env":
                problems.append(f"real env file in archive: {n}")
            for s in secrets:
                if s in data:
                    problems.append(f"local secret value found in: {n}")
            for pat in SECRET_PATTERNS:
                if pat.search(data):
                    problems.append(f"secret-like pattern {pat.pattern[:18]!r} in: {n}")
            if Path(n).name == ".env.example":
                m = re.search(rb"^SECRET_KEY=(.*)$", data, re.M)
                if m and m.group(1).decode().strip() != PLACEHOLDER_SECRET:
                    problems.append(".env.example SECRET_KEY is not the placeholder")
        for n in names:
            if any(seg in n.split("/") for seg in ("node_modules", ".git", "__pycache__", ".venv")):
                problems.append(f"excluded directory content in archive: {n}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/mnt/user-data/outputs") if Path("/mnt/user-data/outputs").exists() else ROOT.parent)
    ap.add_argument("--name", default=f"project_backup_{dt.datetime.now():%Y-%m-%d_%H-%M}.zip")
    args = ap.parse_args()
    archive = build(args.out, args.name)
    problems = verify(archive)
    with zipfile.ZipFile(archive) as z:
        n = len(z.namelist())
    size_mb = archive.stat().st_size / 1e6
    print(f"archive: {archive}  ({n} entries, {size_mb:.2f} MB, sha256 {sha(archive.read_bytes())[:16]}…)")
    if problems:
        print("VERIFICATION FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("verification OK: integrity, contents==working tree (sha256), no excluded dirs, no .env, no secrets found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
