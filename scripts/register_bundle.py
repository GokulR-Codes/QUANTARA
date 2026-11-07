"""Register a pipeline bundle (upstream weights + QAOA mapping + VQC) with the backend DB.

Usage:  python scripts/register_bundle.py PATH_TO_BUNDLE_DIR [--inactive]

The bundle is fully validated first (files, checksums, dimension chain, VQC load).  A bundle
whose manifest declares synthetic training data is registered with a loud warning and will
still be REFUSED for real-video inference unless ALLOW_SYNTHETIC_MODEL_INFERENCE=true.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle_dir", type=Path)
    ap.add_argument("--inactive", action="store_true", help="register without making it the active bundle")
    args = ap.parse_args()

    from backend.app.db import SessionLocal, init_db
    from backend.app.services.model_service import register_bundle
    from src.config.settings import get_settings
    from src.pipeline.bundle import BundleError, load_bundle

    settings = get_settings()
    try:
        bundle = load_bundle(args.bundle_dir, settings, allow_synthetic=True)
    except BundleError as exc:
        print(f"REFUSED [{exc.code}]: {exc.message}")
        return 2

    td = bundle.manifest["training_data"]
    print(f"Bundle {bundle.manifest['bundle_id']} v{bundle.manifest['version']}: validated "
          f"(fused_dim={bundle.mapping['fused_dim']}, selected={bundle.mapping['selected_indices']}, "
          f"vqc_input_dim={bundle.vqc.input_dim}, classes={bundle.vqc.num_classes})")
    print(f"Declared training data: {td.get('kind')} — {td.get('description', '')}")
    if bundle.synthetic:
        usable = settings.allow_synthetic_model_inference
        print("WARNING: synthetic-trained bundle. Outputs have NO behavioural meaning. "
              f"Currently {'ALLOWED' if usable else 'REFUSED'} for video inference "
              "(ALLOW_SYNTHETIC_MODEL_INFERENCE).")

    init_db()
    with SessionLocal() as db:
        rec = register_bundle(db, args.bundle_dir, make_active=not args.inactive)
        print(f"Registered as model id {rec.id} (active={rec.is_active})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
