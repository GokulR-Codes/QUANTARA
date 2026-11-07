# Changelog

## Integration & repair sprint (2026-10-26)

### Fixed — defects found in the uploaded code (all verified by execution)
- **Tracking could never work.** `cv2.TrackerCSRT_create`/`KCF` do not exist in `opencv-python-headless 4.10`
  (code silently fell back to MIL while claiming CSRT) **and** `tracker.init()` returns `None` in OpenCV ≥ 4.5.1,
  which the code treated as failure → every tracking attempt reported "init failed", swallowed by a blanket `except`.
  Now: `opencv-contrib-python-headless==4.10.0.84`, the real backend is chosen and *reported* (`tracker_backend`),
  `None` from `init()` is success.
- **QAOA Hamiltonian had the wrong sign** (ground state was the *complement* of the QUBO optimum: it would select the two
  least relevant features) and used a mean-field `⟨Z_i⟩⟨Z_j⟩` cost. Rewritten with exact Ising coefficients
  (`h_i = r_i/2 + λ(k − n/2)`), exact expectation from circuit probabilities, no global `np.random.seed`,
  no phantom zero-relevance qubits, and a report of the gap to the exhaustive QUBO optimum.
- **PennyLane could not be imported** with the declared pins (`autoray` resolved to 0.11 which removed `NumpyMimic`;
  `pennylane-lightning` resolved to 0.40 for PennyLane 0.38). This — not the code — caused the "10 failing tests":
  the original suite passes 52/52 once `autoray==0.6.12` / `pennylane-lightning==0.38.0` are pinned.
- **Inference used randomly initialised CNN/LSTM/fusion** and fed the 64-d fused vector to a VQC trained on 2-4 synthetic
  columns (dimension mismatch, `selected_feature_names` ignored). Replaced by checksummed *pipeline bundles*.
- QAOA+VQC *experiment* registered a checkpoint trained on random labels as the active model; it could then have produced a
  "prediction" for a real video. Now flagged synthetic, never usable for video inference.
- Frame sampling used index strides from a possibly bogus FPS; live-mode (MediaRecorder-style) WebM yielded nonsense
  metadata (frame_count ≈ −1.4e17). Now timestamp-based sampling + decode-scan probing.
- Extractor wrote zero-filled "padding" files and kept surplus frames on disk that the manifest did not list.
- Worker overwrote `cancelled` with later stages; `/result` returned 202 forever for cancelled jobs; `/process` on a job whose
  raw video was already deleted would clobber a good result; delete ignored active jobs and left crops behind.
- Upload read whole file into memory, computed a container signature and ignored it, never enforced the duration limit and
  trusted client-supplied frame dimensions.
- `pyproject.toml` build backend (`setuptools.backends.legacy:build`) does not exist → `setuptools.build_meta`.
- Frontend: 13 TypeScript diagnostics (6 unused + 7 real type errors), not 6.

### Added
- `src/video/geometry.py` (strict bbox validation vs. real frame size, letterboxed target crops),
  `src/video/tracker.py` (state machine: tracked / ambiguous / lost / out_of_frame; appearance, jump and scale guards;
  no automatic re-acquisition by default), `src/video/target_pipeline.py` (tracking aligned to sampled-frame timestamps,
  crops only for trusted frames, `tracking.json` audit record, invalid intervals).
- `src/pipeline/{features,bundle,runner}.py`: real CNN→LSTM→fusion over ordered crops (mask-only padding, provenance),
  bundle format with manifest + sha256 + dimension-chain validation, QAOA mapping + VQC head, explicit `model_status`.
- API: `GET /videos/{id}/frame-info`, `GET /videos/{id}/preview-frame`, `GET /videos/{id}/tracking`; selection now
  mandatory before `/process`; `/ready` reports tracker backend, PennyLane, bundle usability.
- Frontend: `TargetSelectionPage` (draw box on exact source frame, numeric fields, frame slider, server-validated confirm),
  pure geometry module `utils/bbox.ts` + 8 `node:test` tests, richer Results page, model registry panel on Settings.
- Additive SQLite column migration (`init_db`), `scripts/register_bundle.py`, `scripts/run_*.sh`, `scripts/package_delivery.py`.
- Tests: 209 Python tests (130 unit, 63 end-to-end through HTTP, 16 API) isolated in a temp dir; mutation-checked.

### Changed
- Tests no longer write into the repo (`conftest.py` redirects all storage to a temp dir).
- `requirements.txt` pins (see "Fixed"); `.env.example` completed (`CLASS_NAMES`, tracking options, model policy).
- `VQCClassifier.load` uses `torch.load(weights_only=True)`.

### Removed from the delivery (not from the source tree's functionality)
- `.env`, `.git`, `node_modules`, caches, `*.db`, uploads/frames, and `artifacts/checkpoints/classical_mlp_best.pt`
  (a checkpoint trained on synthetic data by the old tests, not a research asset).
