# Changelog

## Q-CARE v2.0 — Guardian Authentication & Age-Adaptive Assessment (2026-11-09)

### Added — Guardian Authentication System
- `backend/app/models_auth.py` — 7 new SQLAlchemy ORM models: `Guardian`, `OTPRecord`, `ChildProfile`, `ConsentRecord`, `AssessmentSession`, `AgeModelConfig`, `AuditLog`
- `backend/app/services/auth_service.py` — Full auth service: Argon2id password hashing, HS256 JWT sessions, OTP generation (SHA-256 hashed, 60s cooldown, 5-attempt limit), login throttling (5 failures → 15-minute lockout), password reset flow
- `backend/app/core/security.py` — FastAPI JWT dependencies: `get_current_guardian`, `get_current_verified_guardian`, `get_current_admin`
- `backend/app/api/v1/auth.py` — Auth router: register, login, logout, verify-otp, resend-otp, forgot-password, reset-password, get-me
- `backend/app/api/v1/children.py` — Child profiles CRUD with guardian isolation, mandatory age group validation, DoB conflict detection
- `backend/app/api/v1/assessments.py` — Assessment sessions linking child + job + age_group (recorded at assessment time)
- `backend/app/api/v1/age_models.py` — Age model status endpoint: 3 groups with processing config
- `src/models/age_profiles.py` — Age-specific QAOA/VQC/normalization/threshold configs per group (5–7, 7–10, 10–12)
- `src/models/age_model_loader.py` — Age-aware model loader (never substitutes across age groups)
- `scripts/create_admin.py` — CLI script for first admin account creation (idempotent)

### Added — Frontend (React/TypeScript)
- `frontend/src/types/auth.ts` — TypeScript types for auth, child profiles, assessments, age models
- `frontend/src/api/authClient.ts` — Auth API client with token management (localStorage)
- `frontend/src/contexts/AuthContext.tsx` — React auth context + `useAuth()` hook
- `frontend/src/components/ProtectedRoute.tsx` — Route guard (redirects to /auth)
- `frontend/src/pages/AuthPage.tsx` — Login, Register, ForgotPassword, VerifyEmail forms
- `frontend/src/pages/GuardianDashboard.tsx` — Post-login dashboard: child cards, model status, assessments
- `frontend/src/pages/ChildProfilePage.tsx` — Create/view/edit/delete child profiles with consent checkboxes
- `frontend/src/pages/PrivacyPage.tsx` — Full privacy & consent policy
- `frontend/src/pages/AgeModelsPage.tsx` — Age-specific model status page at `/age-models`
- `frontend/src/App.tsx` — Rewritten with `AuthProvider`, `AuthenticatedLayout`, protected routes

### Added — Tests
- `tests/integration/test_auth_children.py` — 35 integration tests covering registration, login, child profiles, age group validation, assessment sessions, OTP flow, guardian isolation

### Fixed
- **Password verification argument order** (`auth_service.py`): `argon2_hasher.verify(hashed, plain)` → `argon2_hasher.verify(plain, hashed)`. The passlib argon2 API takes `(secret, hash)` — the reversed arguments silently returned `False` for every password, making login impossible.
- **Test fixture**: `test_auth_children.py` `client` fixture took non-existent `app` parameter — removed; now uses the session-scoped `client` from `conftest.py`.
- **PytestReturnNotNoneWarning**: removed stray `return body` from `test_create_child_success`.

### Changed
- Backend title: "Q-CARE — Quantum-Centric AI for Child Behavioural Risk Evaluation"
- Camera/Upload pages now accept `?child=<childId>` query param and auto-link assessment sessions after upload
- Results page shows assessment context banner with child name and age group
- `.env.example` updated with auth/SMTP/age-model configuration variables
- `PROJECT_STATUS.md` updated with v2.0 results (244 tests, 99 modules, 32 requirements)
- 10 new DB tables (additive; existing 3 tables untouched)
- 47+ API routes (up from ~35)

### Database
- New tables: `guardians`, `otp_records`, `child_profiles`, `consent_records`, `assessment_sessions`, `age_model_configs`, `audit_logs`
- Existing tables (`jobs`, `experiments`, `model_registry`) unchanged

### Security
- Passwords: Argon2id (never plaintext, never logged)
- Sessions: HS256 JWT with configurable expiry
- OTPs: SHA-256 hashed, rate-limited, time-limited; dev mode prints to terminal only
- Login throttling: 5 failed attempts → 15-minute lockout
- Guardian isolation: children and assessments scoped per guardian

---

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
