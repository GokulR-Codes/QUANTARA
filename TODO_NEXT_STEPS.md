# TODO — Final State

Status legend: [x] done  [ ] open  [~] partly done

## P0 — Complete ✅

- [x] Live acceptance run: upload → select → process → independent retrieval ✅
- [x] README.md created (full architecture, quickstart, limitations) ✅
- [x] docs/ directory created (architecture, quantum_methodology, video_processing, privacy_and_safety, troubleshooting, dataset_preparation) ✅
- [x] Final full test run: **209 passed, 0 failed** ✅
- [x] Frontend `npm run build` verified: 90 modules, 0 TypeScript errors ✅
- [x] Delivery ZIP: `d:\Final\QCFL_project_delivery.zip` (verified, no secrets) ✅
- [x] `scripts/create_test_fixtures.py` created ✅
- [x] `datetime.utcnow()` deprecation fixed across all backend files ✅
- [x] `asyncio_default_fixture_loop_scope` warning fixed (pyproject.toml) ✅
- [x] ffmpeg 9.x compatibility (`-vsync` → `-fps_mode`) fixed in test_extraction.py ✅
- [x] Frontend title fixed (`frontend_temp` → `Quantum Federated Behaviour AI`) ✅

## P1 — Known Gaps (require user / hardware / data) ⚠️

- [ ] **No trained pipeline bundle exists.** Real predictions need: labelled video dataset, trained CNN/LSTM/fusion weights, QAOA selection, VQC training → `scripts/register_bundle.py`.
      Until then every job ends `completed_without_prediction` (`model_not_ready`) — by design.
- [ ] Browser camera recording (`useCamera`/MediaRecorder) and the draw-a-box UI were **never executed in a real browser** (no browser in CI). Test manually: record → "Save & Select Participant" → draw box → Confirm → Start.
- [ ] Tracker box accuracy on flat objects is modest (CSRT box shrinks); validate on real footage.
- [ ] Appearance guard is an HSV-histogram heuristic: cannot guarantee identity (similar clothing can pass).

## P2 — Polish (non-blocking) ℹ️

- [ ] 4 pre-existing oxlint warnings (`useJob`, `Dashboard`, `ExperimentsPage` effects) — cosmetic
- [ ] Authentication is not implemented (settings declare JWT fields only) — do not expose beyond localhost
