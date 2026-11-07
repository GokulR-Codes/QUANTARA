"""Environment check script — verifies all required dependencies.

Runs on Windows (cp1252 console) and Linux/macOS (UTF-8).
"""
import sys

# Force UTF-8 output on Windows so Unicode symbols display correctly.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

print(f"Python: {sys.version}")

checks = [
    ("torch",           "PyTorch"),
    ("torchvision",     "TorchVision"),
    ("numpy",           "NumPy"),
    ("sklearn",         "scikit-learn"),
    ("pennylane",       "PennyLane"),
    ("cv2",             "OpenCV"),
    ("fastapi",         "FastAPI"),
    ("uvicorn",         "Uvicorn"),
    ("sqlalchemy",      "SQLAlchemy"),
    ("pydantic",        "Pydantic"),
    ("pydantic_settings","pydantic-settings"),
    ("pandas",          "Pandas"),
    ("scipy",           "SciPy"),
]

OK  = "[OK]"
ERR = "[MISSING]"

all_ok = True
for mod, name in checks:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {OK} {name:<20} {ver}")
    except ImportError as e:
        print(f"  {ERR} {name:<20} MISSING — {e}")
        all_ok = False

print()

# Test quantum device
try:
    import pennylane as qml
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def test_circuit():
        qml.Hadamard(wires=0)
        return qml.probs(wires=[0, 1])

    result = test_circuit()
    print(f"  {OK} PennyLane quantum circuit: OK (probs={result.tolist()})")
except Exception as e:
    print(f"  {ERR} PennyLane quantum circuit: FAILED — {e}")
    all_ok = False

# Test PyTorch
try:
    import torch
    x = torch.randn(2, 3)
    _ = x @ x.T
    print(f"  {OK} PyTorch computation: OK (device=cpu, version={torch.__version__})")
except Exception as e:
    print(f"  {ERR} PyTorch computation: FAILED — {e}")
    all_ok = False

# Test OpenCV tracker
try:
    import cv2
    t = cv2.TrackerCSRT_create()
    backend = "CSRT"
    del t
    print(f"  {OK} OpenCV tracker: {backend} available ({cv2.__version__})")
except AttributeError:
    try:
        import cv2
        t = cv2.TrackerMIL_create()
        del t
        print(f"  [WARN] OpenCV tracker: CSRT unavailable; MIL available ({cv2.__version__}). "
              "Install opencv-contrib-python-headless for CSRT.")
    except Exception as e:
        print(f"  {ERR} OpenCV tracker: FAILED — {e}")
        all_ok = False
except Exception as e:
    print(f"  {ERR} OpenCV: FAILED — {e}")
    all_ok = False

# pip check
import subprocess
result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"  {OK} pip check: No broken requirements")
else:
    print(f"  {ERR} pip check: {result.stdout.strip()}")
    all_ok = False

print()
print("=" * 50)
if all_ok:
    print("All checks PASSED. Environment is ready.")
else:
    print("Some checks FAILED. Run: pip install -r requirements.txt")
    sys.exit(1)
