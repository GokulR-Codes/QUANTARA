import pytest
from fastapi.testclient import TestClient

from tests.helpers.media import make_target_video


@pytest.fixture(scope="session")
def client():
    from backend.app.db import init_db
    from backend.app.main import app
    init_db()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _clear_registry():
    from backend.app.db import SessionLocal
    from backend.app.models import ModelRegistry
    with SessionLocal() as s:
        s.query(ModelRegistry).delete()
        s.commit()


@pytest.fixture(autouse=True)
def isolation(monkeypatch, client):
    """Every test starts with no registered models, the real tracker, and default settings."""
    from backend.app.workers import processing_worker
    monkeypatch.setattr(processing_worker, "TRACKER_FACTORY", None)
    _clear_registry()
    yield
    _clear_registry()


@pytest.fixture()
def set_setting(monkeypatch, settings):
    def _set(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(settings, k, v)
    return _set


@pytest.fixture(scope="session")
def moving_video(tmp_path_factory):
    """40 frames @10 fps, 160x120, target present throughout (4.0 s)."""
    p = tmp_path_factory.mktemp("vid") / "moving.mp4"
    return p, make_target_video(p, n_frames=40, fps=10.0)


def make_variant(tmp_path_factory, name, **kw):
    p = tmp_path_factory.mktemp("vid") / name
    return p, make_target_video(p, **kw)


@pytest.fixture()
def register_bundle(synthetic_bundle_dir):
    from backend.app.db import SessionLocal
    from backend.app.services.model_service import register_bundle

    def _reg(path=None, make_active=True):
        with SessionLocal() as s:
            return register_bundle(s, path or synthetic_bundle_dir, make_active=make_active).id
    return _reg
