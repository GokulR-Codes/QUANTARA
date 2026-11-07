"""Small HTTP helpers shared by integration tests."""
from pathlib import Path


def upload(client, path: Path, name: str | None = None, mime: str = "video/mp4"):
    with open(path, "rb") as f:
        return client.post("/api/v1/videos/upload", files={"file": (name or Path(path).name, f, mime)})


def select(client, job_id: str, bbox: dict, size: tuple[int, int], frame_index: int = 0, **extra):
    body = {"bbox_x": bbox["x"], "bbox_y": bbox["y"], "bbox_w": bbox["w"], "bbox_h": bbox["h"],
            "frame_width": size[0], "frame_height": size[1], "frame_index": frame_index, **extra}
    return client.post(f"/api/v1/videos/{job_id}/select-target", json=body)


def run_job(client, video: Path, gt: dict, *, frame_index: int = 0):
    """upload -> explicit selection -> process (synchronous in TestClient) -> status JSON."""
    up = upload(client, video)
    assert up.status_code == 202, up.text
    job_id = up.json()["job_id"]
    bbox = {k: gt["truth"][frame_index][k] for k in ("x", "y", "w", "h")}
    sel = select(client, job_id, bbox, gt["size"], frame_index)
    assert sel.status_code == 200, sel.text
    pr = client.post(f"/api/v1/videos/{job_id}/process")
    assert pr.status_code == 202, pr.text
    st = client.get(f"/api/v1/videos/{job_id}/status")
    assert st.status_code == 200, st.text
    return job_id, st.json()
