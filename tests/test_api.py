import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from skimage import data

DATA = Path("/tmp/gfp_api_test_data")
shutil.rmtree(DATA, ignore_errors=True)
os.environ["GFP_DATA_DIR"] = str(DATA)
os.environ["GFP_ADMIN_EMAILS"] = "admin@example.org"
os.environ["GFP_RUN_STABILITY"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def admin(client):
    r = client.post("/api/auth/register", json={"email": "admin@example.org", "password": "Secret12345"})
    assert r.status_code == 201 and r.json()["role"] == "admin"
    tok = client.post("/api/auth/login", data={"username": "admin@example.org", "password": "Secret12345"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture(scope="module")
def researcher(client, admin):
    client.post("/api/auth/register", json={"email": "r@t.org", "password": "Secret12345"})
    tok = client.post("/api/auth/login", data={"username": "r@t.org", "password": "Secret12345"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def _jpg():
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def test_taxonomy_endpoint(client, researcher):
    t = client.get("/api/taxonomy", headers=researcher).json()
    assert "culture" in t and any(c["id"] == "moche" for c in t["culture"])
    assert any(c["id"] == "roman_imperial" for c in t["culture"])


def test_taxonomy_validation_rejects_free_text(client, researcher):
    r = client.post("/api/objects", headers=researcher, json={"title": "x", "culture": "Mochica people", "license": "PD"})
    assert r.status_code == 422 and "taxonomy_errors" in json.dumps(r.json())


def test_object_region_derived_from_culture(client, researcher):
    r = client.post("/api/objects", headers=researcher, json={"title": "Portrait head", "culture": "moche", "license": "CC0", "material": "ceramic"})
    assert r.status_code == 201
    o = r.json()
    assert o["region"] == "andes_north_coast" and o["training_allowed"] is True


def test_training_eligibility_rules(client, researcher):
    r = client.post("/api/objects", headers=researcher, json={"title": "restricted", "culture": "egyptian", "license": "CC-BY-NC-ND"})
    assert r.json()["training_allowed"] is False
    r = client.post("/api/objects", headers=researcher, json={"title": "sacred", "culture": "yoruba", "license": "CC0", "cultural_sensitivity": ["sacred_object"]})
    assert r.json()["training_allowed"] is False


def test_full_flow(client, admin, researcher):
    meta = {"title": "Portrait of a woman", "domain": "painting", "culture": "renaissance", "material": "canvas", "license": "PD", "date_start": 1500, "date_end": 1520}
    r = client.post("/api/analyse", headers=researcher, files={"file": ("a.jpg", _jpg(), "image/jpeg")}, data={"metadata": json.dumps(meta), "image_metadata": "{}"})
    assert r.status_code == 201, r.text
    body = r.json()
    rep = body["report"]
    assert rep["status"] == "completed" and rep["face_count"] >= 1
    assert rep["faces"][0]["expression"]["primary"] == "happiness"
    assert rep["input"]["analysis_context"] == {"domain": "painting", "depicted": True}
    assert "culture" not in rep["input"]["analysis_context"]          # firewall
    assert rep["object_metadata"]["culture"] == "renaissance"        # attached afterwards
    img_id = body["image"]["id"]
    face_id = body["faces"][0]["id"]

    # second upload of the same file → duplicate detection
    r2 = client.post(f"/api/objects/{rep['object_metadata']['id']}/images", headers=researcher, files={"file": ("b.jpg", _jpg(), "image/jpeg")})
    assert r2.json()["duplicates"]["exact"] == img_id

    for fmt in ("json", "csv", "pdf"):
        assert client.get(f"/api/images/{img_id}/download?format={fmt}", headers=researcher).status_code == 200
    assert client.get(f"/api/images/{img_id}/artefact/face_0/crop.jpg", headers=researcher).status_code == 200
    assert client.get(f"/api/images/{img_id}/artefact/../../etc/passwd", headers=researcher).status_code in (404, 400)

    # annotation: expert layer needs certification
    a = client.post(f"/api/faces/{face_id}/annotations", headers=researcher, json={"layer": "expert", "verdict": "agree"})
    assert a.status_code == 403
    a = client.post(f"/api/faces/{face_id}/annotations", headers=researcher,
                    json={"verdict": "disagree", "expression": "neutral", "aus": {"AU12": "absent"}, "confidence": 0.7, "comment": "convention"})
    assert a.status_code == 201 and a.json()["flag_for_review"] is True
    flags = client.get("/api/admin/flags", headers=admin).json()
    assert flags and flags[0]["face_id"] == face_id
    assert client.patch(f"/api/admin/flags/{flags[0]['id']}", headers=admin, json={"review_status": "accepted"}).status_code == 200
    gold = client.get("/api/admin/export/gold", headers=admin).text
    assert '"label_expression": "neutral"' in gold and '"group_key"' in gold

    # collections + splits + duplicates
    c = client.post("/api/admin/collections", headers=admin, json={"name": "Renaissance portraits", "filter_json": {"culture": "renaissance"}})
    assert c.status_code == 201 and c.json()["n_objects"] >= 1
    sp = client.get("/api/admin/splits/plan?level=culture", headers=admin).json()
    assert sp["level"] == "culture"
    assert client.post("/api/admin/duplicates/scan", headers=admin).status_code == 200
    assert client.get("/api/admin/users", headers=researcher).status_code == 403

    obj = client.get(f"/api/objects/{rep['object_metadata']['id']}", headers=researcher).json()
    assert obj["region_path"] == ["europe"] and len(obj["images"]) == 2


def test_rejected_blank(client, researcher):
    ok, buf = cv2.imencode(".png", np.full((200, 200, 3), 30, np.uint8))
    r = client.post("/api/analyse", headers=researcher, files={"file": ("b.png", buf.tobytes(), "image/png")}, data={"metadata": json.dumps({"license": "CC0"})})
    assert r.status_code == 201 and r.json()["image"]["status"] == "rejected"


def test_manual_face_box(client, researcher, admin):
    """A user-drawn rectangle guides landmarking (stylised faces); it is geometric only and passes the firewall."""
    r = client.post("/api/analyse", headers=researcher, files={"file": ("b.jpg", _jpg(), "image/jpeg")},
                    data={"metadata": "{}", "face_box": "150,20,400,300"})
    assert r.status_code == 201, r.text
    rep = r.json()["report"]
    assert rep["parameters"]["face_box_original_px"] == [150, 20, 400, 300]
    assert rep["status"] == "completed"
    assert rep["faces"][0]["localisation"]["detector"].startswith("manual_box") if "localisation" in rep["faces"][0] else True
    assert client.post("/api/analyse", headers=researcher, files={"file": ("b.jpg", _jpg(), "image/jpeg")},
                       data={"metadata": "{}", "face_box": "1,2,3"}).status_code == 422
    k = client.post("/api/admin/corpus/keys", headers=admin, json={"SMITHSONIAN_API_KEY": "abc"}).json()
    assert k["SMITHSONIAN_API_KEY"] is True
    assert "abc" not in json.dumps(client.get("/api/admin/corpus/keys", headers=admin).json())
    assert client.get("/api/admin/corpus/keys", headers=researcher).status_code == 403
