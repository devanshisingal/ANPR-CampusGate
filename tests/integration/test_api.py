"""
Integration tests for the FastAPI server endpoints.

Tests:
- POST /entry with valid body returns 201
- POST /entry with missing fields returns 422
- GET /logs returns array ordered by timestamp
- GET /search returns filtered results
- GET /health returns 200
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.database import db as database


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Initialize an in-memory test database before each test."""
    db_path = str(tmp_path / "test_api.db")
    database.init_db(db_path)
    yield
    database._engine = None
    database._SessionFactory = None


client = TestClient(app)


# ---------------------------------------------------------------------------
# POST /entry
# ---------------------------------------------------------------------------

class TestPostEntry:
    def _valid_payload(self, **overrides) -> dict:
        base = {
            "plate_number": "KA19TR0234",
            "vehicle_type": "Private",
            "plate_color":  "White",
            "series_type":  "normal",
            "direction":    "IN",
            "image_path":   "",
        }
        base.update(overrides)
        return base

    def test_valid_entry_returns_201(self):
        response = client.post("/entry", json=self._valid_payload())
        assert response.status_code == 201

    def test_valid_entry_returns_event_with_id(self):
        response = client.post("/entry", json=self._valid_payload())
        data = response.json()
        assert "id" in data
        assert data["plate_number"] == "KA19TR0234"

    def test_missing_plate_number_returns_422(self):
        payload = self._valid_payload()
        del payload["plate_number"]
        response = client.post("/entry", json=payload)
        assert response.status_code == 422

    def test_missing_direction_returns_422(self):
        payload = self._valid_payload()
        del payload["direction"]
        response = client.post("/entry", json=payload)
        assert response.status_code == 422

    def test_invalid_direction_returns_422(self):
        payload = self._valid_payload(direction="SIDEWAYS")
        response = client.post("/entry", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self):
        response = client.post("/entry", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /logs
# ---------------------------------------------------------------------------

class TestGetLogs:
    def test_empty_db_returns_empty_list(self):
        response = client.get("/logs")
        assert response.status_code == 200
        assert response.json() == []

    def test_logs_returns_inserted_events(self):
        client.post("/entry", json={
            "plate_number": "KA19TR0234", "vehicle_type": "Private",
            "plate_color": "White", "series_type": "normal",
            "direction": "IN", "image_path": "",
        })
        response = client.get("/logs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["plate_number"] == "KA19TR0234"

    def test_logs_ordered_newest_first(self):
        for plate in ["PLATE_A", "PLATE_B", "PLATE_C"]:
            client.post("/entry", json={
                "plate_number": plate, "vehicle_type": "Private",
                "plate_color": "White", "series_type": "normal",
                "direction": "IN", "image_path": "",
            })
        response = client.get("/logs")
        data = response.json()
        timestamps = [e["timestamp"] for e in data]
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------

class TestGetSearch:
    def test_search_returns_matching_plate(self):
        client.post("/entry", json={
            "plate_number": "KA19TR0234", "vehicle_type": "Private",
            "plate_color": "White", "series_type": "normal",
            "direction": "IN", "image_path": "",
        })
        client.post("/entry", json={
            "plate_number": "MH12AB1234", "vehicle_type": "Commercial",
            "plate_color": "Yellow", "series_type": "normal",
            "direction": "OUT", "image_path": "",
        })
        response = client.get("/search?plate=KA19TR0234")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["plate_number"] == "KA19TR0234"

    def test_search_unknown_plate_returns_empty(self):
        response = client.get("/search?plate=XX99ZZ9999")
        assert response.status_code == 200
        assert response.json() == []

    def test_search_missing_plate_param_returns_422(self):
        response = client.get("/search")
        assert response.status_code == 422


class TestDashboardFiltersAndAccessStatus:
    def test_stats_and_logs_honor_date_and_direction_filters(self):
        for plate, direction in (("DL01AA1111", "IN"), ("DL01AA2222", "OUT")):
            response = client.post("/entry", json={
                "plate_number": plate,
                "vehicle_type": "Private",
                "plate_color": "White",
                "series_type": "normal",
                "direction": direction,
                "image_path": "",
            })
            assert response.status_code == 201
            assert response.json()["access_status"] == "Unknown"

        today = datetime.now().astimezone().date().isoformat()
        params = {"start_date": today, "end_date": today, "direction": "IN"}
        response = client.get("/logs", params=params)
        assert response.status_code == 200
        assert [event["plate_number"] for event in response.json()] == ["DL01AA1111"]

        response = client.get("/stats", params=params)
        assert response.status_code == 200
        assert response.json()["entries"] == 1
        assert response.json()["exits"] == 0
        assert response.json()["total_events"] == 1

    def test_invalid_direction_filter_is_rejected(self):
        response = client.get("/logs", params={"direction": "SIDEWAYS"})
        assert response.status_code == 400

    def test_api_classifies_authorized_and_blacklisted_plates(self, tmp_path, monkeypatch):
        from src.alerts import blacklist as blacklist_module
        from src.alerts import whitelist as whitelist_module

        whitelist = whitelist_module.Whitelist(tmp_path / "whitelist.yaml")
        blacklist = blacklist_module.Blacklist(tmp_path / "blacklist.yaml")
        whitelist.add("KA01AA1234", owner="Campus")
        blacklist.add("KA01AA5678", reason="Flagged")
        monkeypatch.setattr(whitelist_module, "_default_whitelist", whitelist)
        monkeypatch.setattr(blacklist_module, "_default", blacklist)

        for plate, expected in (
            ("KA01AA1234", "Authorized"),
            ("KA01AA5678", "Blacklisted"),
        ):
            response = client.post("/entry", json={
                "plate_number": plate,
                "vehicle_type": "Private",
                "plate_color": "White",
                "series_type": "normal",
                "direction": "IN",
                "image_path": "",
            })
            assert response.status_code == 201
            assert response.json()["access_status"] == expected


class TestCameraFeedStatus:
    def test_camera_status_reflects_fresh_per_camera_frame(self, tmp_path, monkeypatch):
        from src.api import trajectory_routes
        from src.cameras.models import CameraConfig
        from src.cameras.registry import CameraRegistry

        registry = CameraRegistry([
            CameraConfig("CAM001", "Main Gate", None, None, 1),
            CameraConfig("CAM002", "Gate 2", None, None, 2),
        ], settings={"processing": {"status_file": str(tmp_path / "status.json")}})
        monkeypatch.setattr(trajectory_routes, "_get_registry", lambda: registry)
        monkeypatch.setattr(
            trajectory_routes,
            "_feed_settings",
            lambda: (
                {"live_frame_stale_seconds": 15},
                tmp_path,
                tmp_path / "status.json",
                None,
            ),
        )
        (tmp_path / "CAM001.jpg").write_bytes(b"\xff\xd8fresh frame\xff\xd9")

        response = client.get("/trajectory-api/cameras/feeds")
        assert response.status_code == 200
        feeds = {feed["camera_id"]: feed for feed in response.json()}
        assert feeds["CAM001"]["online"] is True
        assert feeds["CAM002"]["online"] is False
        assert response.json()[0]["stream_url"].endswith("/CAM001/stream")


class TestAccessControlEndpoints:
    def test_whitelist_can_be_added_listed_and_removed(self, tmp_path, monkeypatch):
        from src.alerts import whitelist as whitelist_module

        whitelist = whitelist_module.Whitelist(tmp_path / "whitelist.yaml")
        monkeypatch.setattr(whitelist_module, "_default_whitelist", whitelist)

        added = client.post("/access-control/whitelist", json={
            "plate": "ab-1234",
            "owner": "Campus Facilities",
        })
        assert added.status_code == 201
        assert added.json()["plate"] == "AB1234"

        listed = client.get("/access-control/whitelist")
        assert listed.status_code == 200
        assert len(listed.json()) == 1
        assert listed.json()[0]["plate"] == "AB1234"
        assert listed.json()[0]["owner"] == "Campus Facilities"
        assert listed.json()[0]["category"] == "Authorized"

        removed = client.delete("/access-control/whitelist/AB1234")
        assert removed.status_code == 200
        assert client.get("/access-control/whitelist").json() == []

    def test_blacklist_validates_and_manages_entries(self, tmp_path, monkeypatch):
        from src.alerts import blacklist as blacklist_module

        blacklist = blacklist_module.Blacklist(tmp_path / "blacklist.yaml")
        monkeypatch.setattr(blacklist_module, "_default", blacklist)

        invalid = client.post("/access-control/blacklist", json={"plate": "!!"})
        assert invalid.status_code == 400

        added = client.post("/access-control/blacklist", json={
            "plate": "zz88xy",
            "reason": "Security alert",
        })
        assert added.status_code == 201
        assert client.get("/access-control/blacklist").json()[0]["reason"] == "Security alert"
        assert client.delete("/access-control/blacklist/ZZ88XY").status_code == 200
        assert client.get("/access-control/blacklist").json() == []


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
