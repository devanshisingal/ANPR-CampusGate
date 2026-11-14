"""
FastAPI routes for the multi-camera trajectory system.

All mounted under /trajectory-api so they cannot collide with the existing
single-gate routes (/entry, /logs, /search, ...). Every existing endpoint
keeps its path and its behaviour; this module only adds.

  GET  /trajectory-api/cameras                camera registry
  POST /trajectory-api/cameras                add a camera site
  PATCH /trajectory-api/cameras/{id}          rename a camera site
  DEL  /trajectory-api/cameras/{id}           remove a camera site
  POST /trajectory-api/cameras/upload         upload every camera's video at once
  POST /trajectory-api/cameras/{id}/upload    upload one camera's video
  POST /trajectory-api/cameras/{id}/rtsp      switch a camera to RTSP
  POST /trajectory-api/processing/start       start the run (files in turn, streams together)
  POST /trajectory-api/processing/stop        request a stop
  GET  /trajectory-api/processing/status      live progress
  GET  /trajectory-api/trajectory/{plate}     reconstructed path
  GET  /trajectory-api/map/{plate}            path as GeoJSON
  GET  /trajectory-api/map/{plate}/html       path as a Leaflet page
  GET  /trajectory-api/vehicles               vehicle profiles, newest first
  GET  /trajectory-api/vehicles/{plate}       one vehicle's profile
  GET  /trajectory-api/statistics             session/camera aggregates
  GET  /trajectory-api/plates                 plates worth plotting
  GET  /trajectory-api/sessions               recent processing sessions
  DEL  /trajectory-api/sessions/{id}          delete a session and its events
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from src.api.schemas import (
    AddCameraRequest,
    CameraResponse,
    RenameCameraRequest,
    RTSPRequest,
    StartProcessingRequest,
    StartProcessingResponse,
    TrajectoryResponse,
    VehicleProfileResponse,
)
from src.cameras.registry import CameraConfigError, load_camera_registry
from src.cameras.status_store import StatusStore
from src.database import db as database
from src.database.vehicle_profiles import get_profile, list_profiles
from src.trajectory import TrajectoryEngine, trajectory_to_geojson
from src.utils.logger import get_logger

_logger = get_logger("api.trajectory_routes")
_REPO_ROOT = Path(__file__).resolve().parents[2]

router = APIRouter(prefix="/trajectory-api", tags=["multi-camera"])

# Largest video the API will accept, to keep one bad request from filling
# the Jetson's disk. Read fully into memory before writing, so this is also
# the memory ceiling per upload.
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ── lazily-built shared state ─────────────────────────────────────────────
#
# The registry is parsed once per process and cached. The CameraManager is
# built only when something actually asks to process or to see status --
# constructing it is cheap, but the pipeline it eventually imports is not,
# and a read-only request for the camera list should not pay for that.

_registry = None
_manager = None


def _get_registry():
    global _registry
    if _registry is None:
        try:
            _registry = load_camera_registry(
                str(_REPO_ROOT / "config" / "camera_config.yaml")
            )
        except CameraConfigError as exc:
            _logger.error("Camera configuration unavailable: %s", exc)
            raise HTTPException(
                status_code=503, detail=f"Camera configuration unavailable: {exc}"
            )
    return _registry


def _get_manager():
    global _manager
    if _manager is None:
        from src.cameras.manager import CameraManager
        from src.utils.config import load_config

        try:
            config = load_config(str(_REPO_ROOT / "config" / "config.yaml"))
        except Exception as exc:
            _logger.error("Could not load config.yaml for the camera manager: %s", exc)
            raise HTTPException(status_code=503, detail=f"Configuration unavailable: {exc}")
        _manager = CameraManager(config=config, registry=_get_registry())
    return _manager


def _get_engine() -> TrajectoryEngine:
    """A TrajectoryEngine configured from the registry's `trajectory:` block.

    Built per request rather than cached: it is a three-field object with no
    connections to hold, and rebuilding it means a config edit takes effect
    without restarting the API.
    """
    return TrajectoryEngine.from_registry(_get_registry())


_feed_settings_cache: Optional[tuple[tuple[int, int, int], tuple[dict, Path, Path, Optional[str]]]] = None


def _feed_settings() -> tuple[dict, Path, Path, Optional[str]]:
    """Resolve shared frame/status paths relative to the repository root."""
    global _feed_settings_cache
    from src.utils.config import load_config

    config_path = _REPO_ROOT / "config" / "config.yaml"
    try:
        config_mtime = config_path.stat().st_mtime_ns
    except OSError:
        config_mtime = 0
    registry = _get_registry()
    reg_path = getattr(registry, "source_path", None)
    try:
        reg_mtime = Path(reg_path).stat().st_mtime_ns if reg_path else 0
    except OSError:
        reg_mtime = 0
    status_file_cfg = str(registry.processing.get("status_file", ""))
    cache_key = (config_mtime, reg_mtime, id(registry), status_file_cfg)
    if _feed_settings_cache is not None and _feed_settings_cache[0] == cache_key:
        return _feed_settings_cache[1]

    config = load_config(str(config_path))
    api_config = config.get("api") or {}

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else _REPO_ROOT / path

    frames_dir = resolve(api_config.get("live_frames_dir", "data/live_frames"))
    status_path = resolve(registry.processing.get(
        "status_file", "data/processing_status.json"
    ))
    camera = config.get("camera") or {}
    configured_camera_id = camera.get("camera_id")
    result = (
        api_config, frames_dir, status_path,
        str(configured_camera_id) if configured_camera_id else None,
    )
    _feed_settings_cache = (cache_key, result)
    return result


def _camera_frame_path(
    camera_id: str,
    settings: Optional[tuple[dict, Path, Path, Optional[str]]] = None,
) -> Path:
    if not _CAMERA_ID_PATTERN.fullmatch(camera_id):
        raise HTTPException(status_code=400, detail="Invalid camera id")
    camera = _get_registry().get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"No camera {camera_id!r}")

    api_config, frames_dir, _, configured_camera_id = settings or _feed_settings()
    frame_path = frames_dir / f"{camera_id}.jpg"
    # The single-camera runner keeps its longstanding shared-frame path.
    if not frame_path.is_file():
        if configured_camera_id == camera_id:
            path = Path(api_config.get("live_frame_path", "data/live_frame.jpg"))
            frame_path = path if path.is_absolute() else _REPO_ROOT / path
    return frame_path


@router.get("/cameras/feeds")
def camera_feed_status():
    """Return each configured camera's real, freshness-based feed health."""
    settings = _feed_settings()
    api_config, _, status_path, _ = settings
    status = StatusStore(str(status_path)).read() or {}
    progress_by_id = {
        item.get("camera_id"): item for item in status.get("cameras", [])
        if item.get("camera_id")
    }
    now = time.time()
    stale_after = max(1.0, float(api_config.get("live_frame_stale_seconds", 15.0)))
    feeds = []
    for camera in _get_registry():
        frame_path = _camera_frame_path(camera.camera_id, settings)
        try:
            stat_result = frame_path.stat()
            modified_at = stat_result.st_mtime
            if stat_result.st_size < 4:
                raise OSError("camera frame is not a complete JPEG")
            with frame_path.open("rb") as frame_file:
                header = frame_file.read(2)
                frame_file.seek(-2, 2)
                trailer = frame_file.read(2)
            if header != b"\xff\xd8" or trailer != b"\xff\xd9":
                raise OSError("camera frame is not a complete JPEG")
            age = max(0.0, now - modified_at)
            online = camera.enabled and age <= stale_after
            last_frame_at = datetime.fromtimestamp(modified_at).astimezone().isoformat()
        except OSError:
            age = None
            online = False
            last_frame_at = None
        progress = progress_by_id.get(camera.camera_id) or {}
        processing_state = progress.get("state", "idle")
        is_running = processing_state == "running"
        has_frame = last_frame_at is not None
        online = camera.enabled and (is_running or (age is not None and age <= stale_after))
        feeds.append({
            "camera_id": camera.camera_id,
            "camera_name": camera.camera_name,
            "enabled": camera.enabled,
            "online": online,
            "has_frame": has_frame,
            "last_frame_at": last_frame_at,
            "frame_age_seconds": round(age, 2) if age is not None else None,
            "processing_state": processing_state,
            "snapshot_url": f"/trajectory-api/cameras/{camera.camera_id}/snapshot",
            "stream_url": f"/trajectory-api/cameras/{camera.camera_id}/stream",
        })
    return feeds


@router.get("/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: str):
    """Serve the latest annotated frame produced by this camera."""
    path = _camera_frame_path(camera_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No frame available for this camera")
    return FileResponse(
        str(path),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/cameras/{camera_id}/stream")
async def camera_stream(camera_id: str):
    """Serve the camera's annotated frames as an independent MJPEG stream."""
    settings = _feed_settings()
    path = _camera_frame_path(camera_id, settings)
    api_config = settings[0]
    boundary = "alprframe"
    poll_interval = 1.0 / max(
        float(api_config.get("live_frames_fps", 15.0)), 0.1
    )

    async def frame_generator():
        last_mtime = None
        last_data = None
        last_sent = 0.0
        while True:
            try:
                if path.is_file():
                    modified_at = path.stat().st_mtime
                    if modified_at != last_mtime:
                        last_mtime = modified_at
                        data = path.read_bytes()
                        last_data = data
                        last_sent = time.monotonic()
                        yield (
                            f"--{boundary}\r\n"
                            f"Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(data)}\r\n\r\n"
                        ).encode("ascii") + data + b"\r\n"
                    elif last_data is not None and (time.monotonic() - last_sent) >= 1.0:
                        # Heartbeat: repeat the latest frame once a second to keep proxies from hanging up
                        last_sent = time.monotonic()
                        yield (
                            f"--{boundary}\r\n"
                            f"Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(last_data)}\r\n\r\n"
                        ).encode("ascii") + last_data + b"\r\n"
            except FileNotFoundError:
                pass
            except asyncio.CancelledError:
                break
            except Exception:
                pass
            await asyncio.sleep(poll_interval)

    return StreamingResponse(
        frame_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


def _reconstruct(plate: str, processing_session: Optional[str]):
    """Load a plate's detections and reconstruct its path.

    Shared by every endpoint that returns a trajectory in some encoding, so
    the three of them cannot drift apart in how they order or collapse
    points.
    """
    try:
        with database.get_session() as session:
            detections = database.get_plate_detections(
                session, plate, processing_session=processing_session
            )
    except RuntimeError as exc:
        _logger.error("Database unavailable reconstructing %s: %s", plate, exc)
        raise HTTPException(status_code=503, detail="Database unavailable")

    return _get_engine().build(plate, detections, processing_session=processing_session)


# ── camera configuration ──────────────────────────────────────────────────


@router.get("/cameras", response_model=list[CameraResponse])
def list_cameras():
    """Return every configured camera, in processing order."""
    return [CameraResponse(**camera.to_dict()) for camera in _get_registry()]


@router.post("/cameras", response_model=CameraResponse, status_code=201)
def add_camera(request: AddCameraRequest):
    """Register a new camera site, and persist it.

    The camera is complete immediately: it takes an upload or an RTSP URL
    like any other, joins the end of the processing queue, and appears on the
    camera wall, the map and the heatmap.
    """
    manager = _get_manager()
    try:
        camera = manager.add_camera(
            camera_name=request.camera_name,
            latitude=request.latitude,
            longitude=request.longitude,
            camera_id=request.camera_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CameraResponse(**camera.to_dict())


@router.patch("/cameras/{camera_id}", response_model=CameraResponse)
def rename_camera(camera_id: str, request: RenameCameraRequest):
    """Rename a camera site.

    Only the label changes: the camera keeps its id, coordinates, queue
    position and source. Events already recorded keep the name they were
    stamped with -- they record what the site was called at the time.
    """
    manager = _get_manager()
    try:
        camera = manager.rename_camera(camera_id, request.camera_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No camera {camera_id!r}")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return CameraResponse(**camera.to_dict())


@router.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str):
    """Remove a camera site, its uploaded video and its preview frame.

    Detections it already recorded are kept: they are history, and they carry
    their own camera id and coordinates, so past trajectories still plot.
    """
    manager = _get_manager()
    try:
        camera = manager.remove_camera(camera_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No camera {camera_id!r}")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "camera_id": camera.camera_id,
        "camera_name": camera.camera_name,
        "message": f"Removed {camera.camera_name}; its recorded events were kept",
    }


@router.post("/cameras/{camera_id}/upload")
async def upload_camera_video(camera_id: str, file: UploadFile = File(...)):
    """Store an uploaded video and point *camera_id* at it for this session.

    Each camera gets its own file even when the same recording is uploaded
    four times -- the cameras stay independent, exactly as they would be
    with four real streams.
    """
    manager = _get_manager()
    if manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A processing session is running; stop it before changing sources",
        )

    try:
        data = await file.read()
    except Exception as exc:
        _logger.error("Failed to read upload for %s: %s", camera_id, exc)
        raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}")

    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {_MAX_UPLOAD_BYTES // (1024**3)} GB limit",
        )

    try:
        path = manager.save_upload(camera_id, file.filename or "", data)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No camera {camera_id!r}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        _logger.error("Failed to store upload for %s: %s", camera_id, exc)
        raise HTTPException(status_code=500, detail=f"Could not store upload: {exc}")

    return {
        "camera_id": camera_id,
        "video_path": str(path),
        "size_bytes": len(data),
        "message": f"Video stored for {camera_id}",
    }


@router.post("/cameras/upload")
async def upload_camera_videos(files: list[UploadFile] = File(...)):
    """Upload every camera's footage in one request.

    The bulk counterpart to the per-camera route above, and what the
    dashboard's single uploader calls. One file is broadcast to all enabled
    cameras (the four-virtual-cameras demo); several are assigned in queue
    order. Either way each camera receives its own copy on disk, so no two
    cameras share a video_path.
    """
    manager = _get_manager()
    if manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A processing session is running; stop it before changing sources",
        )

    uploads: list[tuple[str, bytes]] = []
    total_bytes = 0
    for upload in files:
        try:
            data = await upload.read()
        except Exception as exc:
            _logger.error("Failed to read upload %r: %s", upload.filename, exc)
            raise HTTPException(
                status_code=400, detail=f"Could not read {upload.filename!r}: {exc}"
            )
        total_bytes += len(data)
        # Checked against the running total, not per file: the limit exists to
        # protect the Jetson's disk and memory, and ten files just under the
        # cap would defeat a per-file check.
        if total_bytes > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploads exceed the {_MAX_UPLOAD_BYTES // (1024**3)} GB limit",
            )
        uploads.append((upload.filename or "", data))

    try:
        saved = manager.assign_uploads(uploads)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except OSError as exc:
        _logger.error("Failed to store bulk upload: %s", exc)
        raise HTTPException(status_code=500, detail=f"Could not store uploads: {exc}")

    return {
        "assigned": {camera_id: str(path) for camera_id, path in saved.items()},
        "files_received": len(uploads),
        "cameras_assigned": len(saved),
        "total_bytes": total_bytes,
        "message": (
            f"{len(uploads)} file(s) assigned to {len(saved)} camera(s)"
            + (" (one video broadcast to every camera)" if len(uploads) == 1 else "")
        ),
    }


@router.post("/cameras/{camera_id}/rtsp", response_model=CameraResponse)
def set_camera_rtsp(camera_id: str, request: RTSPRequest):
    """Switch a camera to a live RTSP stream for this session.

    The migration path in one call: nothing else about the camera, the
    pipeline or the dashboard changes. Persist it in camera_config.yaml to
    survive a restart.
    """
    manager = _get_manager()
    if manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A processing session is running; stop it before changing sources",
        )
    try:
        camera = manager.set_rtsp_url(camera_id, request.rtsp_url)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No camera {camera_id!r}")
    return CameraResponse(**camera.to_dict())


# ── processing control ────────────────────────────────────────────────────


def _queue_message(manager, queue) -> str:
    """Say what the run will actually do, per kind of source."""
    recorded, live = manager.split_queue(queue)
    parts = []
    if recorded:
        parts.append(
            f"{len(recorded)} recorded camera(s) one at a time: "
            + " -> ".join(camera.camera_name for camera in recorded)
        )
    if live:
        parts.append(
            f"{len(live)} live stream(s) together until stopped: "
            + ", ".join(camera.camera_name for camera in live)
        )
    return "; ".join(parts) or "Nothing to process"


@router.post("/processing/start", response_model=StartProcessingResponse)
def start_processing(request: StartProcessingRequest):
    """Start the run. Returns as soon as the queue is scheduled.

    Recorded videos are processed one at a time, in queue order; live RTSP
    cameras then all run together until /processing/stop.
    """
    manager = _get_manager()
    try:
        queue = manager.build_queue(request.camera_ids)
        session_id = manager.start(
            camera_ids=request.camera_ids, session_id=request.session_id
        )
    except RuntimeError as exc:
        # Already running, or nothing to run -- both are the caller's state
        # problem, not a server fault.
        raise HTTPException(status_code=409, detail=str(exc))

    return StartProcessingResponse(
        session_id=session_id,
        queued_cameras=[camera.camera_id for camera in queue],
        message=_queue_message(manager, queue),
    )


@router.post("/processing/stop")
def stop_processing():
    """Request that the running session stop after the current frame."""
    stopped = _get_manager().stop()
    return {
        "stopped": stopped,
        "message": (
            "Stop requested; the current camera will finish its flush"
            if stopped else "No processing session is running"
        ),
    }


@router.get("/processing/status")
def processing_status():
    """Live progress of the current or most recent session.

    Falls back to the status file when this process did not start the run --
    the dashboard and the API are usually separate processes.
    """
    try:
        return _get_manager().get_status()
    except HTTPException:
        # The manager needs config.yaml; the status file does not. A missing
        # config should not hide a run that is genuinely in progress.
        stored = StatusStore(
            str(_REPO_ROOT / _get_registry().processing.get(
                "status_file", "data/processing_status.json"
            ))
        ).read()
        if stored is None:
            raise
        return stored


# ── trajectory + map ──────────────────────────────────────────────────────


@router.get("/trajectory/{plate}", response_model=TrajectoryResponse)
def get_trajectory(
    plate: str,
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
):
    """Reconstruct where a vehicle went, as ordered points and legs."""
    trajectory = _reconstruct(plate, session)
    if trajectory.is_empty:
        raise HTTPException(
            status_code=404, detail=f"No detections for plate {plate.upper()}"
        )
    body = trajectory.to_dict()
    try:
        with database.get_session() as db_session:
            body["profile"] = get_profile(db_session, plate)
    except RuntimeError:
        body["profile"] = None
    return TrajectoryResponse(**body)


@router.get("/map/{plate}")
def get_map_data(
    plate: str,
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
):
    """The trajectory as GeoJSON: one LineString plus one Point per visit.

    Returned even when nothing is mappable (an empty FeatureCollection), so
    a map client never has to special-case the response shape.
    """
    return JSONResponse(trajectory_to_geojson(_reconstruct(plate, session)))


@router.get("/map/{plate}/html", response_class=HTMLResponse)
def get_map_html(
    plate: str,
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
    satellite: bool = Query(False, description="Open on satellite imagery"),
):
    """The trajectory as a standalone Leaflet page, ready to embed."""
    from src.mapping import render_trajectory_map

    trajectory = _reconstruct(plate, session)
    return HTMLResponse(
        render_trajectory_map(
            trajectory, repo_root=_REPO_ROOT, satellite_default=satellite
        )
    )


# ── statistics ────────────────────────────────────────────────────────────


@router.get("/vehicles", response_model=list[VehicleProfileResponse])
def list_vehicle_profiles(limit: int = Query(100, ge=1, le=1000)):
    """Vehicle profiles, most recently seen first."""
    try:
        with database.get_session() as db_session:
            return [VehicleProfileResponse(**p) for p in list_profiles(db_session, limit=limit)]
    except RuntimeError as exc:
        _logger.error("Database unavailable on /vehicles: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")


@router.get("/vehicles/{plate}", response_model=VehicleProfileResponse)
def get_vehicle_profile(plate: str):
    """One vehicle's profile: attributes, best images, camera visits."""
    try:
        with database.get_session() as db_session:
            profile = get_profile(db_session, plate)
    except RuntimeError as exc:
        _logger.error("Database unavailable on /vehicles/{plate}: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No vehicle profile for {plate.upper()}")
    return VehicleProfileResponse(**profile)


@router.get("/statistics")
def get_statistics(
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
):
    """Detection and unique-plate counts, overall and per camera."""
    try:
        with database.get_session() as db_session:
            return database.get_session_stats(db_session, processing_session=session)
    except RuntimeError as exc:
        _logger.error("Database unavailable on /statistics: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")


@router.get("/plates")
def list_trackable_plates(
    min_cameras: int = Query(2, ge=1, description="Minimum distinct cameras"),
    session: Optional[str] = Query(None, description="Restrict to one processing session"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Plates seen at enough cameras to have a path worth drawing."""
    try:
        with database.get_session() as db_session:
            return database.get_multi_camera_plates(
                db_session,
                min_cameras=min_cameras,
                processing_session=session,
                limit=limit,
            )
    except RuntimeError as exc:
        _logger.error("Database unavailable on /plates: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")


@router.delete("/sessions/{session_id}")
def delete_processing_session_route(session_id: str):
    """Delete one processing session and every event recorded in it.

    Removes the rows and the plate/vehicle crops they referenced. Refused
    while a session is running: deleting the events of a run that is still
    writing them would leave the status file describing detections that no
    longer exist.
    """
    from src.utils.config import load_config
    from src.utils.data_reset import delete_processing_session

    manager = _get_manager()
    if manager.is_running():
        raise HTTPException(
            status_code=409,
            detail="A processing session is running; stop it before deleting sessions",
        )

    try:
        config = load_config(str(_REPO_ROOT / "config" / "config.yaml"))
        counts = delete_processing_session(config, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        _logger.error("Database unavailable deleting session %s: %s", session_id, exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Failed to delete session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Clear the live status whether or not rows were found: a status file
    # can outlive its events (deleted elsewhere, or a cleared database).
    manager.forget_session(session_id)

    if counts["events"] == 0:
        raise HTTPException(
            status_code=404, detail=f"No events found for session {session_id!r}"
        )

    _logger.info(
        "Deleted session %s: %d event(s), %d image(s)",
        session_id, counts["events"], counts["images"],
    )
    return {
        "processing_session": session_id,
        "deleted_events": counts["events"],
        "deleted_images": counts["images"],
        "message": f"Deleted {counts['events']} event(s) from session {session_id}",
    }


@router.get("/sessions")
def list_sessions(limit: int = Query(20, ge=1, le=200)):
    """Recent processing sessions, newest first."""
    try:
        with database.get_session() as db_session:
            return database.get_processing_sessions(db_session, limit=limit)
    except RuntimeError as exc:
        _logger.error("Database unavailable on /sessions: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
