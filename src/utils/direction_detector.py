"""
Campus-specific entry/exit direction detector for the ALPR University Gate system.

Supports camera-specific movement trajectory mappings:
  - CAMERA 1 (ch01 - Main Gate 2):
      IN:  Vehicle moves towards the lower-right of the image (+x, +y)
      OUT: Vehicle moves towards the upper-left of the image (-x, -y)
  - CAMERA 2 (ch02 - Gate 2):
      IN:  Vehicle moves towards the lower-left of the image (-x, +y)
      OUT: Vehicle moves towards the upper-right of the image (+x, -y)

Features:
  - Trajectory movement vector estimation across consecutive frames
  - Multi-frame confirmation before locking direction
  - Stationary vehicle suppression (min displacement threshold)
  - Rejection of ambiguous movement (returns UNKNOWN)
  - Single-crossing lock to prevent duplicate flip-flop events
  - Preserves VirtualLine crossing interface for backwards compatibility
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger

_logger = get_logger("utils.direction_detector")


@dataclass
class VirtualLine:
    """A virtual line defined by two points in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_config(cls, config: dict) -> "VirtualLine":
        line_cfg = config.get("direction", {}).get("virtual_line", {})
        return cls(
            x1=int(line_cfg.get("x1", 0)),
            y1=int(line_cfg.get("y1", 360)),
            x2=int(line_cfg.get("x2", 1280)),
            y2=int(line_cfg.get("y2", 360)),
        )


@dataclass
class CameraDirectionConfig:
    """Direction mapping configuration for a specific camera view."""
    camera_id: str
    name: str
    # Unit direction vector corresponding to vehicle entering campus [vx, vy]
    in_vector: tuple[float, float]
    # Unit direction vector corresponding to vehicle exiting campus [vx, vy]
    out_vector: tuple[float, float]
    # Optional virtual line
    virtual_line: Optional[VirtualLine] = None


# Default known campus profiles based on physical CCTV surveys
_DEFAULT_CAMERA_PROFILES: dict[str, CameraDirectionConfig] = {
    # CAMERA 1 — ch01 (Main Gate 2):
    # IN: lower-right (+x, +y) -> normalized vector (+0.7071, +0.7071)
    # OUT: upper-left (-x, -y) -> normalized vector (-0.7071, -0.7071)
    "ch01": CameraDirectionConfig(
        camera_id="ch01",
        name="Main Gate 2",
        in_vector=(0.7071, 0.7071),
        out_vector=(-0.7071, -0.7071),
        virtual_line=VirtualLine(x1=200, y1=700, x2=900, y2=300),
    ),
    # CAMERA 2 — ch02 (Gate 2):
    # IN: lower-left (-x, +y) -> normalized vector (-0.7071, +0.7071)
    # OUT: upper-right (+x, -y) -> normalized vector (+0.7071, -0.7071)
    "ch02": CameraDirectionConfig(
        camera_id="ch02",
        name="Gate 2",
        in_vector=(-0.7071, 0.7071),
        out_vector=(0.7071, -0.7071),
        virtual_line=VirtualLine(x1=100, y1=300, x2=800, y2=700),
    ),
    # Generic vertical default (downward = IN, upward = OUT)
    "default": CameraDirectionConfig(
        camera_id="default",
        name="Default Vertical",
        in_vector=(0.0, 1.0),
        out_vector=(0.0, -1.0),
        virtual_line=VirtualLine(x1=0, y1=360, x2=1280, y2=360),
    ),
}


class DirectionDetector:
    """Detect entry/exit direction using multi-frame trajectory vectors and camera profiles."""

    def __init__(
        self,
        line: Optional[VirtualLine] = None,
        dual_camera: bool = False,
        camera_profile: str = "ch01",
        min_displacement_px: float = 25.0,
        min_confirm_frames: int = 3,
        history_frames: int = 15,
        cone_angle_deg: float = 55.0,
        custom_profiles: Optional[dict[str, CameraDirectionConfig]] = None,
    ) -> None:
        self.dual_camera = dual_camera
        self.camera_profile = camera_profile
        self.min_displacement_px = float(min_displacement_px)
        self.min_confirm_frames = int(min_confirm_frames)
        self.history_frames = int(history_frames)
        self.cone_angle_deg = float(cone_angle_deg)
        self.cos_threshold = math.cos(math.radians(self.cone_angle_deg))

        # Profiles
        self._profiles = dict(_DEFAULT_CAMERA_PROFILES)
        if custom_profiles:
            self._profiles.update(custom_profiles)

        # Active configuration
        self._active_config = self._profiles.get(
            self.camera_profile.lower(),
            self._profiles["default"],
        )

        # Backwards compatibility
        self.line = line if line is not None else (self._active_config.virtual_line or VirtualLine(0, 360, 1280, 360))

        # Track state
        self._prev_centroids: dict[int, tuple[float, float]] = {}
        self._track_history: dict[int, deque[tuple[float, float]]] = {}
        self._track_votes: dict[int, deque[str]] = {}
        self._track_confirmed: dict[int, str] = {}

    @classmethod
    def from_config(cls, config: dict, camera_id: Optional[str] = None) -> "DirectionDetector":
        """Construct from application configuration dictionary."""
        dir_cfg = config.get("direction", {})
        dual_camera = bool(dir_cfg.get("dual_camera", False))

        profile_key = (
            camera_id
            or dir_cfg.get("active_camera")
            or config.get("camera", {}).get("camera_id")
            or "ch01"
        ).lower()

        if "01" in profile_key or "main" in profile_key:
            profile_key = "ch01"
        elif "02" in profile_key:
            profile_key = "ch02"
        elif profile_key not in _DEFAULT_CAMERA_PROFILES:
            profile_key = "default"

        custom_profiles = {}
        for cam_k, cam_v in dir_cfg.get("cameras", {}).items():
            if isinstance(cam_v, dict):
                in_vec = tuple(cam_v.get("in_vector", [0.7071, 0.7071]))
                out_vec = tuple(cam_v.get("out_vector", [-in_vec[0], -in_vec[1]]))
                vl_cfg = cam_v.get("virtual_line")
                vl = VirtualLine(x1=vl_cfg["x1"], y1=vl_cfg["y1"], x2=vl_cfg["x2"], y2=vl_cfg["y2"]) if vl_cfg else None
                custom_profiles[cam_k.lower()] = CameraDirectionConfig(
                    camera_id=cam_k,
                    name=cam_v.get("name", cam_k),
                    in_vector=(float(in_vec[0]), float(in_vec[1])),
                    out_vector=(float(out_vec[0]), float(out_vec[1])),
                    virtual_line=vl,
                )

        line = VirtualLine.from_config(config)

        return cls(
            line=line,
            dual_camera=dual_camera,
            camera_profile=profile_key,
            min_displacement_px=float(dir_cfg.get("min_displacement_px", 25.0)),
            min_confirm_frames=int(dir_cfg.get("min_confirm_frames", 3)),
            history_frames=int(dir_cfg.get("history_frames", 15)),
            cone_angle_deg=float(dir_cfg.get("cone_angle_deg", 55.0)),
            custom_profiles=custom_profiles,
        )

    def set_camera_profile(self, profile_name: str) -> None:
        """Switch active camera profile at runtime."""
        key = profile_name.lower()
        if "01" in key or "main" in key:
            key = "ch01"
        elif "02" in key:
            key = "ch02"

        if key in self._profiles:
            self.camera_profile = key
            self._active_config = self._profiles[key]
            if self._active_config.virtual_line:
                self.line = self._active_config.virtual_line
            _logger.info("Direction detector switched to profile: %s (%s)", key, self._active_config.name)
        else:
            _logger.warning("Unknown camera profile %s; retaining %s", profile_name, self.camera_profile)

    def update(
        self,
        track_id: int,
        centroid: tuple[float, float],
        camera_id: int = 0,
    ) -> Optional[str]:
        """Update tracker with new centroid and evaluate direction."""
        if self.dual_camera:
            return self._dual_camera_direction(camera_id)

        # 1. Maintain track centroid history buffer
        if track_id not in self._track_history:
            self._track_history[track_id] = deque(maxlen=self.history_frames)
            self._track_votes[track_id] = deque(maxlen=self.history_frames)

        history = self._track_history[track_id]
        history.append(centroid)

        prev = self._prev_centroids.get(track_id)
        self._prev_centroids[track_id] = centroid

        # If already confirmed, lock direction for this track to avoid duplicate flip-flop events
        if track_id in self._track_confirmed:
            return self._track_confirmed[track_id]

        # 2. Check virtual line crossing (provides instant crossing detection if configured)
        if prev is not None:
            line_cross = self._check_virtual_line(prev, centroid)
            if line_cross is not None:
                self._track_confirmed[track_id] = line_cross
                return line_cross

        # 3. Check if we have enough observations to evaluate trajectory vector
        if len(history) < 2:
            return None

        # 4. Calculate displacement over the history window
        first_pt = history[0]
        curr_pt = history[-1]
        dx = curr_pt[0] - first_pt[0]
        dy = curr_pt[1] - first_pt[1]
        displacement = math.hypot(dx, dy)

        # Requirement 6: Avoid classifying stationary vehicles or vehicles merely appearing
        if displacement < self.min_displacement_px:
            return None

        # 5. Compute unit movement vector
        vx = dx / displacement
        vy = dy / displacement

        # 6. Measure alignment against active camera profile
        in_vec = self._active_config.in_vector
        out_vec = self._active_config.out_vector

        dot_in = vx * in_vec[0] + vy * in_vec[1]
        dot_out = vx * out_vec[0] + vy * out_vec[1]

        # Classify current frame vote
        if dot_in >= self.cos_threshold:
            frame_vote = "IN"
        elif dot_out >= self.cos_threshold:
            frame_vote = "OUT"
        else:
            # Requirement 8: Ambiguous movement
            frame_vote = "UNKNOWN"

        self._track_votes[track_id].append(frame_vote)

        # 7. Multi-frame confirmation (Requirement 5)
        votes = list(self._track_votes[track_id])
        if len(votes) >= self.min_confirm_frames:
            recent_votes = votes[-self.min_confirm_frames:]
            if all(v == "IN" for v in recent_votes):
                self._track_confirmed[track_id] = "IN"
                return "IN"
            elif all(v == "OUT" for v in recent_votes):
                self._track_confirmed[track_id] = "OUT"
                return "OUT"

        return None

    def get_direction(self, track_id: int) -> str:
        """Return the confirmed direction for track_id, or estimated/UNKNOWN if unconfirmed."""
        if track_id in self._track_confirmed:
            return self._track_confirmed[track_id]

        history = self._track_history.get(track_id)
        if not history or len(history) < 2:
            return "UNKNOWN"

        dx = history[-1][0] - history[0][0]
        dy = history[-1][1] - history[0][1]
        displacement = math.hypot(dx, dy)

        if displacement < self.min_displacement_px:
            return "UNKNOWN"

        votes = list(self._track_votes.get(track_id, []))
        if len(votes) >= self.min_confirm_frames:
            recent_votes = votes[-self.min_confirm_frames:]
            if all(v == "IN" for v in recent_votes):
                return "IN"
            elif all(v == "OUT" for v in recent_votes):
                return "OUT"

        return "UNKNOWN"

    def is_confirmed(self, track_id: int) -> bool:
        """Return whether a direction has been definitively confirmed for track_id."""
        return track_id in self._track_confirmed

    def remove_track(self, track_id: int) -> None:
        """Clean up state when a track is retired."""
        self._prev_centroids.pop(track_id, None)
        self._track_history.pop(track_id, None)
        self._track_votes.pop(track_id, None)
        self._track_confirmed.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal & Virtual Line Helpers
    # ------------------------------------------------------------------

    def _check_virtual_line(
        self,
        prev: tuple[float, float],
        curr: tuple[float, float],
    ) -> Optional[str]:
        """Check if centroid crossed virtual line."""
        lx = self.line.x2 - self.line.x1
        ly = self.line.y2 - self.line.y1

        px = prev[0] - self.line.x1
        py = prev[1] - self.line.y1
        cx = curr[0] - self.line.x1
        cy = curr[1] - self.line.y1

        cross_prev = lx * py - ly * px
        cross_curr = lx * cy - ly * cx

        if cross_prev * cross_curr >= 0:
            return None

        return "IN" if cross_prev < 0 else "OUT"

    @staticmethod
    def _dual_camera_direction(camera_id: int) -> str:
        """Assign direction based on camera source."""
        return "IN" if camera_id == 0 else "OUT"
