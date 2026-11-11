"""
Vehicle whitelist: authorized campus vehicles (students, faculty, staff, delivery).

Stored in config/whitelist.yaml so it survives restarts and can be edited
from the Streamlit dashboard or by hand. The file is watched and re-read
whenever it changes on disk.
"""

from __future__ import annotations

import re
import threading
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from src.utils.logger import get_logger

_logger = get_logger("alerts.whitelist")

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "whitelist.yaml"


def normalize_plate(plate: Optional[str]) -> str:
    """Normalize plate string: upper-case, strip spaces and dashes."""
    return re.sub(r"[\s\-]", "", (plate or "")).upper()


class Whitelist:
    """Manage authorized campus registration numbers stored in YAML."""

    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime: Optional[float] = None
        self._entries: dict[str, dict] = {}

    # ── reading ──────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._entries, self._mtime = {}, None
            return
        if mtime == self._mtime:
            return
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            _logger.warning("Could not read whitelist %s: %s", self.path, exc)
            return

        entries = {}
        for item in data.get("plates") or []:
            if isinstance(item, str):
                item = {"plate": item}
            plate = normalize_plate(item.get("plate") if isinstance(item, dict) else None)
            if plate:
                entries[plate] = {
                    "plate": plate,
                    "owner": item.get("owner") or "",
                    "category": item.get("category") or "Authorized",
                    "added": str(item.get("added") or ""),
                }
        self._entries, self._mtime = entries, mtime

    def entries(self) -> list[dict]:
        with self._lock:
            self._refresh()
            return sorted(self._entries.values(), key=lambda e: e["plate"])

    def plates(self) -> set[str]:
        return {e["plate"] for e in self.entries()}

    def get(self, plate: Optional[str]) -> Optional[dict]:
        with self._lock:
            self._refresh()
            return self._entries.get(normalize_plate(plate))

    def contains(self, plate: Optional[str]) -> bool:
        return self.get(plate) is not None

    # ── writing ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        body = {
            "plates": [
                {
                    "plate": e["plate"],
                    "owner": e["owner"],
                    "category": e["category"],
                    "added": e["added"],
                }
                for e in sorted(self._entries.values(), key=lambda e: e["plate"])
            ]
        }
        header = "# Whitelisted vehicles -- authorized campus plates (students, staff, faculty, delivery).\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".yaml.tmp")
        tmp.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        tmp.replace(self.path)
        self._mtime = None

    def add(self, plate: str, owner: str = "", category: str = "Authorized") -> str:
        """Add an authorized vehicle plate."""
        normalized = normalize_plate(plate)
        if not re.fullmatch(r"[A-Z0-9]{4,12}", normalized):
            raise ValueError(f"{plate!r} is not a valid plate number")
        with self._lock:
            self._refresh()
            self._entries[normalized] = {
                "plate": normalized,
                "owner": owner.strip(),
                "category": category.strip() or "Authorized",
                "added": date.today().isoformat(),
            }
            self._save()
        _logger.info("Whitelisted %s (Owner: %s, Category: %s)", normalized, owner, category)
        return normalized

    def remove(self, plate: str) -> bool:
        """Remove a vehicle plate from whitelist."""
        normalized = normalize_plate(plate)
        with self._lock:
            self._refresh()
            if normalized not in self._entries:
                return False
            del self._entries[normalized]
            self._save()
        _logger.info("Removed %s from the whitelist", normalized)
        return True


_default_whitelist: Optional[Whitelist] = None


def get_whitelist() -> Whitelist:
    """Return the global Whitelist singleton."""
    global _default_whitelist
    if _default_whitelist is None:
        _default_whitelist = Whitelist()
    return _default_whitelist


def classify_vehicle(plate: Optional[str]) -> dict:
    """Classify a vehicle plate as Blacklisted, Authorized, or Unknown.

    Returns:
        dict:
            status: 'Blacklisted' | 'Authorized' | 'Unknown'
            badge_text: 'BLACKLISTED' | 'AUTHORIZED' | 'UNKNOWN'
            badge_class: CSS class name for badge
            color: Hex color string (#dc3545, #28a745, #f59e0b)
            details: Matching entry dict or None
    """
    from src.alerts.blacklist import get_blacklist

    normalized = normalize_plate(plate)
    if not normalized:
        return {
            "status": "Unknown",
            "badge_text": "UNKNOWN",
            "badge_class": "badge-unknown",
            "color": "#f59e0b",
            "details": None,
        }

    # 1. Check Blacklist first (Security priority)
    bl_entry = get_blacklist().get(normalized)
    if bl_entry:
        return {
            "status": "Blacklisted",
            "badge_text": "BLACKLISTED",
            "badge_class": "badge-blacklisted",
            "color": "#dc3545",
            "details": bl_entry,
        }

    # 2. Check Whitelist
    wl_entry = get_whitelist().get(normalized)
    if wl_entry:
        return {
            "status": "Authorized",
            "badge_text": "AUTHORIZED",
            "badge_class": "badge-authorized",
            "color": "#28a745",
            "details": wl_entry,
        }

    # 3. Default: Unknown / Visitor
    return {
        "status": "Unknown",
        "badge_text": "UNKNOWN",
        "badge_class": "badge-unknown",
        "color": "#f59e0b",
        "details": None,
    }


def classify_status_batch(plates) -> dict[str, str]:
    """Return {normalized_plate: status} for many plates with two cached lookups."""
    from src.alerts.blacklist import get_blacklist

    bl_plates = get_blacklist().plates()
    wl_plates = get_whitelist().plates()
    statuses: dict[str, str] = {}
    for plate in plates:
        norm = normalize_plate(plate)
        if not norm or norm in statuses:
            continue
        if norm in bl_plates:
            statuses[norm] = "Blacklisted"
        elif norm in wl_plates:
            statuses[norm] = "Authorized"
        else:
            statuses[norm] = "Unknown"
    return statuses
