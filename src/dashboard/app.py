"""
Enhanced Streamlit dashboard for ALPR University Gate system.

Features:
  - Live camera feed with real-time frame monitoring and stale frame detection
  - Dynamic database filtering (direction, date range, plate search)
  - Security Blacklist alert banners with snapshots and repeat suppression
  - Campus Whitelist access control (Authorized / Blacklisted / Unknown classification)
  - Daily traffic statistics and analytics
  - Vehicle history tracking and timeline
  - CSV export functionality

Run with:
  streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.alerts.blacklist import get_blacklist
from src.alerts.whitelist import classify_vehicle, get_whitelist
from src.database import db as database
from src.utils.config import load_config

# ── Page Configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="ALPR University Gate",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling badges and cards
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
    }
    .plate-badge {
        background-color: #FFD700;
        padding: 4px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-family: monospace;
        font-size: 1.05rem;
        color: #111;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-authorized {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-blacklisted {
        background-color: #dc3545;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-unknown {
        background-color: #f59e0b;
        color: #111;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .alert-card-danger {
        background-color: #fff5f5;
        border: 2px solid #dc3545;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .status-in { color: #28a745; font-weight: bold; }
    .status-out { color: #dc3545; font-weight: bold; }
    .status-unknown { color: #f59e0b; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# ── Session State & Connection ───────────────────────────────────────
@st.cache_resource
def init_db_connection() -> bool:
    """Initialize database connection."""
    try:
        config_path = str(REPO_ROOT / "config" / "config.yaml")
        config = load_config(config_path)
        db_path = config.get("database", {}).get("path", "data/alpr.db")
        database.init_db(db_path)
        return True
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return False


@st.cache_data(ttl=5)
def load_config_cached() -> dict:
    """Load and cache configuration."""
    try:
        config_path = str(REPO_ROOT / "config" / "config.yaml")
        return load_config(config_path)
    except Exception as e:
        st.error(f"Config load failed: {e}")
        return {}


# ── Data Loading Functions ────────────────────────────────────────────
def load_recent_events(
    direction_filter: list[str] | None = None,
    days: int = 7,
    plate_filter: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Load events from database applying direction, date range, and plate search filters."""
    try:
        with database.get_session() as session:
            start_date = None
            if days:
                start_date = (
                    datetime.now().astimezone().date()
                    - timedelta(days=max(1, days) - 1)
                )
            return database.get_filtered_events(
                session,
                start_date=start_date,
                end_date=datetime.now().astimezone().date() if days else None,
                directions=direction_filter,
                plate_search=plate_filter,
                limit=limit,
            )
    except Exception as exc:
        st.error(f"Failed to load events: {exc}")
        return []


def load_direction_events(direction: str, limit: int = 50) -> list[dict]:
    """Load events filtered strictly by direction."""
    try:
        with database.get_session() as session:
            return database.get_events_by_direction(session, direction, limit=limit)
    except Exception as exc:
        st.error(f"Failed to load {direction} events: {exc}")
        return []


def search_plate_history(plate: str) -> list[dict]:
    """Search all historical events for an exact plate number."""
    try:
        with database.get_session() as session:
            return database.search_events(session, plate_number=plate.strip().upper())
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return []


def get_daily_stats() -> dict:
    """Get today's traffic statistics."""
    try:
        with database.get_session() as session:
            return database.get_daily_stats(session)
    except Exception as exc:
        st.error(f"Failed to get stats: {exc}")
        return {}


def prepare_dataframe(events: list[dict]) -> pd.DataFrame:
    """Convert events list to a Pandas DataFrame."""
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _html_table(df: pd.DataFrame) -> None:
    """Render a DataFrame as plain HTML to avoid pyarrow ABI mismatch crashes."""
    from html import escape

    if df is None or len(df) == 0:
        st.caption("Nothing to show.")
        return

    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{escape('--' if pd.isna(v) else str(v))}</td>" for v in row
        ) + "</tr>"
        for row in df.itertuples(index=False, name=None)
    )
    st.markdown(
        '<div style="overflow-x:auto"><table style="border-collapse:collapse;'
        'width:100%;font-size:.85rem">'
        f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ── Dashboard Main ────────────────────────────────────────────────────
def main():
    if not init_db_connection():
        st.stop()

    config = load_config_cached()

    st.title("🚗 ALPR University Gate — Campus Monitor")
    st.markdown("Real-time license plate recognition, live CCTV surveillance, and access control")

    # ── Sidebar Controls ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")

        auto_refresh = st.checkbox("Auto-refresh", value=True, key="auto_refresh")
        refresh_interval = st.slider("Refresh interval (sec)", 2, 30, 5)

        st.divider()

        st.subheader("🔍 Quick Plate Search")
        search_plate = st.text_input(
            "Filter by plate number", "", placeholder="e.g., DL3CBJ1384"
        ).strip().upper()

        st.divider()

        st.subheader("🎯 Filters")
        filter_direction = st.multiselect("Direction", ["IN", "OUT", "UNKNOWN"], default=["IN", "OUT"])
        filter_days = st.slider("Last N days", 1, 30, 7)

        st.divider()

        st.subheader("📊 System Action")
        if st.button("Refresh Now"):
            st.rerun()

    # ── Main Tabs ─────────────────────────────────────────────────────
    tab_live, tab_search, tab_analytics, tab_history, tab_access, tab_settings = st.tabs(
        ["📋 Live Feed", "🔍 Search", "📊 Analytics", "🚗 Vehicle History", "🛡️ Access Control", "⚙️ Settings"]
    )

    # ──────────────────────── TAB: Live Feed ──────────────────────────
    with tab_live:
        # 1. Live Camera Feed Section
        st.subheader("📹 Live Gate Feed & Real-Time Surveillance")
        col_feed, col_feed_info = st.columns([3, 2])

        live_frame_rel = config.get("api", {}).get("live_frame_path", "data/live_frame.jpg")
        live_frame_file = REPO_ROOT / live_frame_rel

        with col_feed:
            if live_frame_file.exists():
                mtime = live_frame_file.stat().st_mtime
                age_seconds = time.time() - mtime
                if age_seconds <= 12.0:
                    st.caption(f"🟢 **Camera Stream Active** · Frame latency: {age_seconds:.1f}s")
                else:
                    st.caption(f"⚠️ **Feed Idle / Paused** · Last frame received: {int(age_seconds)}s ago")
                st.image(str(live_frame_file), caption="Live CCTV Stream (YOLO Detections & OCR)", use_container_width=True)
            else:
                st.warning("📷 **Camera feed offline**: 'data/live_frame.jpg' not found. Start `python scripts/run_pipeline.py` to stream frames.")

        with col_feed_info:
            st.markdown("#### ℹ️ Gate & Camera Details")
            cam_meta = config.get("camera", {})
            st.write(f"**Gate Name:** {cam_meta.get('camera_name', 'Main Gate')}")
            st.write(f"**Camera ID:** `{cam_meta.get('camera_id', 'GATE-01')}`")
            st.write(f"**Video Source:** `{config.get('video', {}).get('source', 'ALPR.mp4')}`")
            det_cfg = config.get("detection", {})
            ensemble_cfg = det_cfg.get("plate_ensemble", {})
            if ensemble_cfg.get("enabled"):
                st.write("**Plate Detector:** Ensemble Fusion (`best.pt` + `best_finetuned.pt`)")
            else:
                plate_model_name = Path(det_cfg.get("plate_model_path", "best_finetuned.onnx")).name
                st.write(f"**Plate Detector:** Single Model (`{plate_model_name}`)")
            st.write(f"**OCR Engine:** `{config.get('ocr', {}).get('backend', 'rapidocr').title()}`")
            st.caption("Frames are analyzed with ByteTrack tracking, plate cropping, SR upscaling, and OCR majority voting.")

        st.divider()

        # 2. Blacklist Security Alerts Section
        bl = get_blacklist()
        all_recent_events = load_recent_events(limit=50)
        blacklisted_sightings = [e for e in all_recent_events if bl.contains(e.get("plate_number"))]

        if blacklisted_sightings:
            # Group consecutive frames per plate to avoid flooding repeated alerts
            unique_alerts = {}
            for ev in blacklisted_sightings:
                plate = ev.get("plate_number")
                if plate not in unique_alerts:
                    unique_alerts[plate] = ev

            st.markdown("### 🚨 SECURITY ALERT — Blacklisted Vehicle Detected")
            for plate, ev in unique_alerts.items():
                entry = bl.get(plate) or {}
                reason = entry.get("reason") or "Flagged vehicle / Unauthorized entry"
                ts = ev.get("timestamp", "N/A")
                camera = ev.get("camera_name") or ev.get("camera_id") or "Main Gate"
                direction = ev.get("direction", "IN")
                snap_path = ev.get("vehicle_thumbnail_path") or ev.get("plate_thumbnail_path") or ev.get("image_path")

                col_alert_text, col_alert_snap = st.columns([3, 1])
                with col_alert_text:
                    st.markdown(
                        f"""
                        <div class="alert-card-danger">
                            <span class="badge-blacklisted">⛔ BLACKLISTED VEHICLE: {plate}</span>
                            <p style="margin-top: 8px; margin-bottom: 2px; font-size: 1rem;">
                                <strong>Reason:</strong> {reason}<br>
                                <strong>Direction:</strong> {direction} &nbsp;|&nbsp;
                                <strong>Camera:</strong> {camera} &nbsp;|&nbsp;
                                <strong>Time:</strong> {ts}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_alert_snap:
                    if snap_path and Path(snap_path).exists():
                        st.image(snap_path, width=140, caption=f"Capture: {plate}")

        # 3. Filtered Recent Events Table
        st.subheader("📋 Recent Vehicle Detections")

        events = load_recent_events(
            direction_filter=filter_direction,
            days=filter_days,
            plate_filter=search_plate,
            limit=100,
        )

        if events:
            col1, col2, col3, col4 = st.columns(4)
            in_events = sum(1 for e in events if e.get("direction") == "IN")
            out_events = sum(1 for e in events if e.get("direction") == "OUT")
            unique_plates = len(set(e.get("plate_number", "") for e in events))

            with col1:
                st.metric("📥 Filtered Entries", in_events)
            with col2:
                st.metric("📤 Filtered Exits", out_events)
            with col3:
                st.metric("🚗 Unique Plates", unique_plates)
            with col4:
                st.metric("📍 Matching Events", len(events))

            st.write(f"**Showing {len(events)} matching event(s)**")

            for event in events[:25]:
                with st.container():
                    col_img, col_info = st.columns([1, 4])

                    with col_img:
                        vehicle_thumb = event.get("vehicle_thumbnail_path") or ""
                        plate_thumb = event.get("plate_thumbnail_path") or ""
                        img_path = event.get("image_path", "")
                        shown = False
                        if vehicle_thumb and Path(vehicle_thumb).exists():
                            st.image(vehicle_thumb, width=120)
                            shown = True
                        plate_img = plate_thumb if plate_thumb and Path(plate_thumb).exists() else img_path
                        if plate_img and Path(plate_img).exists():
                            st.image(plate_img, width=120)
                            shown = True
                        if not shown:
                            st.markdown("📷 *No snapshot*")

                    with col_info:
                        plate = event.get("plate_number", "N/A")
                        direction = event.get("direction", "N/A")
                        timestamp = event.get("timestamp", "N/A")
                        vehicle_type = event.get("vehicle_type", "N/A")
                        color = event.get("plate_color", "N/A")
                        series = event.get("series_type", "N/A")
                        camera_name = event.get("camera_name") or "Main Gate"
                        camera_id = event.get("camera_id") or "GATE-01"
                        latitude = event.get("latitude")
                        longitude = event.get("longitude")
                        confidence = event.get("confidence")

                        # Access control classification (Authorized / Blacklisted / Unknown)
                        classification = classify_vehicle(plate)

                        # Badges line
                        if direction == "IN":
                            direction_class = "status-in"
                            direction_emoji = "📥"
                        elif direction == "OUT":
                            direction_class = "status-out"
                            direction_emoji = "📤"
                        else:
                            direction_class = "status-unknown"
                            direction_emoji = "❓" 
                        conf_str = f" · OCR Conf: {confidence:.2f}" if confidence else ""

                        st.markdown(
                            f'<span class="plate-badge">{plate}</span> &nbsp; '
                            f'<span class="{classification["badge_class"]}">{classification["badge_text"]}</span> &nbsp; '
                            f'<span class="{direction_class}">{direction_emoji} {direction}</span>{conf_str}',
                            unsafe_allow_html=True,
                        )

                        col_d1, col_d2, col_d3 = st.columns(3)
                        vehicle_class = event.get("vehicle_class")
                        vehicle_color = event.get("vehicle_color")
                        with col_d1:
                            st.write(f"🚗 {vehicle_class.title()} ({vehicle_type})" if vehicle_class else f"🚗 {vehicle_type}")
                        with col_d2:
                            st.write(f"🎨 {vehicle_color}" if vehicle_color else f"🎨 {color}")
                        with col_d3:
                            st.write(f"📌 {series} · plate {color}")

                        loc_str = f" · 📍 {latitude:.5f}, {longitude:.5f}" if (latitude and longitude) else ""
                        st.caption(f"⏰ {timestamp} · 🎥 {camera_name} ({camera_id}){loc_str}")

                    st.divider()
        else:
            st.info("No events match the selected filters (direction / date / plate search).")

    # ──────────────────────── TAB: Search ─────────────────────────────
    with tab_search:
        st.subheader("🔍 Search by Plate Number")

        active_search = search_plate or st.text_input("Enter plate number", "", placeholder="e.g., DL3CBJ1384", key="search_tab_input").strip().upper()

        if active_search:
            results = search_plate_history(active_search)

            if results:
                classification = classify_vehicle(active_search)
                st.markdown(
                    f"### Results for `{active_search}` &nbsp; "
                    f"<span class='{classification['badge_class']}'>{classification['badge_text']}</span>",
                    unsafe_allow_html=True,
                )

                df_results = prepare_dataframe(results)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Sightings", len(results))
                with col2:
                    entries = sum(1 for r in results if r.get("direction") == "IN")
                    st.metric("Entries", entries)
                with col3:
                    exits = sum(1 for r in results if r.get("direction") == "OUT")
                    st.metric("Exits", exits)

                st.divider()

                st.write("**Event Timeline**")
                if not df_results.empty and "timestamp" in df_results.columns:
                    fig = px.timeline(
                        df_results,
                        x_start="timestamp",
                        x_end="timestamp",
                        y="direction",
                        color="direction",
                        title=f"Activity for {active_search}",
                        color_discrete_map={"IN": "#28a745", "OUT": "#dc3545"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.write("**Detailed Event Log**")
                display_cols = ["plate_number", "direction", "vehicle_type", "plate_color",
                                "camera_id", "camera_name", "timestamp"]
                df_display = df_results[[col for col in display_cols if col in df_results.columns]]
                _html_table(df_display)
            else:
                st.warning(f"No events found for plate `{active_search}`")
        else:
            st.info("Enter a plate number above to view historical sightings and timeline.")

    # ──────────────────────── TAB: Analytics ──────────────────────────
    with tab_analytics:
        st.subheader("📊 Traffic Analytics")

        all_events = load_recent_events(limit=500)

        if all_events:
            df_all = prepare_dataframe(all_events)

            if "direction" in df_all.columns:
                col1, col2 = st.columns(2)

                with col1:
                    direction_counts = df_all["direction"].value_counts()
                    fig_direction = px.pie(
                        values=direction_counts.values,
                        names=direction_counts.index,
                        title="Traffic by Direction",
                        color_discrete_map={"IN": "#28a745", "OUT": "#dc3545"},
                    )
                    st.plotly_chart(fig_direction, use_container_width=True)

                with col2:
                    if "vehicle_type" in df_all.columns:
                        vehicle_counts = df_all["vehicle_type"].value_counts().head(10)
                        fig_vehicle = px.bar(
                            x=vehicle_counts.index,
                            y=vehicle_counts.values,
                            title="Top Vehicle Types",
                            labels={"x": "Vehicle Type", "y": "Count"},
                        )
                        st.plotly_chart(fig_vehicle, use_container_width=True)

            if "timestamp" in df_all.columns:
                df_all["hour"] = pd.to_datetime(df_all["timestamp"]).dt.hour
                hourly_counts = df_all.groupby("hour").size()

                fig_timeline = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Events by Hour of Day",
                    labels={"x": "Hour", "y": "Event Count"},
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            if "plate_color" in df_all.columns:
                color_counts = df_all["plate_color"].value_counts()
                fig_color = px.pie(
                    values=color_counts.values,
                    names=color_counts.index,
                    title="Plate Colors (Registration Category)",
                )
                st.plotly_chart(fig_color, use_container_width=True)
        else:
            st.info("No data available for analysis.")

    # ──────────────────────── TAB: Vehicle History ────────────────────
    with tab_history:
        st.subheader("🚗 Vehicle History Profile")

        plate_input = st.text_input("Enter plate number", "", placeholder="DL3CBJ1384", key="history_plate").upper()

        if plate_input:
            history = search_plate_history(plate_input)

            if history:
                classification = classify_vehicle(plate_input)
                st.markdown(
                    f"### Profile: **{plate_input}** &nbsp; "
                    f"<span class='{classification['badge_class']}'>{classification['badge_text']}</span>",
                    unsafe_allow_html=True,
                )

                df_history = prepare_dataframe(history)

                col1, col2, col3, col4 = st.columns(4)
                first_seen = df_history["timestamp"].min() if "timestamp" in df_history.columns else None
                last_seen = df_history["timestamp"].max() if "timestamp" in df_history.columns else None
                entries_count = sum(1 for h in history if h.get("direction") == "IN")
                exits_count = sum(1 for h in history if h.get("direction") == "OUT")

                with col1:
                    st.metric("First Seen", first_seen.strftime("%Y-%m-%d %H:%M") if first_seen else "N/A")
                with col2:
                    st.metric("Last Seen", last_seen.strftime("%Y-%m-%d %H:%M") if last_seen else "N/A")
                with col3:
                    st.metric("Entries", entries_count)
                with col4:
                    st.metric("Exits", exits_count)

                st.divider()

                vehicle_info = history[0] if history else {}
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if vehicle_info.get("vehicle_class"):
                        st.write(f"**Type:** {vehicle_info['vehicle_class'].title()} ({vehicle_info.get('vehicle_type', 'N/A')})")
                    else:
                        st.write(f"**Type:** {vehicle_info.get('vehicle_type', 'N/A')}")
                with col2:
                    if vehicle_info.get("vehicle_color"):
                        st.write(f"**Color:** {vehicle_info['vehicle_color']} (plate {vehicle_info.get('plate_color', 'N/A')})")
                    else:
                        st.write(f"**Plate Color:** {vehicle_info.get('plate_color', 'N/A')}")
                with col3:
                    st.write(f"**Series:** {vehicle_info.get('series_type', 'N/A')}")
                with col4:
                    st.write(f"**Total Sightings:** {len(history)}")

                st.divider()

                st.write("**Complete Event Log**")
                display_cols = ["direction", "vehicle_type", "camera_id", "camera_name", "timestamp"]
                present_cols = [col for col in display_cols if col in df_history.columns]
                if present_cols:
                    _html_table(df_history[present_cols])
            else:
                st.warning(f"No history found for {plate_input}")
        else:
            st.info("Enter a plate number to view vehicle history.")

    # ──────────────────────── TAB: Access Control ─────────────────────
    with tab_access:
        st.subheader("🛡️ Campus Access Control & Gate Authorization")
        st.markdown("Manage Authorized (Whitelisted) and Flagged (Blacklisted) vehicles for campus entry.")

        col_wl, col_bl = st.columns(2)

        # ── Whitelist Column ──
        with col_wl:
            st.markdown("### ✅ Campus Whitelist (Authorized)")
            wl = get_whitelist()
            wl_entries = wl.entries()

            st.write(f"**Total Authorized Vehicles:** {len(wl_entries)}")

            with st.expander("➕ Add Vehicle to Whitelist", expanded=False):
                with st.form("form_add_whitelist", clear_on_submit=True):
                    add_wl_plate = st.text_input("Registration Plate", placeholder="e.g. DL01AB1234").strip().upper()
                    add_wl_owner = st.text_input("Owner / Affiliation", placeholder="e.g. Dr. Sharma / CSE Dept")
                    add_wl_cat = st.selectbox("Category", ["Faculty", "Staff", "Student", "Campus Transit", "Official Delivery", "Authorized Guest"])
                    submit_wl = st.form_submit_button("Authorize Plate")
                    if submit_wl:
                        if add_wl_plate:
                            try:
                                wl.add(add_wl_plate, owner=add_wl_owner, category=add_wl_cat)
                                st.success(f"Plate `{add_wl_plate}` added to Whitelist!")
                                st.rerun()
                            except ValueError as err:
                                st.error(str(err))
                        else:
                            st.error("Please enter a valid registration plate.")

            if wl_entries:
                df_wl = pd.DataFrame(wl_entries)
                _html_table(df_wl)

                # Remove from whitelist
                st.write("**Remove from Whitelist**")
                remove_wl_plate = st.selectbox("Select plate to remove", [e["plate"] for e in wl_entries], key="sel_rm_wl")
                if st.button("Remove Selected Plate", key="btn_rm_wl"):
                    if wl.remove(remove_wl_plate):
                        st.success(f"Removed `{remove_wl_plate}` from Whitelist.")
                        st.rerun()
            else:
                st.info("No vehicles currently whitelisted.")

        # ── Blacklist Column ──
        with col_bl:
            st.markdown("### ⛔ Security Blacklist (Alert On Sight)")
            bl_entries = bl.entries()

            st.write(f"**Total Blacklisted Vehicles:** {len(bl_entries)}")

            with st.expander("➕ Add Vehicle to Blacklist", expanded=False):
                with st.form("form_add_blacklist", clear_on_submit=True):
                    add_bl_plate = st.text_input("Registration Plate", placeholder="e.g. HR26DK9999").strip().upper()
                    add_bl_reason = st.text_input("Flag Reason", placeholder="e.g. Stolen vehicle / Unauthorized entry")
                    submit_bl = st.form_submit_button("Flag Plate")
                    if submit_bl:
                        if add_bl_plate:
                            try:
                                bl.add(add_bl_plate, reason=add_bl_reason)
                                st.success(f"Plate `{add_bl_plate}` added to Blacklist!")
                                st.rerun()
                            except ValueError as err:
                                st.error(str(err))
                        else:
                            st.error("Please enter a valid registration plate.")

            if bl_entries:
                df_bl = pd.DataFrame(bl_entries)
                _html_table(df_bl)

                # Remove from blacklist
                st.write("**Remove from Blacklist**")
                remove_bl_plate = st.selectbox("Select plate to remove", [e["plate"] for e in bl_entries], key="sel_rm_bl")
                if st.button("Remove Selected Flag", key="btn_rm_bl"):
                    if bl.remove(remove_bl_plate):
                        st.success(f"Removed `{remove_bl_plate}` from Blacklist.")
                        st.rerun()
            else:
                st.info("No vehicles currently blacklisted.")

    # ──────────────────────── TAB: Settings ───────────────────────────
    with tab_settings:
        st.subheader("⚙️ System Settings & Configuration")

        st.write("**Database Configuration**")
        db_type = database.get_db_type() if hasattr(database, "get_db_type") else "unknown"
        st.info(f"Database Backend: **{db_type.upper()}** (Path: `data/alpr.db`)")

        st.write("**Detection & Model Ensemble**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Vehicle Confidence: `{config.get('detection', {}).get('vehicle_confidence', 0.5)}`")
            st.write(f"Plate Confidence: `{config.get('detection', {}).get('plate_confidence', 0.4)}`")
        with col2:
            ensemble_cfg = config.get("detection", {}).get("plate_ensemble", {})
            st.write(f"Ensemble Fusion: `{'Enabled' if ensemble_cfg.get('enabled') else 'Disabled'}`")
            st.write(f"Strategy: `{ensemble_cfg.get('fusion_strategy', 'weighted_box')}`")

        st.write("**OCR Configuration**")
        ocr_backend = config.get("ocr", {}).get("backend", "rapidocr")
        st.write(f"Active OCR Backend: **{ocr_backend.upper()}**")

        st.divider()
        st.subheader("📥 Export Data")

        events_export = load_recent_events(limit=1000)
        if events_export:
            df_export = prepare_dataframe(events_export)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download Full Event Log (CSV)",
                data=csv,
                file_name=f"alpr_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # ── Auto-refresh Trigger (Executed gently at bottom) ──────────────
    if auto_refresh and not os.getenv("STREAMLIT_TESTING"):
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
