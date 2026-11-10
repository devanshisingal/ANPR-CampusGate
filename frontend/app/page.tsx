"use client";

import {
  Activity,
  ArrowDownLeft,
  ArrowUpRight,
  Bell,
  Camera,
  CarFront,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Database,
  FileVideo2,
  Gauge,
  LayoutDashboard,
  LoaderCircle,
  MapPinned,
  Plus,
  Radio,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
  Upload,
  Users,
  X,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type Direction = "IN" | "OUT" | "UNKNOWN";
type AccessStatus = "Authorized" | "Blacklisted" | "Unknown";
type EventRecord = {
  id: number;
  plate_number: string;
  vehicle_type: string;
  plate_color: string;
  direction: Direction;
  timestamp: string;
  image_path?: string;
  vehicle_thumbnail_path?: string;
  plate_thumbnail_path?: string;
  camera_id?: string;
  camera_name?: string;
  access_status?: AccessStatus;
  confidence?: number;
};
type Camera = {
  camera_id: string;
  camera_name: string;
  latitude?: number | null;
  longitude?: number | null;
  order: number;
  source_type: string;
  video_path?: string | null;
  enabled: boolean;
};
type Feed = {
  camera_id: string;
  camera_name: string;
  online: boolean;
  enabled: boolean;
  last_frame_at?: string | null;
  frame_age_seconds?: number | null;
  processing_state: string;
  stream_url: string;
};
type Stats = {
  entries: number;
  exits: number;
  unique_vehicles: number;
  total_events: number;
};
type ListEntry = {
  plate: string;
  owner?: string;
  category?: string;
  reason?: string;
  added?: string;
};
type VehicleProfile = {
  plate_number: string;
  vehicle_class?: string;
  vehicle_color?: string;
  vehicle_type?: string;
  vehicle_thumbnail_path?: string;
  plate_thumbnail_path?: string;
  first_seen: string;
  last_seen: string;
  total_detections: number;
  trajectory_history: Array<{
    camera_id?: string;
    camera_name?: string;
    first_seen?: string;
    last_seen?: string;
    detections: number;
  }>;
};
type Trajectory = {
  plate_number: string;
  cameras_visited: number;
  total_detections: number;
  total_distance_km?: number | null;
  points: Array<{
    camera_id: string;
    camera_name: string;
    timestamp?: string | null;
    direction?: string | null;
    latitude?: number | null;
    longitude?: number | null;
  }>;
};
type ProcessingStatus = {
  state?: string;
  session_id?: string | null;
  cameras?: Array<{ camera_id: string; state: string }>;
};
type Tab = "Overview" | "Cameras" | "Activity" | "Access control" | "Vehicle history" | "Settings";

const tabs: Array<{ name: Tab; icon: typeof LayoutDashboard }> = [
  { name: "Overview", icon: LayoutDashboard },
  { name: "Cameras", icon: Camera },
  { name: "Activity", icon: Activity },
  { name: "Access control", icon: ShieldCheck },
  { name: "Vehicle history", icon: MapPinned },
  { name: "Settings", icon: SlidersHorizontal },
];

function localDate(date: Date) {
  return new Date(date.getTime() - date.getTimezoneOffset() * 60_000).toISOString().slice(0, 10);
}

function formatTime(value?: string | null) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function mediaUrl(path?: string | null, kind: "media" | "thumbnails" = "media") {
  if (!path) return "";
  const cleanPath = path.replace(/\\/g, "/");
  if (kind === "thumbnails") {
    const match = cleanPath.match(/thumbnails\/(vehicles|plates)\/([^/]+)$/);
    if (match) return `/backend/thumbnails/${match[1]}/${encodeURIComponent(match[2])}`;
  }
  const filename = cleanPath.split("/").pop();
  return filename ? `/backend/media/${encodeURIComponent(filename)}` : "";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`/backend${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload && typeof payload.detail === "string" ? payload.detail : response.statusText;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return payload as T;
}

function accessClass(status?: string) {
  return (status || "Unknown").toLowerCase();
}

function App() {
  const [activeTab, setActiveTab] = useState<Tab>("Overview");
  const [startDate, setStartDate] = useState(localDate(new Date()));
  const [endDate, setEndDate] = useState(localDate(new Date()));
  const [direction, setDirection] = useState("ALL");
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [stats, setStats] = useState<Stats>({ entries: 0, exits: 0, unique_vehicles: 0, total_events: 0 });
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [feeds, setFeeds] = useState<Feed[]>([]);
  const [processing, setProcessing] = useState<ProcessingStatus>({});
  const [systemSettings, setSystemSettings] = useState<Record<string, unknown>>({});
  const [whitelist, setWhitelist] = useState<ListEntry[]>([]);
  const [blacklist, setBlacklist] = useState<ListEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiConnected, setApiConnected] = useState(false);
  const [databaseConnected, setDatabaseConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [searchText, setSearchText] = useState("");
  const [vehicleQuery, setVehicleQuery] = useState("");
  const [vehicle, setVehicle] = useState<VehicleProfile | null>(null);
  const [trajectory, setTrajectory] = useState<Trajectory | null>(null);
  const [selectedAccess, setSelectedAccess] = useState<"whitelist" | "blacklist">("whitelist");
  const [accessPlate, setAccessPlate] = useState("");
  const [accessDetails, setAccessDetails] = useState("");
  const [accessCategory, setAccessCategory] = useState("Authorized");

  const dateQuery = useMemo(() => {
    const params = new URLSearchParams();
    if (startDate) params.set("start_date", startDate);
    if (endDate) params.set("end_date", endDate);
    if (!startDate && !endDate) params.set("all_time", "true");
    if (direction !== "ALL") params.set("direction", direction);
    return params.toString();
  }, [startDate, endDate, direction]);

  const refresh = useCallback(async () => {
    const query = dateQuery ? `?${dateQuery}` : "";
    const plateSearch = searchText.trim();
    const recordsPath = plateSearch
      ? `/search?plate=${encodeURIComponent(plateSearch)}${query ? `&${dateQuery}` : ""}`
      : `/logs?limit=250${query ? `&${dateQuery}` : ""}`;
    const [nextEvents, nextStats, nextFeeds, nextCameras, nextProcessing] = await Promise.all([
      request<EventRecord[]>(recordsPath),
      request<Stats>(`/stats${query}`),
      request<Feed[]>("/trajectory-api/cameras/feeds"),
      request<Camera[]>("/trajectory-api/cameras"),
      request<ProcessingStatus>("/trajectory-api/processing/status"),
    ]);
    setEvents(nextEvents);
    setStats(nextStats);
    setFeeds(nextFeeds);
    setCameras(nextCameras);
    setProcessing(nextProcessing);
    setApiConnected(true);
    setDatabaseConnected(true);
    setError("");
    setLoading(false);
  }, [dateQuery, searchText]);

  const refreshAccessLists = useCallback(async () => {
    const [authorized, blocked] = await Promise.all([
      request<ListEntry[]>("/access-control/whitelist"),
      request<ListEntry[]>("/access-control/blacklist"),
    ]);
    setWhitelist(authorized);
    setBlacklist(blocked);
  }, []);

  const loadSettings = useCallback(async () => {
    setError("");
    try {
      setSystemSettings(await request<Record<string, unknown>>("/settings"));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load system settings");
    }
  }, []);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        await Promise.all([refresh(), refreshAccessLists()]);
      } catch (reason) {
        if (active) {
          setApiConnected(false);
          setDatabaseConnected(false);
          setError(reason instanceof Error ? reason.message : "Could not load dashboard data");
          setLoading(false);
        }
      }
    };
    void load();
    const timer = window.setInterval(() => {
      void refresh().catch((reason: unknown) => {
        if (active) {
          setApiConnected(false);
          setDatabaseConnected(false);
          setError(reason instanceof Error ? reason.message : "Dashboard refresh failed");
        }
      });
    }, 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [refresh, refreshAccessLists]);

  useEffect(() => {
    if (activeTab === "Settings") void loadSettings();
  }, [activeTab, loadSettings]);

  const filteredEvents = useMemo(() => {
    const needle = searchText.trim().toUpperCase();
    return needle ? events.filter((event) => event.plate_number.toUpperCase().includes(needle)) : events;
  }, [events, searchText]);
  const onlineCount = feeds.filter((feed) => feed.online).length;
  const blacklistedDetections = events.filter((event) => event.access_status === "Blacklisted");

  function downloadCsv() {
    const columns = ["plate_number", "access_status", "direction", "vehicle_type", "plate_color", "camera_id", "camera_name", "timestamp"];
    const quote = (value: unknown) => {
      const text = String(value ?? "");
      const safe = /^[=+\-@]/.test(text) ? `'${text}` : text;
      return `"${safe.replace(/"/g, '""')}"`;
    };
    const content = [columns.join(","), ...filteredEvents.map((event) => columns.map((column) => quote(event[column as keyof EventRecord])).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob([content], { type: "text/csv;charset=utf-8" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `campusgate-events-${localDate(new Date())}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function runAction(action: () => Promise<unknown>, successMessage: string) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
      setNotice(successMessage);
      await Promise.all([refresh(), refreshAccessLists()]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The operation failed");
    } finally {
      setBusy(false);
    }
  }

  async function submitAccessEntry(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const path = `/access-control/${selectedAccess}`;
    const body = selectedAccess === "whitelist"
      ? { plate: accessPlate, owner: accessDetails, category: accessCategory }
      : { plate: accessPlate, reason: accessDetails };
    await runAction(
      () => request(path, { method: "POST", body: JSON.stringify(body) }),
      `${accessPlate.toUpperCase()} added to the ${selectedAccess}.`,
    );
    setAccessPlate("");
    setAccessDetails("");
  }

  async function removeAccessEntry(list: "whitelist" | "blacklist", plate: string) {
    await runAction(
      () => request(`/access-control/${list}/${encodeURIComponent(plate)}`, { method: "DELETE" }),
      `${plate} removed from the ${list}.`,
    );
  }

  async function searchVehicle(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const plate = vehicleQuery.trim().toUpperCase();
    if (!plate) return;
    setBusy(true);
    setError("");
    setVehicle(null);
    setTrajectory(null);
    try {
      const [profile, path] = await Promise.all([
        request<VehicleProfile>(`/trajectory-api/vehicles/${encodeURIComponent(plate)}`),
        request<Trajectory>(`/trajectory-api/trajectory/${encodeURIComponent(plate)}`),
      ]);
      setVehicle(profile);
      setTrajectory(path);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load vehicle history");
    } finally {
      setBusy(false);
    }
  }

  async function startProcessing() {
    await runAction(
      () => request("/trajectory-api/processing/start", { method: "POST", body: JSON.stringify({}) }),
      "Processing started for enabled cameras.",
    );
  }

  async function stopProcessing() {
    await runAction(
      () => request("/trajectory-api/processing/stop", { method: "POST" }),
      "Processing stop requested.",
    );
  }

  function navigate(tab: Tab) {
    setActiveTab(tab);
    setError("");
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><span /></div>
          <div><strong>Campus<span>Gate</span></strong><small>ANPR OPERATIONS</small></div>
        </div>
        <div className="campus-select">
          <div className="campus-icon"><MapPinned size={17} /></div>
          <div><small>MONITORING SITE</small><strong>University Campus</strong></div>
          <ChevronDown size={15} />
        </div>
        <p className="nav-label">WORKSPACE</p>
        <nav className="nav-list" aria-label="Main navigation">
          {tabs.map(({ name, icon: Icon }) => (
            <button key={name} className={`nav-item ${activeTab === name ? "selected" : ""}`} onClick={() => navigate(name)}>
              <Icon size={18} strokeWidth={1.8} /><span>{name}</span>
              {name === "Access control" && blacklistedDetections.length > 0 && (
                <i className="nav-count">{blacklistedDetections.length}</i>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="system-card">
          <div className="system-heading"><span className="pulse-dot" /> SYSTEM STATUS</div>
          <div className="system-row"><span>API connection</span><strong className={apiConnected ? "online-text" : "offline-text"}>{apiConnected ? "Connected" : "Unavailable"}</strong></div>
          <div className="system-row"><span>Camera feeds</span><strong>{onlineCount}/{feeds.length} online</strong></div>
          <div className="system-row"><span>Database</span><strong className={databaseConnected ? "online-text" : "offline-text"}><Database size={13} />{databaseConnected ? "Connected" : "Unavailable"}</strong></div>
        </div>
        <div className="user-card"><div className="avatar">CG</div><div><strong>Gate Operations</strong><small>Campus security</small></div><CircleHelp size={17} /></div>
      </aside>

      <section className="main-column">
        <header className="topbar">
          <div className="breadcrumb"><span>CampusGate</span><b>/</b><strong>{activeTab}</strong></div>
          <div className="topbar-actions">
            <div className="live-indicator"><span className={apiConnected ? "pulse-dot" : "offline-dot"} />{apiConnected ? "API CONNECTED" : "API OFFLINE"}</div>
            <button className="icon-button" aria-label="Refresh data" onClick={() => void refresh()}><RefreshCw size={17} /></button>
            <button className="icon-button notification-button" aria-label="Blacklisted vehicle alerts" onClick={() => navigate("Access control")}>
              <Bell size={17} />{blacklistedDetections.length > 0 && <i />}
            </button>
            <div className="top-avatar">CG</div>
          </div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> CAMPUS SECURITY · VEHICLE INTELLIGENCE</div>
              <h1>{activeTab === "Overview" ? "Operations overview" : activeTab}</h1>
              <p>{activeTab === "Overview" ? "Monitor gate activity, camera health, and vehicle access in real time." : pageDescription(activeTab)}</p>
            </div>
            <div className="heading-actions">
              <button className="button button-muted" onClick={() => void refresh()}><RefreshCw size={15} /> Refresh</button>
              {processing.state === "running"
                ? <button className="button button-danger" disabled={busy} onClick={() => void stopProcessing()}><span className="button-live" /> Stop processing</button>
                : <button className="button button-primary" disabled={busy || cameras.length === 0} onClick={() => void startProcessing()}><Radio size={16} /> Start processing</button>}
            </div>
          </div>

          {error && <div className="alert-banner error-banner"><ShieldAlert size={18} /><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError("")}><X size={16} /></button></div>}
          {notice && <div className="alert-banner notice-banner"><Check size={18} /><span>{notice}</span><button aria-label="Dismiss notice" onClick={() => setNotice("")}><X size={16} /></button></div>}
          {!loading && feeds.length === 0 && !error && <div className="alert-banner info-banner"><CircleHelp size={18} /><span>No cameras are configured. Open Camera management to add a site and its surveyed coordinates.</span></div>}

          {(activeTab === "Overview" || activeTab === "Cameras") && (
            <section className="panel camera-panel">
              <div className="section-heading">
                <div><h2>{activeTab === "Overview" ? "Live camera wall" : "Camera management"}</h2><p>Live previews and connection health for each configured gate.</p></div>
                <div className="section-tools"><span className={`health-pill ${onlineCount ? "healthy" : "unhealthy"}`}><span />{onlineCount} of {feeds.length} online</span></div>
              </div>
              <div className={`camera-grid ${feeds.length === 1 ? "single-camera" : ""}`}>
                {feeds.map((feed) => {
                  const camera = cameras.find((item) => item.camera_id === feed.camera_id);
                  return <CameraTile key={feed.camera_id} feed={feed} camera={camera} disabled={busy || processing.state === "running"} onUpload={(file) => {
                    const form = new FormData();
                    form.set("file", file);
                    void runAction(() => request(`/trajectory-api/cameras/${encodeURIComponent(feed.camera_id)}/upload`, { method: "POST", body: form }), `${file.name} uploaded for ${feed.camera_name}.`);
                  }} onSetRtsp={(url) => void runAction(() => request(`/trajectory-api/cameras/${encodeURIComponent(feed.camera_id)}/rtsp`, { method: "POST", body: JSON.stringify({ rtsp_url: url }) }), `RTSP source set for ${feed.camera_name}.`)}
                  onRename={(name) => void runAction(() => request(`/trajectory-api/cameras/${encodeURIComponent(feed.camera_id)}`, { method: "PATCH", body: JSON.stringify({ camera_name: name }) }), `Camera renamed to ${name}.`)}
                  onRemove={() => { if (window.confirm(`Remove ${feed.camera_name}? Existing event history will be retained.`)) void runAction(() => request(`/trajectory-api/cameras/${encodeURIComponent(feed.camera_id)}`, { method: "DELETE" }), `${feed.camera_name} removed.`); }} />;
                })}
                {!feeds.length && !loading && <div className="empty-state camera-empty"><Camera size={24} /><strong>No cameras configured</strong><span>Configure camera locations in config/camera_config.yaml.</span></div>}
              </div>
              {activeTab === "Cameras" && <>
                <form className="new-camera-form" onSubmit={(event) => {
                  event.preventDefault();
                  const form = new FormData(event.currentTarget);
                  const cameraName = String(form.get("camera_name") || "").trim();
                  void runAction(() => request("/trajectory-api/cameras", { method: "POST", body: JSON.stringify({
                    camera_name: cameraName,
                    latitude: Number(form.get("latitude")),
                    longitude: Number(form.get("longitude")),
                    ...(String(form.get("camera_id") || "").trim() ? { camera_id: String(form.get("camera_id")).trim() } : {}),
                  }) }), `${cameraName} added to the camera network.`);
                  event.currentTarget.reset();
                }}>
                  <div><strong>Add a camera site</strong><span>Coordinates are required for location and trajectory mapping.</span></div>
                  <input name="camera_name" required placeholder="Camera name" aria-label="New camera name" />
                  <input name="camera_id" placeholder="ID (optional)" aria-label="New camera ID" />
                  <input name="latitude" type="number" step="any" required placeholder="Latitude" aria-label="New camera latitude" />
                  <input name="longitude" type="number" step="any" required placeholder="Longitude" aria-label="New camera longitude" />
                  <button className="button button-primary" disabled={busy || processing.state === "running"}><Plus size={14} /> Add camera</button>
                </form>
                <div className="camera-footnote"><FileVideo2 size={15} /> Upload a recording or configure an RTSP source for each camera. Stream credentials are not displayed in the dashboard.</div>
              </>}
            </section>
          )}

          {(activeTab === "Overview" || activeTab === "Activity") && (
            <>
              <div className="filter-bar">
                <div className="filter-title"><Gauge size={16} /><strong>Traffic analytics</strong><span>Filtered to selected period and direction</span></div>
                <div className="filters">
                  <label><span>FROM</span><input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} aria-label="Start date" /></label>
                  <label><span>TO</span><input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} aria-label="End date" /></label>
                  <label><span>DIRECTION</span><select value={direction} onChange={(event) => setDirection(event.target.value)}><option value="ALL">All traffic</option><option value="IN">Entries</option><option value="OUT">Exits</option><option value="UNKNOWN">Unknown</option></select></label>
                  <button className="clear-filter" onClick={() => { setStartDate(""); setEndDate(""); setDirection("ALL"); }}>Clear</button>
                </div>
              </div>
              <div className="stats-grid">
                <StatCard label="VEHICLES ENTERED" value={loading ? null : stats.entries} icon={ArrowDownLeft} tone="green" footnote="IN direction events" />
                <StatCard label="VEHICLES EXITED" value={loading ? null : stats.exits} icon={ArrowUpRight} tone="blue" footnote="OUT direction events" />
                <StatCard label="UNIQUE VEHICLES" value={loading ? null : stats.unique_vehicles} icon={CarFront} tone="purple" footnote="Distinct plates in range" />
                <StatCard label="TOTAL RECOGNITIONS" value={loading ? null : stats.total_events} icon={Activity} tone="amber" footnote="Persisted crossing events" />
              </div>
              <section className="panel activity-panel">
                <div className="section-heading activity-heading">
                  <div><h2>{activeTab === "Overview" ? "Recent recognitions" : "Recognition activity"}</h2><p>Latest persisted plate reads, access status, and direction.</p></div>
                  <div className="activity-tools">
                    <label className="search-field"><Search size={16} /><input value={searchText} onChange={(event) => setSearchText(event.target.value)} placeholder="Search plate number" /></label>
                    <button className="button button-small button-muted export-button" onClick={downloadCsv}><Database size={13} /> Export CSV</button>
                    <span className="result-count">{filteredEvents.length} records</span>
                  </div>
                </div>
                <EventsTable events={filteredEvents.slice(0, activeTab === "Overview" ? 8 : 250)} />
                {activeTab === "Overview" && <button className="text-link" onClick={() => navigate("Activity")}>View all activity <ArrowUpRight size={14} /></button>}
              </section>
            </>
          )}

          {activeTab === "Overview" && (
            <div className="bottom-grid">
              <section className="panel compact-panel">
                <div className="section-heading"><div><h2>Blacklisted vehicle alerts</h2><p>Blacklisted plates recognized in the selected range.</p></div><span className="alert-count">{blacklistedDetections.length}</span></div>
                {blacklistedDetections.length
                  ? <div className="alert-list">{blacklistedDetections.slice(0, 4).map((event) => <div className="alert-row" key={event.id}><div className="alert-icon"><ShieldAlert size={16} /></div><div className="alert-main"><strong>{event.plate_number}</strong><span>{event.camera_name || event.camera_id || "Camera"} · {formatTime(event.timestamp)}</span></div><span className={`direction-tag ${event.direction.toLowerCase()}`}>{event.direction}</span></div>)}</div>
                  : <div className="empty-state small-empty"><ShieldCheck size={21} /><span>No blacklist matches in this period.</span></div>}
              </section>
              <section className="panel compact-panel">
                <div className="section-heading"><div><h2>Camera health</h2><p>Connection state uses backend frame freshness.</p></div><Camera size={17} className="muted-icon" /></div>
                <div className="health-list">{feeds.map((feed) => <div className="health-row" key={feed.camera_id}><span className={`health-bullet ${feed.online ? "online" : "offline"}`} /><div><strong>{feed.camera_name}</strong><small>{feed.camera_id} · {feed.online ? `Frame ${Math.round(feed.frame_age_seconds ?? 0)}s ago` : "No fresh frame"}</small></div><span className={`health-label ${feed.online ? "online-text" : "offline-text"}`}>{feed.online ? "ONLINE" : "OFFLINE"}</span></div>)}{feeds.length === 0 && <div className="empty-state small-empty">No cameras configured.</div>}</div>
              </section>
            </div>
          )}

          {activeTab === "Access control" && (
            <AccessControl
              whitelist={whitelist}
              blacklist={blacklist}
              events={blacklistedDetections}
              selected={selectedAccess}
              setSelected={setSelectedAccess}
              plate={accessPlate}
              setPlate={setAccessPlate}
              details={accessDetails}
              setDetails={setAccessDetails}
              category={accessCategory}
              setCategory={setAccessCategory}
              disabled={busy}
              onSubmit={submitAccessEntry}
              onRemove={removeAccessEntry}
            />
          )}

          {activeTab === "Vehicle history" && (
            <section className="panel history-panel">
              <div className="section-heading"><div><h2>Vehicle journey lookup</h2><p>Search recognized plates for persisted vehicle profile, visit history, and map.</p></div><Users size={18} className="muted-icon" /></div>
              <form className="history-search" onSubmit={(event) => void searchVehicle(event)}>
                <label className="search-field"><Search size={17} /><input value={vehicleQuery} onChange={(event) => setVehicleQuery(event.target.value)} placeholder="Enter registration number (e.g. DL8CAF1234)" /></label>
                <button className="button button-primary" disabled={busy || !vehicleQuery.trim()}>{busy ? <LoaderCircle className="spin" size={16} /> : <Search size={16} />} Search vehicle</button>
              </form>
              {vehicle && trajectory && <VehicleHistory profile={vehicle} trajectory={trajectory} />}
              {!vehicle && <div className="history-placeholder"><div className="history-placeholder-icon"><MapPinned size={22} /></div><strong>Vehicle journey details</strong><span>Enter a plate number to view its profile and recorded camera visits.</span></div>}
            </section>
          )}

          {activeTab === "Settings" && <SettingsPanel settings={systemSettings} processing={processing} onRefresh={() => void loadSettings()} />}

          <footer className="page-footer"><span>CampusGate ANPR</span><span>Real-time operations · Data sourced from the connected system database</span><span>{processing.state === "running" ? "Processing active" : "Monitoring ready"}</span></footer>
        </div>
      </section>
    </main>
  );
}

function pageDescription(tab: Tab) {
  const descriptions: Record<Tab, string> = {
    Overview: "Monitor gate activity, camera health, and vehicle access in real time.",
    Cameras: "Manage camera sources, inspect live feeds, and monitor connection health.",
    Activity: "Review persisted recognitions with date, direction, and plate filters.",
    "Access control": "Manage authorized and blocked vehicles and review matching alerts.",
    "Vehicle history": "Inspect vehicle profiles and reconstructed camera visit timelines.",
    Settings: "Review safe system configuration and export recognition data.",
  };
  return descriptions[tab];
}

function StatCard({ label, value, icon: Icon, tone, footnote }: { label: string; value: number | null; icon: typeof Activity; tone: string; footnote: string }) {
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className={`stat-icon ${tone}`}><Icon size={17} /></span></div><div className="stat-value">{value == null ? "—" : value.toLocaleString()}</div><div className="stat-foot"><span className={`stat-dot ${tone}`} />{footnote}</div></article>;
}

function CameraTile({ feed, camera, disabled, onUpload, onSetRtsp, onRename, onRemove }: { feed: Feed; camera?: Camera; disabled: boolean; onUpload: (file: File) => void; onSetRtsp: (url: string) => void; onRename: (name: string) => void; onRemove: () => void }) {
  const [showSourceForm, setShowSourceForm] = useState(false);
  const [showNameForm, setShowNameForm] = useState(false);
  const [rtsp, setRtsp] = useState("");
  const [cameraName, setCameraName] = useState(feed.camera_name);
  const [imageBroken, setImageBroken] = useState(false);
  const online = feed.online;
  return (
    <article className="camera-card">
      <div className="camera-preview">
        {online && !imageBroken
          ? <img src={`/backend/trajectory-api/cameras/${encodeURIComponent(feed.camera_id)}/stream`} alt={`Live feed from ${feed.camera_name}`} onError={() => setImageBroken(true)} />
          : <div className="feed-placeholder"><div className="feed-placeholder-icon"><Camera size={22} /></div><strong>{online ? "Connecting to camera…" : "Feed unavailable"}</strong><span>{online ? "Waiting for the live preview." : "A fresh camera frame has not been received."}</span></div>}
        <span className="preview-label"><span className={online ? "pulse-dot" : "offline-dot"} />{online ? "LIVE" : "NO SIGNAL"}</span>
        <span className="preview-channel">{feed.camera_id}</span>
      </div>
      <div className="camera-card-body">
        <div className="camera-title-row"><div><h3>{feed.camera_name}</h3><span>{camera?.source_type === "rtsp" ? "RTSP stream" : camera?.source_type === "upload" ? "Video source" : "Camera source not set"}</span></div><span className={`camera-status ${online ? "camera-online" : "camera-offline"}`}><i />{online ? "Online" : "Offline"}</span></div>
        <div className="camera-metadata"><span><Clock3 size={13} />{feed.last_frame_at ? `Last frame ${formatTime(feed.last_frame_at)}` : "No frame received"}</span><span><Activity size={13} />{feed.processing_state || "idle"}</span></div>
        <div className="camera-actions">
          <label className="button button-small button-muted upload-button"><Upload size={14} /> Upload video<input type="file" accept="video/*" disabled={disabled} onChange={(event) => { const file = event.target.files?.[0]; if (file) onUpload(file); event.currentTarget.value = ""; }} /></label>
          <button className="button button-small button-muted" disabled={disabled} onClick={() => setShowSourceForm((current) => !current)}><Radio size={14} /> RTSP source</button>
          <button className="button button-small button-muted" disabled={disabled} onClick={() => { setCameraName(feed.camera_name); setShowNameForm((current) => !current); }}>Rename</button>
          <button className="remove-button camera-remove" aria-label={`Remove ${feed.camera_name}`} disabled={disabled} onClick={onRemove}><Trash2 size={14} /></button>
        </div>
        {showSourceForm && <form className="rtsp-form" onSubmit={(event) => { event.preventDefault(); if (rtsp.trim()) { onSetRtsp(rtsp.trim()); setRtsp(""); setShowSourceForm(false); } }}><input type="url" required placeholder="rtsp://user:password@host/stream" value={rtsp} onChange={(event) => setRtsp(event.target.value)} aria-label={`RTSP URL for ${feed.camera_name}`} /><button className="button button-small button-primary" disabled={disabled}>Save</button></form>}
        {showNameForm && <form className="rtsp-form" onSubmit={(event) => { event.preventDefault(); if (cameraName.trim()) { onRename(cameraName.trim()); setShowNameForm(false); } }}><input required maxLength={100} value={cameraName} onChange={(event) => setCameraName(event.target.value)} aria-label={`New name for ${feed.camera_name}`} /><button className="button button-small button-primary" disabled={disabled}>Save name</button></form>}
      </div>
    </article>
  );
}

function SettingsPanel({ settings, processing, onRefresh }: { settings: Record<string, unknown>; processing: ProcessingStatus; onRefresh: () => void }) {
  const entries = Object.entries(settings).filter(([, value]) => value && typeof value === "object" && !Array.isArray(value));
  const camera = settings.camera as Record<string, unknown> | undefined;
  return <section className="panel settings-panel">
    <div className="section-heading"><div><h2>System configuration</h2><p>Read-only values from the connected FastAPI service; secrets and camera passwords are omitted.</p></div><button className="button button-muted button-small" onClick={onRefresh}><RefreshCw size={13} /> Reload configuration</button></div>
    <div className="settings-summary">
      <div><small>CAMERA ID</small><strong>{String(camera?.camera_id ?? "—")}</strong></div>
      <div><small>CAMERA NAME</small><strong>{String(camera?.camera_name ?? "—")}</strong></div>
      <div><small>PROCESSING STATE</small><strong className={processing.state === "running" ? "online-text" : ""}>{processing.state ?? "idle"}</strong></div>
      <div><small>SESSION ID</small><strong>{processing.session_id || "—"}</strong></div>
    </div>
    <div className="settings-grid">{entries.map(([section, value]) => <article className="settings-card" key={section}><h3>{section.replace(/_/g, " ")}</h3><dl>{Object.entries(value as Record<string, unknown>).map(([key, item]) => <div key={key}><dt>{key.replace(/_/g, " ")}</dt><dd>{typeof item === "object" ? JSON.stringify(item) : String(item ?? "—")}</dd></div>)}</dl></article>)}</div>
    <div className="settings-note"><CircleHelp size={15} /> Settings are read-only here. Update camera/network configuration in the repository config files, then restart the API service.</div>
  </section>;
}

function EventsTable({ events }: { events: EventRecord[] }) {
  if (!events.length) return <div className="empty-state table-empty"><Database size={23} /><strong>No recognition records</strong><span>Events will appear here when the connected pipeline records a vehicle crossing.</span></div>;
  return (
    <div className="table-scroll"><table><thead><tr><th>VEHICLE</th><th>ACCESS</th><th>DIRECTION</th><th>CAMERA</th><th>RECOGNIZED</th></tr></thead><tbody>
      {events.map((event) => {
        const thumb = mediaUrl(event.plate_thumbnail_path, "thumbnails") || mediaUrl(event.image_path);
        return <tr key={event.id}><td><div className="vehicle-cell">{thumb ? <img src={thumb} alt="" loading="lazy" onError={(imageEvent) => { imageEvent.currentTarget.style.visibility = "hidden"; }} /> : <span className="plate-placeholder"><CarFront size={16} /></span>}<div><strong>{event.plate_number}</strong><small>{event.vehicle_type || "Vehicle"}{event.plate_color ? ` · ${event.plate_color}` : ""}</small></div></div></td><td><span className={`access-badge ${accessClass(event.access_status)}`}><i />{event.access_status || "Unknown"}</span></td><td><span className={`direction-tag ${event.direction.toLowerCase()}`}>{event.direction === "IN" ? <ArrowDownLeft size={13} /> : <ArrowUpRight size={13} />}{event.direction}</span></td><td><div className="camera-cell"><strong>{event.camera_name || "—"}</strong><small>{event.camera_id || "Camera not assigned"}</small></div></td><td><span className="timestamp">{formatTime(event.timestamp)}</span></td></tr>;
      })}
    </tbody></table></div>
  );
}

function AccessControl({ whitelist, blacklist, events, selected, setSelected, plate, setPlate, details, setDetails, category, setCategory, disabled, onSubmit, onRemove }: {
  whitelist: ListEntry[];
  blacklist: ListEntry[];
  events: EventRecord[];
  selected: "whitelist" | "blacklist";
  setSelected: (value: "whitelist" | "blacklist") => void;
  plate: string;
  setPlate: (value: string) => void;
  details: string;
  setDetails: (value: string) => void;
  category: string;
  setCategory: (value: string) => void;
  disabled: boolean;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onRemove: (list: "whitelist" | "blacklist", plate: string) => void;
}) {
  const entries = selected === "whitelist" ? whitelist : blacklist;
  return <div className="access-layout">
    <section className="panel access-manager">
      <div className="access-tabs"><button className={selected === "whitelist" ? "active" : ""} onClick={() => setSelected("whitelist")}><ShieldCheck size={16} /> Authorized <span>{whitelist.length}</span></button><button className={selected === "blacklist" ? "active black-tab" : ""} onClick={() => setSelected("blacklist")}><ShieldAlert size={16} /> Blocklist <span>{blacklist.length}</span></button></div>
      <div className="section-heading access-heading"><div><h2>{selected === "whitelist" ? "Authorized vehicles" : "Blacklisted vehicles"}</h2><p>{selected === "whitelist" ? "Plates on the campus authorization list." : "Blocked plates trigger alerts when recognized."}</p></div></div>
      <form className="access-form" onSubmit={(event) => void onSubmit(event)}>
        <label><span>LICENSE PLATE</span><input required minLength={4} maxLength={16} pattern="[A-Za-z0-9 -]+" value={plate} onChange={(event) => setPlate(event.target.value.toUpperCase())} placeholder="e.g. DL8CAF1234" /></label>
        <label><span>{selected === "whitelist" ? "OWNER / DEPARTMENT" : "REASON / NOTES"}</span><input value={details} onChange={(event) => setDetails(event.target.value)} placeholder={selected === "whitelist" ? "Faculty, staff, delivery…" : "Reason for blocking"} /></label>
        {selected === "whitelist" && <label><span>VEHICLE CATEGORY</span><select value={category} onChange={(event) => setCategory(event.target.value)}><option>Authorized</option><option>Student</option><option>Faculty</option><option>Staff</option><option>Delivery</option><option>Other</option></select></label>}
        <button className={`button ${selected === "blacklist" ? "button-danger" : "button-primary"}`} disabled={disabled}><Plus size={16} /> Add to {selected}</button>
      </form>
      <div className="access-list">
        <div className="list-heading"><span>{selected === "whitelist" ? "PLATE & OWNER" : "PLATE & REASON"}</span><span>ADDED</span><span /></div>
        {entries.map((entry) => <div className="access-list-row" key={entry.plate}><div className="access-plate"><span className={`entry-icon ${selected}`}><CarFront size={15} /></span><div><strong>{entry.plate}</strong><small>{selected === "whitelist" ? entry.owner || entry.category || "Authorized" : entry.reason || "No reason provided"}</small></div></div><span className="added-date">{entry.added || "—"}</span><button className="remove-button" aria-label={`Remove ${entry.plate}`} disabled={disabled} onClick={() => void onRemove(selected, entry.plate)}><Trash2 size={15} /></button></div>)}
        {!entries.length && <div className="empty-state small-empty">No vehicles in this list.</div>}
      </div>
    </section>
    <section className="panel alert-panel">
      <div className="section-heading"><div><h2>Recent blacklist detections</h2><p>Matched against recognition records in the current filter range.</p></div><span className="alert-count">{events.length}</span></div>
      {events.length ? <div className="alert-list">{events.slice(0, 12).map((event) => <div className="alert-row" key={event.id}><div className="alert-icon"><ShieldAlert size={16} /></div><div className="alert-main"><strong>{event.plate_number}</strong><span>{event.camera_name || event.camera_id || "Unknown camera"} · {formatTime(event.timestamp)}</span></div><span className={`direction-tag ${event.direction.toLowerCase()}`}>{event.direction}</span></div>)}</div> : <div className="empty-state small-empty"><ShieldCheck size={21} /><span>No blacklist detections in the selected period.</span></div>}
    </section>
  </div>;
}

function VehicleHistory({ profile, trajectory }: { profile: VehicleProfile; trajectory: Trajectory }) {
  const image = mediaUrl(profile.vehicle_thumbnail_path, "thumbnails") || mediaUrl(profile.plate_thumbnail_path, "thumbnails");
  return <div className="vehicle-history">
    <div className="profile-card">{image ? <img src={image} alt={`Vehicle ${profile.plate_number}`} /> : <div className="profile-placeholder"><CarFront size={27} /></div>}<div className="profile-info"><span className="eyebrow">VEHICLE PROFILE</span><h3>{profile.plate_number}</h3><p>{[profile.vehicle_class, profile.vehicle_color, profile.vehicle_type].filter(Boolean).join(" · ") || "No vehicle attributes recorded"}</p></div><div className="profile-stat"><strong>{profile.total_detections}</strong><small>DETECTIONS</small></div><div className="profile-stat"><strong>{profile.trajectory_history.length}</strong><small>CAMERA VISITS</small></div></div>
    <div className="journey-grid"><div className="journey-left">
      <div className="journey-meta"><div><small>FIRST SEEN</small><strong>{formatTime(profile.first_seen)}</strong></div><div><small>LAST SEEN</small><strong>{formatTime(profile.last_seen)}</strong></div><div><small>CAMERAS VISITED</small><strong>{trajectory.cameras_visited}</strong></div><div><small>TRAVEL DISTANCE</small><strong>{trajectory.total_distance_km == null ? "—" : `${trajectory.total_distance_km.toFixed(2)} km`}</strong></div></div>
      <h3 className="subsection-title">Recorded camera visits</h3>
      {trajectory.points.length
        ? <div className="timeline">{trajectory.points.map((point, index) => <div className="timeline-row" key={`${point.camera_id}-${index}`}><div className="timeline-marker"><span>{index + 1}</span></div><div className="timeline-content"><strong>{point.camera_name}</strong><small>{point.camera_id}{point.direction ? ` · ${point.direction}` : ""}</small></div><time>{formatTime(point.timestamp)}</time></div>)}</div>
        : <div className="empty-state small-empty">No camera visits available for this vehicle.</div>}
      </div><div className="journey-map"><div className="map-title"><MapPinned size={15} /> VEHICLE ROUTE MAP</div><iframe title={`Map for ${profile.plate_number}`} src={`/backend/trajectory-api/map/${encodeURIComponent(profile.plate_number)}/html`} loading="lazy" /></div></div>
  </div>;
}

export default App;
