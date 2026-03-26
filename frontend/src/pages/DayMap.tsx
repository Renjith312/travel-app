import React, { useState, useEffect, useRef } from 'react';
import {
  MapContainer, TileLayer, Polyline, Marker, Tooltip, useMap,
} from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { ChevronDown, MapPin } from 'lucide-react';
import type { Activity, RawActivity } from '../types';
import './DayMap.css';

// ── Activity-type colours ─────────────────────────────────────
const TYPE_COLOR: Record<string, string> = {
  ACCOMMODATION: '#6c8eff',
  TRANSPORT:     '#34d399',
  FOOD:          '#fbbf24',
  SIGHTSEEING:   '#a78bfa',
  ACTIVITY:      '#f97316',
  SHOPPING:      '#ec4899',
  RELAXATION:    '#2dd4bf',
  NIGHTLIFE:     '#818cf8',
  OTHER:         '#94a3b8',
};

// ── Coordinate extraction ─────────────────────────────────────
interface MapPoint {
  latlng: [number, number]; // [lat, lng] — Leaflet convention
  label: string;
  type: string;
}

function parseLocation(loc: RawActivity['location']): [number, number] | null {
  if (!loc) return null;
  if (typeof loc === 'object' && 'lat' in loc) {
    const { lat, lon } = loc as { lat: number; lon: number };
    if (lat && lon) return [lat, lon];
  }
  if (typeof loc === 'string') {
    const parts = loc.split(',').map(s => parseFloat(s.trim()));
    if (parts.length === 2 && !parts.some(isNaN) && parts[0] !== 0) {
      return [parts[0], parts[1]]; // "lat,lon"
    }
  }
  return null;
}

function extractPoints(activities: (Activity | RawActivity)[]): MapPoint[] {
  const pts: MapPoint[] = [];
  for (const act of activities) {
    if ('id' in act) {
      const a = act as Activity;
      if (a.latitude && a.longitude) {
        pts.push({ latlng: [a.latitude, a.longitude], label: a.title, type: a.type || 'OTHER' });
      }
    } else {
      const a = act as RawActivity;
      const coords = parseLocation(a.location);
      if (coords) {
        pts.push({ latlng: coords, label: a.name || a.title || 'Stop', type: a.type || 'OTHER' });
      }
    }
  }
  return pts;
}

// ── OSRM road-routing (free, no key) ─────────────────────────
// Batches all stops into one request (OSRM supports up to 100 waypoints)
async function fetchOsrmRoute(pts: MapPoint[]): Promise<[number, number][] | null> {
  if (pts.length < 2) return null;
  // OSRM wants lng,lat pairs
  const coords = pts.map(p => `${p.latlng[1]},${p.latlng[0]}`).join(';');
  const url =
    `https://router.project-osrm.org/route/v1/driving/${coords}` +
    `?overview=full&geometries=geojson`;
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(12000) });
    if (!res.ok) return null;
    const data = await res.json();
    const coords2d: [number, number][] =
      data?.routes?.[0]?.geometry?.coordinates ?? null;
    if (!coords2d) return null;
    // OSRM returns [lng, lat] — flip to Leaflet [lat, lng]
    return coords2d.map(([lng, lat]) => [lat, lng]);
  } catch {
    return null;
  }
}

// ── Auto-fit bounds ───────────────────────────────────────────
function FitBounds({ points }: { points: MapPoint[] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length === 0) return;
    const bounds = L.latLngBounds(points.map(p => p.latlng));
    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 });
  }, [points.length]);
  return null;
}

// ── Custom numbered DivIcon ───────────────────────────────────
function makeIcon(num: number, color: string) {
  return L.divIcon({
    className: '',
    iconSize: [30, 30],
    iconAnchor: [15, 15],
    html: `<div class="map-pin" style="border-color:${color}">
             <span class="map-pin-num">${num}</span>
           </div>`,
  });
}

// ── Main component ────────────────────────────────────────────
interface DayMapProps {
  activities: (Activity | RawActivity)[];
  dayNumber: number;
  dayLabel: string;
}

export default function DayMap({ activities, dayNumber, dayLabel }: DayMapProps) {
  const [open, setOpen]           = useState(true);
  const [routePath, setRoutePath] = useState<[number, number][] | null>(null);
  const [loading, setLoading]     = useState(false);
  const fetchedKey                = useRef<string>('');

  const points = extractPoints(activities);

  // Straight-line fallback positions
  const straightLine = points.map(p => p.latlng);

  // Centre for initial map view (before FitBounds fires)
  const centre: [number, number] = points.length > 0
    ? [
        points.reduce((s, p) => s + p.latlng[0], 0) / points.length,
        points.reduce((s, p) => s + p.latlng[1], 0) / points.length,
      ]
    : [20, 78]; // fallback: centre of India

  useEffect(() => {
    if (points.length < 2) return;
    const key = points.map(p => p.latlng.join(',')).join('|');
    if (fetchedKey.current === key) return;
    fetchedKey.current = key;
    setLoading(true);
    setRoutePath(null);
    fetchOsrmRoute(points)
      .then(path => setRoutePath(path))
      .finally(() => setLoading(false));
  }, [dayNumber]);

  if (points.length < 2) return null;

  return (
    <div className="day-map-wrap">
      {/* Toggle header */}
      <button className="day-map-toggle" onClick={() => setOpen(o => !o)}>
        <MapPin size={13} />
        <span>Route Map · {points.length} stops</span>
        {loading && <span className="day-map-loading">fetching route…</span>}
        <ChevronDown size={13} className={`day-map-chevron${open ? ' open' : ''}`} />
      </button>

      {open && (
        <div className="day-map-container">
          <MapContainer
            center={centre}
            zoom={11}
            style={{ width: '100%', height: 320 }}
            zoomControl={true}
            attributionControl={false}
          >
            {/* CartoDB Dark Matter — free, no API key */}
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>'
            />

            {/* Auto-fit the map to all stops */}
            <FitBounds points={points} />

            {/* Road route from OSRM */}
            {routePath && (
              <>
                {/* Casing (dark outline) */}
                <Polyline
                  positions={routePath}
                  pathOptions={{ color: '#1e293b', weight: 7, opacity: 0.6 }}
                />
                {/* Route fill */}
                <Polyline
                  positions={routePath}
                  pathOptions={{ color: '#6c8eff', weight: 4, opacity: 0.95 }}
                />
              </>
            )}

            {/* Dashed straight-line while loading or on API failure */}
            {!routePath && (
              <Polyline
                positions={straightLine}
                pathOptions={{ color: '#6c8eff', weight: 2, opacity: 0.7, dashArray: '8 6' }}
              />
            )}

            {/* Numbered markers */}
            {points.map((pt, i) => (
              <Marker
                key={i}
                position={pt.latlng}
                icon={makeIcon(i + 1, TYPE_COLOR[pt.type] || '#94a3b8')}
              >
                <Tooltip direction="top" offset={[0, -14]} opacity={1}>
                  <span className="map-tooltip-text">{pt.label}</span>
                </Tooltip>
              </Marker>
            ))}
          </MapContainer>

          {/* Legend */}
          <div className="day-map-legend">
            {points.map((pt, i) => (
              <div key={i} className="day-map-legend-item">
                <span
                  className="day-map-legend-dot"
                  style={{ background: TYPE_COLOR[pt.type] || '#94a3b8' }}
                >
                  {i + 1}
                </span>
                <span className="day-map-legend-label">{pt.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
