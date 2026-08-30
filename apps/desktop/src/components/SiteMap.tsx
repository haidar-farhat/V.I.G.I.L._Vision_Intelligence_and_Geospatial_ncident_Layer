import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { Snapshot } from '../data/types.ts';

/**
 * The site map.
 *
 * MapLibre with a **local** style. There is no basemap here because no offline map
 * package is installed, and the component says so rather than quietly reaching for
 * an online tile server. That is the whole point: the honest failure is a blank
 * ground with the site geometry drawn on it, plus a notice explaining what is
 * missing and how to fix it.
 *
 * Camera positions, field-of-view footprints, zones and event markers all render
 * from the pipeline's own GeoJSON, so what the operator sees is what the engine
 * computed - including the uncertainty attached to each event position.
 */

type Props = {
  readonly snapshot: Snapshot;
  readonly selectedIncidentId: string | null;
};

/** A style with no remote sources at all. Valid, offline, and deliberately bare. */
const EMPTY_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  // Vector glyphs would be needed for labels; without a local glyph pack we draw
  // no text rather than pulling a font from the network.
  sources: {},
  layers: [
    {
      id: 'ground',
      type: 'background',
      paint: { 'background-color': '#0d1117' },
    },
  ],
};

const SEVERITY_COLOUR: Record<string, string> = {
  LOW: '#4a8fd4',
  MEDIUM: '#d9a441',
  HIGH: '#e2703a',
  CRITICAL: '#e5484d',
};

export const SiteMap = ({ snapshot, selectedIncidentId }: Props): JSX.Element => {
  const container = useRef<HTMLDivElement | null>(null);
  const map = useRef<maplibregl.Map | null>(null);

  useEffect(() => {
    const element = container.current;
    if (element === null || map.current !== null) return;

    const instance = new maplibregl.Map({
      container: element,
      style: EMPTY_STYLE,
      center: [snapshot.cameras[0]?.position.lon ?? 0, snapshot.cameras[0]?.position.lat ?? 0],
      zoom: 16,
      attributionControl: false,
      // Nothing to collect, nothing to send.
      trackResize: true,
    });

    instance.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-right');
    instance.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');

    instance.on('load', () => {
      instance.addSource('zones', { type: 'geojson', data: snapshot.geojson.zones });
      instance.addSource('fov', { type: 'geojson', data: snapshot.geojson.fov });
      instance.addSource('cameras', { type: 'geojson', data: snapshot.geojson.cameras });
      instance.addSource('events', { type: 'geojson', data: snapshot.geojson.events });

      // --- field of view -------------------------------------------------
      // Drawn first, and drawn as the real annular footprint: a tilted camera
      // cannot see the ground at its own mast, and showing coverage it does not
      // have is exactly the assumption that gets a site burgled.
      instance.addLayer({
        id: 'fov-fill',
        type: 'fill',
        source: 'fov',
        paint: { 'fill-color': '#35c2f5', 'fill-opacity': 0.07 },
      });
      instance.addLayer({
        id: 'fov-line',
        type: 'line',
        source: 'fov',
        paint: { 'line-color': '#35c2f5', 'line-opacity': 0.28, 'line-width': 1 },
      });

      // --- zones -----------------------------------------------------------
      instance.addLayer({
        id: 'zone-fill',
        type: 'fill',
        source: 'zones',
        paint: {
          'fill-color': [
            'match',
            ['get', 'purpose'],
            'RESTRICTED', '#e5484d',
            'CRITICAL_ASSET', '#e2703a',
            'PERIMETER', '#4a8fd4',
            '#7d8ba1',
          ],
          'fill-opacity': 0.12,
        },
      });
      instance.addLayer({
        id: 'zone-line',
        type: 'line',
        source: 'zones',
        paint: {
          'line-color': [
            'match',
            ['get', 'purpose'],
            'RESTRICTED', '#e5484d',
            'CRITICAL_ASSET', '#e2703a',
            'PERIMETER', '#4a8fd4',
            '#7d8ba1',
          ],
          'line-width': 1.5,
          'line-dasharray': [3, 2],
        },
      });

      // --- event uncertainty -----------------------------------------------
      // The radius is the uncertainty the pipeline actually reported, converted
      // from metres to screen pixels at the current latitude and zoom. A crisp
      // dot would be a claim of precision the system never made.
      instance.addLayer({
        id: 'event-uncertainty',
        type: 'circle',
        source: 'events',
        paint: {
          'circle-radius': [
            'interpolate',
            ['exponential', 2],
            ['zoom'],
            10,
            ['/', ['get', 'uncertaintyMeters'], 60],
            22,
            ['*', ['get', 'uncertaintyMeters'], 4],
          ],
          'circle-color': [
            'match',
            ['get', 'severity'],
            'CRITICAL', SEVERITY_COLOUR['CRITICAL'] ?? '#e5484d',
            'HIGH', SEVERITY_COLOUR['HIGH'] ?? '#e2703a',
            'MEDIUM', SEVERITY_COLOUR['MEDIUM'] ?? '#d9a441',
            SEVERITY_COLOUR['LOW'] ?? '#4a8fd4',
          ],
          'circle-opacity': 0.16,
          'circle-stroke-width': 1,
          'circle-stroke-color': '#ffffff',
          'circle-stroke-opacity': 0.18,
        },
      });

      instance.addLayer({
        id: 'event-point',
        type: 'circle',
        source: 'events',
        paint: {
          'circle-radius': 4,
          'circle-color': [
            'match',
            ['get', 'severity'],
            'CRITICAL', SEVERITY_COLOUR['CRITICAL'] ?? '#e5484d',
            'HIGH', SEVERITY_COLOUR['HIGH'] ?? '#e2703a',
            'MEDIUM', SEVERITY_COLOUR['MEDIUM'] ?? '#d9a441',
            SEVERITY_COLOUR['LOW'] ?? '#4a8fd4',
          ],
          'circle-stroke-width': 1,
          'circle-stroke-color': '#0a0d12',
        },
      });

      // --- cameras ----------------------------------------------------------
      instance.addLayer({
        id: 'camera-point',
        type: 'circle',
        source: 'cameras',
        paint: {
          'circle-radius': 6,
          'circle-color': '#11161f',
          'circle-stroke-width': 2,
          'circle-stroke-color': '#35c2f5',
        },
      });

      // Fit to everything the site actually contains.
      const bounds = new maplibregl.LngLatBounds();
      for (const feature of snapshot.geojson.cameras.features) {
        bounds.extend(feature.geometry.coordinates as [number, number]);
      }
      for (const feature of snapshot.geojson.zones.features) {
        for (const ring of feature.geometry.coordinates) {
          for (const point of ring) bounds.extend(point as [number, number]);
        }
      }
      if (!bounds.isEmpty()) instance.fitBounds(bounds, { padding: 70, duration: 0 });
    });

    map.current = instance;

    return () => {
      instance.remove();
      map.current = null;
    };
  }, [snapshot]);

  // Highlight the events belonging to the selected incident.
  useEffect(() => {
    const instance = map.current;
    if (instance === null || !instance.isStyleLoaded()) return;

    const incident = snapshot.incidents.find((i) => i.id === selectedIncidentId);
    if (incident === undefined) return;

    const events = snapshot.geojson.events.features.filter((feature) =>
      incident.cameraIds.includes(String(feature.properties['cameraId'])),
    );
    if (events.length === 0) return;

    const bounds = new maplibregl.LngLatBounds();
    for (const feature of events) {
      bounds.extend(feature.geometry.coordinates as [number, number]);
    }
    if (!bounds.isEmpty()) instance.fitBounds(bounds, { padding: 110, duration: 600 });
  }, [selectedIncidentId, snapshot]);

  return (
    <div className="map-wrap">
      <div className="map" ref={container} />

      <div className="map-notice">
        <b>OFFLINE MAP DATA NOT INSTALLED</b>
        No map package is present, so no basemap is drawn. Site geometry, camera
        coverage and event positions render from local data. The application will
        not fetch tiles from the Internet. Import a package under Settings &rarr;
        Maps.
      </div>

      <div className="map-legend">
        <div>
          <i style={{ background: '#35c2f5', opacity: 0.35 }} />
          camera coverage (true ground footprint)
        </div>
        <div>
          <i style={{ background: '#e5484d', opacity: 0.5 }} />
          restricted zone
        </div>
        <div>
          <i style={{ background: '#e2703a', opacity: 0.5 }} />
          protected asset
        </div>
        <div>
          <i style={{ background: '#e5484d' }} />
          event, ringed by its stated uncertainty
        </div>
      </div>
    </div>
  );
};
