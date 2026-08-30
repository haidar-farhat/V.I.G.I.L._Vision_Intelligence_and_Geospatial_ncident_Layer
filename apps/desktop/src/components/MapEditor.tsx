import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import { analyseZoneCoverage, fieldOfViewWedge } from '@sentinel/geometry';
import type { CameraPose, Zone } from '@sentinel/shared-types';
import type { Snapshot, SnapshotPose, SnapshotZone } from '../data/types.ts';

/**
 * Camera placement and coverage.
 *
 * The footprints and blind spots drawn here are computed by `@sentinel/geometry`
 * in the browser - the same code the detection pipeline runs, not a
 * reimplementation for display. That is the whole point: an operator dragging a
 * heading sees the coverage the system will actually have, and if the geometry
 * changes the map changes with it. A UI that draws its own approximation of a
 * field of view will eventually disagree with the engine, and the disagreement
 * will be discovered by someone standing in a gap the map said was covered.
 *
 * Placement is not persisted. There is no API yet, so changes live in this
 * component and the panel says so rather than implying a save that never happens.
 */

type Props = {
  readonly snapshot: Snapshot;
};

const EMPTY_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {},
  layers: [{ id: 'ground', type: 'background', paint: { 'background-color': '#0d1117' } }],
};

const ZONE_COLOUR: Record<string, string> = {
  RESTRICTED: '#e5484d',
  CRITICAL_ASSET: '#e2703a',
  PERIMETER: '#4a8fd4',
};

/** Rebuild a domain Zone from the snapshot so the real analysis can run on it. */
const toZone = (zone: SnapshotZone): Zone =>
  ({
    id: zone.id,
    name: zone.name,
    purpose: zone.purpose,
    geometry: zone.geometry,
    locationId: null,
    parentZoneId: null,
    active: true,
    createdAt: 0,
    updatedAt: 0,
  }) as unknown as Zone;

const toPose = (pose: SnapshotPose): CameraPose => pose as unknown as CameraPose;

const emptyCollection = { type: 'FeatureCollection' as const, features: [] };

export const MapEditor = ({ snapshot }: Props): JSX.Element => {
  const container = useRef<HTMLDivElement | null>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);

  const [selectedId, setSelectedId] = useState<string>(snapshot.cameras[0]?.id ?? '');

  /** Working poses, keyed by camera. Edits live here until an API exists to save them. */
  const [poses, setPoses] = useState<Record<string, SnapshotPose>>(() =>
    Object.fromEntries(snapshot.cameras.map((camera) => [camera.id, camera.pose])),
  );
  const [dirty, setDirty] = useState<ReadonlySet<string>>(new Set());

  const selectedPose = poses[selectedId];

  const zones = useMemo(() => snapshot.zones.map(toZone), [snapshot]);

  /**
   * Coverage, recomputed whenever a pose changes.
   *
   * Cheap enough to run on every slider movement at this site size, and running
   * it eagerly is what makes the blind-spot overlay respond as the operator
   * drags rather than after they let go.
   */
  const coverage = useMemo(
    () =>
      zones.map((zone) =>
        analyseZoneCoverage(
          zone,
          snapshot.cameras.map((camera) => ({
            cameraId: camera.id,
            pose: toPose(poses[camera.id] ?? camera.pose),
          })),
          { sampleSpacingMeters: 6, maxSamples: 1500, maxBlindSpots: 400 },
        ),
      ),
    [zones, poses, snapshot.cameras],
  );

  /** Footprints for every camera, from the production geometry. */
  const fovCollection = useMemo(
    () => ({
      type: 'FeatureCollection' as const,
      features: snapshot.cameras.map((camera) => {
        const pose = toPose(poses[camera.id] ?? camera.pose);
        return {
          type: 'Feature' as const,
          geometry: {
            type: 'Polygon' as const,
            coordinates: [fieldOfViewWedge(pose, 32).map((point) => [point.lon, point.lat])],
          },
          properties: { id: camera.id, selected: camera.id === selectedId ? 1 : 0 },
        };
      }),
    }),
    [poses, selectedId, snapshot.cameras],
  );

  const cameraCollection = useMemo(
    () => ({
      type: 'FeatureCollection' as const,
      features: snapshot.cameras.map((camera) => {
        const pose = poses[camera.id] ?? camera.pose;
        return {
          type: 'Feature' as const,
          geometry: {
            type: 'Point' as const,
            coordinates: [pose.position.lon, pose.position.lat],
          },
          properties: { id: camera.id, selected: camera.id === selectedId ? 1 : 0 },
        };
      }),
    }),
    [poses, selectedId, snapshot.cameras],
  );

  const blindCollection = useMemo(
    () => ({
      type: 'FeatureCollection' as const,
      features: coverage.flatMap((zone) =>
        zone.blindSpots.map((point) => ({
          type: 'Feature' as const,
          geometry: { type: 'Point' as const, coordinates: [point.lon, point.lat] },
          properties: { zone: zone.zoneName },
        })),
      ),
    }),
    [coverage],
  );

  const zoneCollection = useMemo(
    () => ({
      type: 'FeatureCollection' as const,
      features: snapshot.zones.flatMap((zone) => {
        if (zone.geometry.kind !== 'POLYGON' && zone.geometry.kind !== 'RECTANGLE') return [];
        const ring = zone.geometry.ring.map((point) => [point.lon, point.lat]);
        if (ring.length === 0) return [];
        return [
          {
            type: 'Feature' as const,
            geometry: { type: 'Polygon' as const, coordinates: [[...ring, ring[0]!]] },
            properties: { id: zone.id, name: zone.name, purpose: zone.purpose },
          },
        ];
      }),
    }),
    [snapshot.zones],
  );

  // ------------------------------------------------------------------- map

  useEffect(() => {
    const element = container.current;
    if (element === null || map.current !== null) return;

    const first = snapshot.cameras[0];
    const instance = new maplibregl.Map({
      container: element,
      style: EMPTY_STYLE,
      center: [first?.position.lon ?? 0, first?.position.lat ?? 0],
      zoom: 16,
      attributionControl: false,
    });

    instance.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-right');
    instance.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');

    instance.on('load', () => {
      for (const id of ['zones', 'fov', 'blind', 'cameras']) {
        instance.addSource(id, { type: 'geojson', data: emptyCollection });
      }

      instance.addLayer({
        id: 'zone-fill',
        type: 'fill',
        source: 'zones',
        paint: {
          'fill-color': [
            'match',
            ['get', 'purpose'],
            'RESTRICTED', ZONE_COLOUR['RESTRICTED'] ?? '#e5484d',
            'CRITICAL_ASSET', ZONE_COLOUR['CRITICAL_ASSET'] ?? '#e2703a',
            'PERIMETER', ZONE_COLOUR['PERIMETER'] ?? '#4a8fd4',
            '#7d8ba1',
          ],
          'fill-opacity': 0.1,
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
            'RESTRICTED', ZONE_COLOUR['RESTRICTED'] ?? '#e5484d',
            'CRITICAL_ASSET', ZONE_COLOUR['CRITICAL_ASSET'] ?? '#e2703a',
            'PERIMETER', ZONE_COLOUR['PERIMETER'] ?? '#4a8fd4',
            '#7d8ba1',
          ],
          'line-width': 1.5,
          'line-dasharray': [3, 2],
        },
      });

      instance.addLayer({
        id: 'fov-fill',
        type: 'fill',
        source: 'fov',
        paint: {
          'fill-color': ['case', ['==', ['get', 'selected'], 1], '#35c2f5', '#4a8fd4'],
          'fill-opacity': ['case', ['==', ['get', 'selected'], 1], 0.22, 0.07],
        },
      });
      instance.addLayer({
        id: 'fov-line',
        type: 'line',
        source: 'fov',
        paint: {
          'line-color': '#35c2f5',
          'line-opacity': ['case', ['==', ['get', 'selected'], 1], 0.85, 0.25],
          'line-width': ['case', ['==', ['get', 'selected'], 1], 1.6, 1],
        },
      });

      // Blind spots sit above coverage so a gap reads as a hole punched in it.
      instance.addLayer({
        id: 'blind-spots',
        type: 'circle',
        source: 'blind',
        paint: {
          'circle-radius': 3,
          'circle-color': '#e5484d',
          'circle-opacity': 0.75,
          'circle-stroke-width': 0,
        },
      });

      instance.addLayer({
        id: 'camera-point',
        type: 'circle',
        source: 'cameras',
        paint: {
          'circle-radius': ['case', ['==', ['get', 'selected'], 1], 8, 6],
          'circle-color': '#11161f',
          'circle-stroke-width': 2,
          'circle-stroke-color': ['case', ['==', ['get', 'selected'], 1], '#ffffff', '#35c2f5'],
        },
      });

      instance.on('click', 'camera-point', (event) => {
        const id = event.features?.[0]?.properties?.['id'];
        if (typeof id === 'string') setSelectedId(id);
        event.preventDefault();
      });

      instance.on('mouseenter', 'camera-point', () => {
        instance.getCanvas().style.cursor = 'pointer';
      });
      instance.on('mouseleave', 'camera-point', () => {
        instance.getCanvas().style.cursor = 'crosshair';
      });
      instance.getCanvas().style.cursor = 'crosshair';

      setReady(true);
    });

    map.current = instance;
    return () => {
      instance.remove();
      map.current = null;
      setReady(false);
    };
  }, [snapshot]);

  /** Click empty map to move the selected camera there. */
  useEffect(() => {
    const instance = map.current;
    if (instance === null || !ready) return;

    const onClick = (event: maplibregl.MapMouseEvent): void => {
      if (event.defaultPrevented) return;
      if (selectedId === '') return;

      setPoses((current) => {
        const pose = current[selectedId];
        if (pose === undefined) return current;
        return {
          ...current,
          [selectedId]: {
            ...pose,
            position: { ...pose.position, lat: event.lngLat.lat, lon: event.lngLat.lng },
          },
        };
      });
      setDirty((current) => new Set(current).add(selectedId));
    };

    instance.on('click', onClick);
    return () => {
      instance.off('click', onClick);
    };
  }, [ready, selectedId]);

  /** Push recomputed geometry to the map whenever it changes. */
  useEffect(() => {
    const instance = map.current;
    if (instance === null || !ready) return;

    const push = (id: string, data: unknown): void => {
      const source = instance.getSource(id);
      if (source !== undefined && 'setData' in source) {
        (source as maplibregl.GeoJSONSource).setData(data as never);
      }
    };

    push('zones', zoneCollection);
    push('fov', fovCollection);
    push('blind', blindCollection);
    push('cameras', cameraCollection);
  }, [ready, zoneCollection, fovCollection, blindCollection, cameraCollection]);

  const update = useCallback(
    (field: keyof SnapshotPose, value: number): void => {
      setPoses((current) => {
        const pose = current[selectedId];
        if (pose === undefined) return current;
        return { ...current, [selectedId]: { ...pose, [field]: value } };
      });
      setDirty((current) => new Set(current).add(selectedId));
    },
    [selectedId],
  );

  const reset = useCallback((): void => {
    setPoses(Object.fromEntries(snapshot.cameras.map((camera) => [camera.id, camera.pose])));
    setDirty(new Set());
  }, [snapshot.cameras]);

  const selectedCamera = snapshot.cameras.find((camera) => camera.id === selectedId);
  const totalBlind = coverage.reduce((sum, zone) => sum + zone.blindSpots.length, 0);

  return (
    <main className="command map-editor">
      {/* -------------------------------------------------------- cameras */}
      <section className="panel">
        <div className="panel-head">
          Cameras
          <span className="count">{snapshot.cameras.length}</span>
        </div>
        <div className="panel-body">
          {snapshot.cameras.map((camera) => (
            <button
              type="button"
              key={camera.id}
              className="camera as-button"
              aria-selected={camera.id === selectedId}
              onClick={() => setSelectedId(camera.id)}
            >
              <i className={`state${camera.reporting ? '' : ' idle'}`} />
              <div>
                <div className="name">{camera.name}</div>
                <div className="meta">
                  {Math.round(poses[camera.id]?.heading ?? 0)}° ·{' '}
                  {Math.round(poses[camera.id]?.horizontalFov ?? 0)}° fov ·{' '}
                  {Math.round(poses[camera.id]?.rangeMeters ?? 0)} m
                </div>
              </div>
              {dirty.has(camera.id) ? <div className="tracks">•</div> : <div />}
            </button>
          ))}

          <div className="section">
            <h3>Placement</h3>
            <p className="note">
              Click the map to move the selected camera. Footprints and blind spots are
              computed by the same geometry the detection pipeline uses, so what you see
              here is the coverage the system will actually have.
            </p>
            {dirty.size > 0 ? (
              <p className="note warn">
                {dirty.size} camera{dirty.size === 1 ? '' : 's'} moved. Nothing is saved —
                there is no API yet.{' '}
                <button type="button" className="link" onClick={reset}>
                  Reset
                </button>
              </p>
            ) : null}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------ map */}
      <section className="panel map-panel">
        <div className="panel-head">
          Site map
          <span className="count">
            {totalBlind > 0 ? `${totalBlind} blind sample(s)` : 'no gaps found'}
          </span>
        </div>
        <div className="map-wrap">
          <div className="map" ref={container} />
          <div className="map-notice">
            <b>OFFLINE MAP DATA NOT INSTALLED</b>
            No basemap is drawn. Site geometry, coverage and blind spots render from local
            data and will not fetch tiles from the Internet.
          </div>
          <div className="map-legend">
            <div>
              <i style={{ background: '#35c2f5', opacity: 0.5 }} />
              camera coverage (true ground footprint)
            </div>
            <div>
              <i style={{ background: '#e5484d', opacity: 0.5 }} />
              restricted zone
            </div>
            <div>
              <i style={{ background: '#e5484d' }} />
              blind spot — no camera sees this point
            </div>
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------- controls */}
      <section className="panel">
        <div className="panel-head">
          {selectedCamera?.name ?? 'No camera selected'}
        </div>
        <div className="panel-body pad">
          {selectedPose === undefined ? (
            <div className="empty">Select a camera.</div>
          ) : (
            <>
              <h4 className="control-head">Pose</h4>

              <Slider
                label="Heading"
                unit="°"
                min={0}
                max={359}
                step={1}
                value={selectedPose.heading}
                onChange={(value) => update('heading', value)}
              />
              <Slider
                label="Tilt"
                unit="°"
                min={-80}
                max={-1}
                step={1}
                value={selectedPose.pitch}
                onChange={(value) => update('pitch', value)}
                hint="Negative is downward. A steeper tilt sees closer ground and less of it."
              />
              <Slider
                label="Mount height"
                unit=" m"
                min={2}
                max={20}
                step={0.5}
                value={selectedPose.mountHeight}
                onChange={(value) => update('mountHeight', value)}
              />
              <Slider
                label="Horizontal FOV"
                unit="°"
                min={10}
                max={120}
                step={1}
                value={selectedPose.horizontalFov}
                onChange={(value) => update('horizontalFov', value)}
              />
              <Slider
                label="Vertical FOV"
                unit="°"
                min={10}
                max={90}
                step={1}
                value={selectedPose.verticalFov}
                onChange={(value) => update('verticalFov', value)}
              />
              <Slider
                label="Range"
                unit=" m"
                min={10}
                max={300}
                step={5}
                value={selectedPose.rangeMeters}
                onChange={(value) => update('rangeMeters', value)}
              />

              <h4 className="control-head">Coverage</h4>
              {coverage.map((zone) => (
                <div className="coverage-row" key={zone.zoneId}>
                  <div className="coverage-top">
                    <span className={`verdict ${zone.verdict}`}>{zone.verdict}</span>
                    <span className="coverage-name">{zone.zoneName}</span>
                    <span className="coverage-pct">
                      {Math.round(zone.coveredFraction * 100)}%
                    </span>
                  </div>
                  <div className="coverage-bar">
                    <i
                      className="covered"
                      style={{ width: `${zone.coveredFraction * 100}%` }}
                    />
                    <i
                      className="redundant"
                      style={{ width: `${zone.redundantFraction * 100}%` }}
                    />
                  </div>
                  <p className="note">{zone.summary}</p>
                </div>
              ))}

              <p className="note provenance">
                Coverage is sampled on a fixed grid and tested against each camera&apos;s
                real annular footprint. A tilted camera cannot see the ground at its own
                mast, so the blind foreground is excluded rather than assumed covered.
              </p>
            </>
          )}
        </div>
      </section>
    </main>
  );
};

type SliderProps = {
  readonly label: string;
  readonly unit: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly value: number;
  readonly hint?: string;
  readonly onChange: (value: number) => void;
};

const Slider = ({
  label,
  unit,
  min,
  max,
  step,
  value,
  hint,
  onChange,
}: SliderProps): JSX.Element => (
  <div className="slider">
    <label>
      <span className="slider-label">{label}</span>
      <span className="slider-value">
        {Math.round(value * 10) / 10}
        {unit}
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
    {hint === undefined ? null : <p className="slider-hint">{hint}</p>}
  </div>
);
