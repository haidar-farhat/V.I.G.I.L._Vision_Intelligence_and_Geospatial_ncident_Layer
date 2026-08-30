import type {
  LatLon,
  UtcMillis,
  Vec2,
  Zone,
  ZoneGeometry,
  ZoneId,
  ZoneObservation,
  ZoneTransition,
} from '@sentinel/shared-types';
import { createLocalFrame, toLatLon, toLocal, toLocalRing } from './geodesy.ts';
import type { LocalFrame } from './geodesy.ts';
import {
  circleRing,
  distanceToPolyline,
  pointInPolygon,
  segmentsIntersect,
  sideOfLine,
} from './polygon.ts';

/**
 * The zone engine.
 *
 * Zones are authored in geographic coordinates but evaluated in a local metric
 * frame, so a single compiled zone can be tested against thousands of track
 * positions per second without repeating trigonometry.
 *
 * The engine reports **transitions**, not events. "This track entered zone B and
 * has been there 40 seconds" is a fact about geometry; whether that matters is a
 * question for the rule engine. Keeping the two apart means rules can be
 * rewritten, imported or disabled without any risk to the spatial mathematics.
 */

/** A zone pre-converted into a local metric frame, ready for repeated testing. */
export type CompiledZone = {
  readonly id: ZoneId;
  readonly frame: LocalFrame;
  readonly geometry: CompiledGeometry;
};

type CompiledGeometry =
  | { readonly kind: 'AREA'; readonly ring: readonly Vec2[] }
  | { readonly kind: 'LINE'; readonly path: readonly Vec2[] }
  | { readonly kind: 'CORRIDOR'; readonly path: readonly Vec2[]; readonly halfWidth: number };

const anchorOf = (geometry: ZoneGeometry): LatLon => {
  switch (geometry.kind) {
    case 'POLYGON':
    case 'RECTANGLE':
      return geometry.ring[0] ?? { lat: 0, lon: 0 };
    case 'LINE':
    case 'CORRIDOR':
      return geometry.path[0] ?? { lat: 0, lon: 0 };
    case 'CIRCLE':
      return geometry.center;
  }
};

/**
 * Convert a zone's geographic geometry into local metres.
 *
 * Circles become rings here so that every area zone shares one containment
 * implementation; the polygonal approximation is well under a centimetre for
 * realistic radii, far below the metre-scale uncertainty of the positions being
 * tested against it.
 */
export const compileZone = (zone: Zone): CompiledZone => {
  const frame = createLocalFrame(anchorOf(zone.geometry));
  const g = zone.geometry;

  let geometry: CompiledGeometry;
  switch (g.kind) {
    case 'POLYGON':
    case 'RECTANGLE':
      geometry = { kind: 'AREA', ring: toLocalRing(frame, g.ring) };
      break;
    case 'CIRCLE':
      geometry = { kind: 'AREA', ring: circleRing(toLocal(frame, g.center), g.radiusMeters, 48) };
      break;
    case 'LINE':
      geometry = { kind: 'LINE', path: toLocalRing(frame, g.path) };
      break;
    case 'CORRIDOR':
      geometry = {
        kind: 'CORRIDOR',
        path: toLocalRing(frame, g.path),
        halfWidth: g.widthMeters / 2,
      };
      break;
  }

  return { id: zone.id, frame, geometry };
};

/**
 * Whether a geographic point lies inside a zone.
 *
 * A LINE zone has no interior - it is a tripwire, and containment is meaningless
 * for it - so this is always false there. Crossing is tested separately.
 */
export const zoneContains = (zone: CompiledZone, point: LatLon): boolean => {
  const local = toLocal(zone.frame, point);
  switch (zone.geometry.kind) {
    case 'AREA':
      return pointInPolygon(local, zone.geometry.ring);
    case 'CORRIDOR':
      return distanceToPolyline(local, zone.geometry.path) <= zone.geometry.halfWidth;
    case 'LINE':
      return false;
  }
};

/**
 * Whether the movement from `from` to `to` crosses a LINE zone, and in which
 * direction.
 *
 * The sign is the side the track ended up on: +1 for a crossing to the left of
 * the line's own direction, -1 to the right. That is what lets a rule say
 * "entering" versus "leaving" for a tripwire that has no interior.
 */
export const zoneCrossing = (
  zone: CompiledZone,
  from: LatLon,
  to: LatLon,
): { readonly crossed: boolean; readonly sign: -1 | 1 | 0 } => {
  if (zone.geometry.kind !== 'LINE') return { crossed: false, sign: 0 };

  const path = zone.geometry.path;
  const a = toLocal(zone.frame, from);
  const b = toLocal(zone.frame, to);

  for (let i = 0; i < path.length - 1; i += 1) {
    const p = path[i];
    const q = path[i + 1];
    if (p === undefined || q === undefined) continue;
    if (segmentsIntersect(a, b, p, q)) {
      const sign = sideOfLine(p, q, b);
      return { crossed: true, sign: sign === 0 ? 1 : sign };
    }
  }
  return { crossed: false, sign: 0 };
};

/** Centre of a compiled zone in geographic coordinates, for map labels. */
export const zoneCenter = (zone: CompiledZone): LatLon => {
  const points =
    zone.geometry.kind === 'AREA'
      ? zone.geometry.ring
      : zone.geometry.kind === 'LINE'
        ? zone.geometry.path
        : zone.geometry.path;

  if (points.length === 0) return zone.frame.origin;
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const p of points) {
    if (p === undefined) continue;
    sx += p.x;
    sy += p.y;
    n += 1;
  }
  return n === 0 ? zone.frame.origin : toLatLon(zone.frame, { x: sx / n, y: sy / n });
};

// ---------------------------------------------------------------- membership

/** Per-track, per-zone state carried between position updates. */
type Membership = {
  inside: boolean;
  since: UtcMillis;
  lastPoint: LatLon;
  lastDwellReport: UtcMillis;
};

export type ZoneEngineOptions = {
  /**
   * How often a continuously-present track re-reports DWELLING.
   *
   * Dwell is reported periodically rather than once, so a loitering rule with any
   * threshold can be satisfied without the engine knowing what the thresholds
   * are. Defaults to one second.
   */
  readonly dwellReportIntervalMillis?: number;
};

/**
 * Tracks zone membership over time and emits transitions.
 *
 * One instance per camera (or per worker) holds the membership state for every
 * track it is following. State is keyed by track and zone, and is dropped when a
 * track ends, so memory is bounded by the number of live tracks.
 */
export class ZoneEngine {
  readonly #zones = new Map<ZoneId, CompiledZone>();
  readonly #membership = new Map<string, Map<ZoneId, Membership>>();
  readonly #dwellInterval: number;

  constructor(zones: readonly Zone[] = [], options: ZoneEngineOptions = {}) {
    this.#dwellInterval = options.dwellReportIntervalMillis ?? 1000;
    for (const zone of zones) this.setZone(zone);
  }

  setZone(zone: Zone): void {
    if (!zone.active) {
      this.#zones.delete(zone.id);
      return;
    }
    this.#zones.set(zone.id, compileZone(zone));
  }

  removeZone(zoneId: ZoneId): void {
    this.#zones.delete(zoneId);
    for (const zones of this.#membership.values()) zones.delete(zoneId);
  }

  get zoneCount(): number {
    return this.#zones.size;
  }

  compiled(zoneId: ZoneId): CompiledZone | undefined {
    return this.#zones.get(zoneId);
  }

  /**
   * Feed one position update for a track and receive every zone transition it
   * caused. Callers pass positions in chronological order per track.
   */
  update(trackKey: string, point: LatLon, at: UtcMillis): readonly ZoneObservation[] {
    let zones = this.#membership.get(trackKey);
    if (zones === undefined) {
      zones = new Map<ZoneId, Membership>();
      this.#membership.set(trackKey, zones);
    }

    const observations: ZoneObservation[] = [];

    for (const [zoneId, zone] of this.#zones) {
      const prior = zones.get(zoneId);

      if (zone.geometry.kind === 'LINE') {
        // Tripwires have no interior; they fire on the movement itself.
        if (prior !== undefined) {
          const { crossed, sign } = zoneCrossing(zone, prior.lastPoint, point);
          if (crossed && sign !== 0) {
            observations.push({
              zoneId,
              transition: 'CROSSED' satisfies ZoneTransition,
              at,
              crossingSign: sign,
            });
          }
        }
        zones.set(zoneId, {
          inside: false,
          since: prior?.since ?? at,
          lastPoint: point,
          lastDwellReport: prior?.lastDwellReport ?? at,
        });
        continue;
      }

      const inside = zoneContains(zone, point);
      const wasInside = prior?.inside ?? false;

      if (inside && !wasInside) {
        observations.push({ zoneId, transition: 'ENTERED' satisfies ZoneTransition, at });
        zones.set(zoneId, { inside: true, since: at, lastPoint: point, lastDwellReport: at });
        continue;
      }

      if (!inside && wasInside) {
        observations.push({ zoneId, transition: 'EXITED' satisfies ZoneTransition, at });
        zones.set(zoneId, { inside: false, since: at, lastPoint: point, lastDwellReport: at });
        continue;
      }

      if (inside && prior !== undefined) {
        const dwell = at - prior.since;
        const sinceReport = at - prior.lastDwellReport;
        if (sinceReport >= this.#dwellInterval) {
          observations.push({
            zoneId,
            transition: 'DWELLING' satisfies ZoneTransition,
            at,
            dwellMillis: dwell,
          });
          zones.set(zoneId, { ...prior, lastPoint: point, lastDwellReport: at });
          continue;
        }
      }

      zones.set(zoneId, {
        inside,
        since: prior?.since ?? at,
        lastPoint: point,
        lastDwellReport: prior?.lastDwellReport ?? at,
      });
    }

    return observations;
  }

  /** Zones a track is currently inside. */
  zonesFor(trackKey: string): readonly ZoneId[] {
    const zones = this.#membership.get(trackKey);
    if (zones === undefined) return [];
    const result: ZoneId[] = [];
    for (const [zoneId, state] of zones) if (state.inside) result.push(zoneId);
    return result;
  }

  /** Release a track's state. Called when a track ends, to bound memory. */
  forget(trackKey: string): void {
    this.#membership.delete(trackKey);
  }

  get trackedCount(): number {
    return this.#membership.size;
  }
}
