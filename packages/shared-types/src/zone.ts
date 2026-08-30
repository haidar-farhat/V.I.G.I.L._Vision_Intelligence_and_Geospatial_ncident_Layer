import type { LocationId, UtcMillis, ZoneId } from './ids.ts';
import type { LocationKind, ZoneKind, ZonePurpose } from './enums.ts';
import type { GeoPoint, LatLon } from './geo.ts';

/**
 * Zone geometry.
 *
 * All variants carry geographic coordinates so a zone is meaningful on the map
 * independent of any one camera. Zone evaluation projects tracks to the ground
 * plane first, then tests them here.
 */
export type ZoneGeometry =
  | { readonly kind: 'POLYGON'; readonly ring: readonly LatLon[] }
  | { readonly kind: 'RECTANGLE'; readonly ring: readonly LatLon[] }
  | { readonly kind: 'LINE'; readonly path: readonly LatLon[] }
  | { readonly kind: 'CIRCLE'; readonly center: LatLon; readonly radiusMeters: number }
  | {
      readonly kind: 'CORRIDOR';
      readonly path: readonly LatLon[];
      readonly widthMeters: number;
    };

export type Zone = {
  readonly id: ZoneId;
  readonly name: string;
  readonly purpose: ZonePurpose;
  readonly geometry: ZoneGeometry;
  readonly locationId: LocationId | null;
  /** Parent zone, enabling site -> perimeter -> restricted area -> gate nesting. */
  readonly parentZoneId: ZoneId | null;
  readonly active: boolean;
  readonly createdAt: UtcMillis;
  readonly updatedAt: UtcMillis;
};

/** Convenience discriminator matching `ZoneGeometry['kind']`. */
export type ZoneGeometryKind = ZoneKind;

/**
 * A physical place. Locations nest (site -> building -> floor -> room) and may be
 * geographic or purely indoor, which is what lets indoor sites reuse the entire
 * zone and event model without a parallel implementation.
 */
export type Location = {
  readonly id: LocationId;
  readonly name: string;
  readonly kind: LocationKind;
  readonly parentId: LocationId | null;
  readonly position: GeoPoint | null;
  /** Floor plan bounds for indoor locations, in the local frame. */
  readonly indoorFrame?: {
    readonly widthMeters: number;
    readonly heightMeters: number;
    readonly originLatLon: LatLon;
    readonly rotationDegrees: number;
  };
  readonly createdAt: UtcMillis;
  readonly updatedAt: UtcMillis;
};

/** Zone membership transition observed for a track. */
export const ZoneTransition = {
  Entered: 'ENTERED',
  Exited: 'EXITED',
  Dwelling: 'DWELLING',
  Crossed: 'CROSSED',
} as const;
export type ZoneTransition = (typeof ZoneTransition)[keyof typeof ZoneTransition];

/**
 * The zone engine's output: a factual, rule-free statement that a track did
 * something with respect to a zone. Rules turn these into events; keeping the
 * two apart is what allows rules to change without touching geometry.
 */
export type ZoneObservation = {
  readonly zoneId: ZoneId;
  readonly transition: ZoneTransition;
  readonly at: UtcMillis;
  /** For DWELLING: how long the track has been continuously inside. */
  readonly dwellMillis?: number;
  /** For CROSSED on a LINE zone: which way it went, as a signed side change. */
  readonly crossingSign?: -1 | 1;
};
