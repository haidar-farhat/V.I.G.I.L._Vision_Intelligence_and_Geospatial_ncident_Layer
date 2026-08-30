/**
 * Spatial value types.
 *
 * Pure data shapes only; all algorithms live in `@sentinel/geometry`. Keeping the
 * shapes here lets the database, protocol and UI agree on geometry without any of
 * them depending on the math package.
 */

/** A point in a planar coordinate system (metres, or normalised image space). */
export type Vec2 = { readonly x: number; readonly y: number };

/** WGS84 geographic position. Degrees, not radians. */
export type LatLon = { readonly lat: number; readonly lon: number };

export type GeoPoint = LatLon & {
  /** Metres above the WGS84 ellipsoid. Optional: many sites never survey it. */
  readonly altitude?: number;
};

/** Axis-aligned geographic bounds. */
export type GeoBounds = {
  readonly minLat: number;
  readonly minLon: number;
  readonly maxLat: number;
  readonly maxLon: number;
};

/**
 * Detection box in **normalised image space**: 0..1 relative to frame width and
 * height, origin top-left. Normalised rather than pixels so a box survives a
 * resolution change, a sub-stream switch, or a model input-size change.
 */
export type BoundingBox = {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
};

/**
 * A position on the map together with an honest statement of how well it is
 * known. `radiusMeters` is a 1-sigma horizontal uncertainty.
 *
 * Rendering an uncalibrated camera's projection as a crisp point would be false
 * precision, which the UI treats as a defect: the uncertainty travels with the
 * position everywhere, and `source` records how it was obtained.
 */
export type PositionEstimate = {
  readonly point: LatLon;
  readonly radiusMeters: number;
  readonly source: PositionSource;
};

export const PositionSource = {
  /** Projected through a calibrated camera pose onto the ground plane. */
  GroundProjection: 'GROUND_PROJECTION',
  /** Camera position used as a stand-in; the object is somewhere in the FOV. */
  CameraFallback: 'CAMERA_FALLBACK',
  /** Operator placed it by hand. */
  Manual: 'MANUAL',
  /** Simulated ground truth (development and demo only). */
  Simulated: 'SIMULATED',
} as const;
export type PositionSource = (typeof PositionSource)[keyof typeof PositionSource];

/**
 * Camera pose and optics, enough to project image points onto the ground plane.
 *
 * Angles in degrees. `heading` is compass bearing (0 = north, 90 = east).
 * `pitch` is negative when the camera looks down, which is the normal mounting.
 */
export type CameraPose = {
  readonly position: GeoPoint;
  /** Height of the lens above the local ground plane, in metres. */
  readonly mountHeight: number;
  readonly heading: number;
  readonly pitch: number;
  readonly roll: number;
  readonly horizontalFov: number;
  readonly verticalFov: number;
  /** Useful observation distance in metres; beyond this, detections are ignored for mapping. */
  readonly rangeMeters: number;
};

/** Optional intrinsics. Absent for the overwhelming majority of deployments. */
export type CameraIntrinsics = {
  readonly fx: number;
  readonly fy: number;
  readonly cx: number;
  readonly cy: number;
  readonly distortion?: readonly number[];
};

/** Closed ring in planar metres, used by the zone engine. First point is not repeated. */
export type Ring = readonly Vec2[];

/** Closed ring in geographic coordinates. First point is not repeated. */
export type GeoRing = readonly LatLon[];
