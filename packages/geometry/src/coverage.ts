import type { CameraPose, LatLon, Zone } from '@sentinel/shared-types';
import { createLocalFrame, haversineDistance, toLatLon, toLocal } from './geodesy.ts';
import { bearingInFov, farGroundDistance, nearGroundDistance } from './projection.ts';
import { compileZone, zoneContains } from './zones.ts';

/**
 * Camera coverage analysis.
 *
 * Answers the question an operator actually has when placing cameras: **can
 * anything here be seen?**
 *
 * This matters because the failure it prevents is invisible. A site with a
 * restricted zone and four cameras looks protected on a map. If none of those
 * cameras' footprints reach the zone's north-east corner, nothing detects an
 * intrusion there and nothing ever reports that it could not - the system
 * silently monitors ground it cannot see, and the gap is discovered by whoever
 * walks through it.
 *
 * The analysis samples the zone on a grid and tests each point against every
 * camera's real ground footprint - the annular sector between its near and far
 * ground distances, not a pie slice from the mast. A tilted camera cannot see the
 * ground at its own base, and treating it as if it could is how a blind spot ends
 * up marked as covered.
 */

/** Whether a camera can see a specific ground point. */
export const cameraSees = (pose: CameraPose, point: LatLon): boolean => {
  const distance = haversineDistance(pose.position, point);

  const near = nearGroundDistance(pose);
  const far = farGroundDistance(pose);
  const effectiveFar = far === null ? pose.rangeMeters : Math.min(far, pose.rangeMeters);

  if (distance < near || distance > effectiveFar) return false;

  // Bearing is only meaningful once the point is far enough away for the
  // direction to be well defined; a point at the mast has no bearing.
  if (distance < 0.5) return false;

  const bearing = bearingOf(pose.position, point);
  return bearingInFov(pose, bearing);
};

/** Local bearing helper, kept here so coverage does not depend on projection internals. */
const bearingOf = (from: LatLon, to: LatLon): number => {
  const frame = createLocalFrame(from);
  const local = toLocal(frame, to);
  const degrees = (Math.atan2(local.x, local.y) * 180) / Math.PI;
  return degrees < 0 ? degrees + 360 : degrees;
};

export type CoverageSample = {
  readonly point: LatLon;
  /** How many cameras can see this point. Zero is a blind spot. */
  readonly cameraCount: number;
};

export type ZoneCoverage = {
  readonly zoneId: string;
  readonly zoneName: string;
  /** Fraction of the zone visible to at least one camera, 0..1. */
  readonly coveredFraction: number;
  /** Fraction visible to two or more, which is what survives one camera failing. */
  readonly redundantFraction: number;
  readonly totalSamples: number;
  readonly coveredSamples: number;
  /** Points no camera can see. Rendered on the map as the actual gap. */
  readonly blindSpots: readonly LatLon[];
  /** Cameras contributing any coverage at all, most first. */
  readonly contributingCameras: readonly { readonly cameraId: string; readonly samples: number }[];
  readonly verdict: CoverageVerdict;
  readonly summary: string;
};

export const CoverageVerdict = {
  /** Every sampled point is seen by two or more cameras. */
  Redundant: 'REDUNDANT',
  /** Every sampled point is seen, but some by only one camera. */
  Covered: 'COVERED',
  /** Some of the zone is visible, some is not. */
  Partial: 'PARTIAL',
  /** No camera sees any part of this zone. */
  Blind: 'BLIND',
} as const;
export type CoverageVerdict = (typeof CoverageVerdict)[keyof typeof CoverageVerdict];

export type CoverageOptions = {
  /**
   * Grid spacing in metres. Five metres is finer than a person is wide, which is
   * the scale at which a gap starts to matter.
   */
  readonly sampleSpacingMeters?: number;
  /** Cap on samples, so a site-sized zone cannot make the UI unresponsive. */
  readonly maxSamples?: number;
  /** Blind spots returned. The rest are counted but not listed. */
  readonly maxBlindSpots?: number;
};

const DEFAULTS = {
  sampleSpacingMeters: 5,
  maxSamples: 4000,
  maxBlindSpots: 200,
} as const;

/**
 * Analyse how well a set of cameras covers one zone.
 *
 * Cameras with no pose are skipped rather than assumed to see nothing in
 * particular: an unplaced camera may well cover the zone, and reporting a gap
 * that placement would close would send an operator installing hardware they
 * already own.
 */
export const analyseZoneCoverage = (
  zone: Zone,
  cameras: readonly { readonly cameraId: string; readonly pose: CameraPose | null }[],
  options: CoverageOptions = {},
): ZoneCoverage => {
  const spacing = options.sampleSpacingMeters ?? DEFAULTS.sampleSpacingMeters;
  const maxSamples = options.maxSamples ?? DEFAULTS.maxSamples;
  const maxBlindSpots = options.maxBlindSpots ?? DEFAULTS.maxBlindSpots;

  const placed = cameras.filter(
    (camera): camera is { cameraId: string; pose: CameraPose } => camera.pose !== null,
  );

  const samples = sampleZone(zone, spacing, maxSamples);

  if (samples.length === 0) {
    return {
      zoneId: String(zone.id),
      zoneName: zone.name,
      coveredFraction: 0,
      redundantFraction: 0,
      totalSamples: 0,
      coveredSamples: 0,
      blindSpots: [],
      contributingCameras: [],
      verdict: CoverageVerdict.Blind,
      summary: `${zone.name} has no area to analyse.`,
    };
  }

  const perCamera = new Map<string, number>();
  const blindSpots: LatLon[] = [];
  let covered = 0;
  let redundant = 0;

  for (const point of samples) {
    let seenBy = 0;

    for (const camera of placed) {
      if (!cameraSees(camera.pose, point)) continue;
      seenBy += 1;
      perCamera.set(camera.cameraId, (perCamera.get(camera.cameraId) ?? 0) + 1);
    }

    if (seenBy === 0) {
      if (blindSpots.length < maxBlindSpots) blindSpots.push(point);
    } else {
      covered += 1;
      if (seenBy >= 2) redundant += 1;
    }
  }

  const coveredFraction = covered / samples.length;
  const redundantFraction = redundant / samples.length;

  const verdict =
    covered === 0
      ? CoverageVerdict.Blind
      : covered < samples.length
        ? CoverageVerdict.Partial
        : redundant === samples.length
          ? CoverageVerdict.Redundant
          : CoverageVerdict.Covered;

  const contributingCameras = [...perCamera.entries()]
    .map(([cameraId, count]) => ({ cameraId, samples: count }))
    .sort((a, b) => b.samples - a.samples || (a.cameraId < b.cameraId ? -1 : 1));

  return {
    zoneId: String(zone.id),
    zoneName: zone.name,
    coveredFraction,
    redundantFraction,
    totalSamples: samples.length,
    coveredSamples: covered,
    blindSpots,
    contributingCameras,
    verdict,
    summary: describeCoverage(zone.name, verdict, coveredFraction, redundantFraction, placed.length),
  };
};

/**
 * Plain-language verdict.
 *
 * Written for someone deciding where to mount the next camera, so it says what is
 * wrong and what it implies rather than reporting a percentage and stopping.
 */
const describeCoverage = (
  zoneName: string,
  verdict: CoverageVerdict,
  coveredFraction: number,
  redundantFraction: number,
  placedCameras: number,
): string => {
  const percent = (value: number): string => `${Math.round(value * 100)}%`;

  switch (verdict) {
    case CoverageVerdict.Blind:
      return placedCameras === 0
        ? `No camera has been placed on the map, so nothing is known about ${zoneName}.`
        : `No camera can see any part of ${zoneName}. Anything happening here will go undetected.`;
    case CoverageVerdict.Partial:
      return (
        `${percent(coveredFraction)} of ${zoneName} is visible to at least one camera. ` +
        `The remaining ${percent(1 - coveredFraction)} is a blind spot - an intrusion there ` +
        'would not be detected.'
      );
    case CoverageVerdict.Covered:
      return (
        `${zoneName} is fully visible, but ${percent(1 - redundantFraction)} of it is seen by ` +
        'only one camera. Losing that camera would open a gap.'
      );
    case CoverageVerdict.Redundant:
      return `${zoneName} is fully visible to at least two cameras throughout.`;
  }
};

/**
 * Sample the interior of a zone on a regular grid.
 *
 * Grid rather than random: a random sample makes coverage percentages jitter
 * between runs, and an operator moving a camera slightly needs to see the number
 * move because of the camera, not because of the sampler.
 */
export const sampleZone = (
  zone: Zone,
  spacingMeters: number,
  maxSamples: number,
): readonly LatLon[] => {
  const compiled = compileZone(zone);
  const anchor = anchorOf(zone);
  const frame = createLocalFrame(anchor);

  const extent = localExtent(zone, anchor);
  if (extent === null) return [];

  const width = extent.maxX - extent.minX;
  const height = extent.maxY - extent.minY;

  // Widen the spacing rather than truncating the grid, so a large zone is sampled
  // evenly at lower resolution instead of finely in one corner.
  const estimated = ((width / spacingMeters) + 1) * ((height / spacingMeters) + 1);
  const spacing =
    estimated > maxSamples ? spacingMeters * Math.sqrt(estimated / maxSamples) : spacingMeters;

  const points: LatLon[] = [];

  for (let y = extent.minY; y <= extent.maxY && points.length < maxSamples; y += spacing) {
    for (let x = extent.minX; x <= extent.maxX && points.length < maxSamples; x += spacing) {
      const candidate = toLatLon(frame, { x, y });
      if (zoneContains(compiled, candidate)) points.push(candidate);
    }
  }

  return points;
};

const anchorOf = (zone: Zone): LatLon => {
  const geometry = zone.geometry;
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

const localExtent = (
  zone: Zone,
  anchor: LatLon,
): { minX: number; minY: number; maxX: number; maxY: number } | null => {
  const frame = createLocalFrame(anchor);
  const geometry = zone.geometry;

  const points: LatLon[] =
    geometry.kind === 'POLYGON' || geometry.kind === 'RECTANGLE'
      ? [...geometry.ring]
      : geometry.kind === 'LINE' || geometry.kind === 'CORRIDOR'
        ? [...geometry.path]
        : [geometry.center];

  if (points.length === 0) return null;

  // A circle and a corridor extend beyond their defining points.
  const padding =
    geometry.kind === 'CIRCLE'
      ? geometry.radiusMeters
      : geometry.kind === 'CORRIDOR'
        ? geometry.widthMeters / 2
        : 0;

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const point of points) {
    const local = toLocal(frame, point);
    if (local.x < minX) minX = local.x;
    if (local.x > maxX) maxX = local.x;
    if (local.y < minY) minY = local.y;
    if (local.y > maxY) maxY = local.y;
  }

  if (!Number.isFinite(minX)) return null;

  return {
    minX: minX - padding,
    minY: minY - padding,
    maxX: maxX + padding,
    maxY: maxY + padding,
  };
};

/**
 * Which zones a single camera contributes to, for the camera detail page.
 *
 * Answers "what is this camera actually for?" - a question that is surprisingly
 * hard to answer from a map once a site has thirty of them.
 */
export const zonesSeenByCamera = (
  pose: CameraPose | null,
  zones: readonly Zone[],
  options: CoverageOptions = {},
): readonly { readonly zone: Zone; readonly fraction: number }[] => {
  if (pose === null) return [];

  const results: { zone: Zone; fraction: number }[] = [];

  for (const zone of zones) {
    if (!zone.active) continue;

    const coverage = analyseZoneCoverage(zone, [{ cameraId: 'candidate', pose }], options);
    if (coverage.coveredFraction > 0) {
      results.push({ zone, fraction: coverage.coveredFraction });
    }
  }

  return results.sort((a, b) => b.fraction - a.fraction);
};
