import type {
  CameraId,
  CameraPose,
  CameraTopologyEdge,
  LatLon,
  Rule,
  RuleId,
  Zone,
  ZoneId,
} from '@sentinel/shared-types';
import { asId, utcMillis } from '@sentinel/shared-types';
import type { Actor, SimulatedCamera } from './world.ts';
import { offset } from './world.ts';

/**
 * The reference site and its flagship scenario.
 *
 * A linear perimeter road runs east across the site. Cameras are mounted south of
 * it looking north, with deliberate gaps between their fields of view - because a
 * site with continuous coverage would never exercise cross-camera correlation,
 * which is the hardest and most valuable thing this platform does.
 *
 * The scenario: three people walk the road at dusk, cross a restricted zone,
 * disappear into a coverage gap, are picked up again by two further cameras, and
 * end near a protected asset. The system should produce exactly one incident,
 * supported by events from three cameras, with the cross-camera hand-offs scored
 * rather than asserted.
 *
 * Site layout, metres east of the anchor (the road runs along north = 0):
 *
 *     0        30            70   150              260   300
 *     |--------|=============|----|-----------------|-----|   road (north 0)
 *              [ Restricted Zone A ]            [ Asset ]
 *          ^cam07            ^          ^cam08        ^cam09
 *      (sees 12-48)               (sees 132-168)  (sees 242-278)
 */

/** Site anchor. Arbitrary but real coordinates, so the map has something to show. */
export const SITE_ORIGIN: LatLon = { lat: 33.8938, lon: 35.5018 };

/**
 * Standard mast configuration.
 *
 * 6 m high, tilted 18 degrees down with a 40 degree vertical field of view. That
 * places the visible ground band from roughly 8 m to the stated range, which is
 * what a real perimeter camera is set up to do - and, importantly, means the
 * simulated actors are actually within view rather than in the blind foreground.
 */
const mast = (position: LatLon, headingDegrees: number): CameraPose => ({
  position: { ...position, altitude: 0 },
  mountHeight: 6,
  heading: headingDegrees,
  pitch: -18,
  roll: 0,
  horizontalFov: 70,
  verticalFov: 40,
  rangeMeters: 90,
});

export const CAM_07 = asId<CameraId>('cam-07');
export const CAM_08 = asId<CameraId>('cam-08');
export const CAM_09 = asId<CameraId>('cam-09');
export const CAM_10 = asId<CameraId>('cam-10');
export const CAM_11 = asId<CameraId>('cam-11');

export const ZONE_RESTRICTED_A = asId<ZoneId>('zone-restricted-a');
export const ZONE_PERIMETER = asId<ZoneId>('zone-perimeter');
export const ZONE_ASSET = asId<ZoneId>('zone-asset-substation');

export const RULE_RESTRICTED_ENTRY = asId<RuleId>('rule-restricted-entry');
export const RULE_LOITERING = asId<RuleId>('rule-loitering');
export const RULE_ASSET_APPROACH = asId<RuleId>('rule-asset-approach');

export const CAMERAS: readonly SimulatedCamera[] = Object.freeze([
  {
    id: CAM_07,
    name: 'Camera 07 - West Approach',
    pose: mast(offset(SITE_ORIGIN, 30, -25), 0),
    detectionRate: 0.92,
    falsePositiveRate: 0.5,
  },
  {
    id: CAM_08,
    name: 'Camera 08 - Mid Perimeter',
    pose: mast(offset(SITE_ORIGIN, 150, -25), 0),
    detectionRate: 0.9,
    falsePositiveRate: 0.5,
  },
  {
    id: CAM_09,
    name: 'Camera 09 - Substation',
    pose: mast(offset(SITE_ORIGIN, 260, -25), 0),
    detectionRate: 0.9,
    falsePositiveRate: 0.5,
  },
  {
    // Faces away from the road. Present to prove a quiet camera stays quiet:
    // a system that manufactures events on an empty view is worse than useless.
    id: CAM_10,
    name: 'Camera 10 - North Yard',
    pose: mast(offset(SITE_ORIGIN, 30, 60), 0),
    detectionRate: 0.9,
    falsePositiveRate: 0,
  },
  {
    id: CAM_11,
    name: 'Camera 11 - Loading Bay',
    pose: mast(offset(SITE_ORIGIN, 150, 60), 0),
    detectionRate: 0.9,
    falsePositiveRate: 0,
  },
]);

const zone = (
  id: ZoneId,
  name: string,
  purpose: Zone['purpose'],
  ring: readonly LatLon[],
): Zone => ({
  id,
  name,
  purpose,
  geometry: { kind: 'POLYGON', ring },
  locationId: null,
  parentZoneId: null,
  active: true,
  createdAt: utcMillis(0),
  updatedAt: utcMillis(0),
});

export const ZONES: readonly Zone[] = Object.freeze([
  zone(ZONE_RESTRICTED_A, 'Restricted Zone A', 'RESTRICTED', [
    offset(SITE_ORIGIN, 20, -12),
    offset(SITE_ORIGIN, 70, -12),
    offset(SITE_ORIGIN, 70, 12),
    offset(SITE_ORIGIN, 20, 12),
  ]),
  zone(ZONE_ASSET, 'Substation Compound', 'CRITICAL_ASSET', [
    offset(SITE_ORIGIN, 250, -15),
    offset(SITE_ORIGIN, 285, -15),
    offset(SITE_ORIGIN, 285, 15),
    offset(SITE_ORIGIN, 250, 15),
  ]),
  zone(ZONE_PERIMETER, 'Perimeter Road', 'PERIMETER', [
    offset(SITE_ORIGIN, -10, -20),
    offset(SITE_ORIGIN, 310, -20),
    offset(SITE_ORIGIN, 310, 20),
    offset(SITE_ORIGIN, -10, 20),
  ]),
]);

/**
 * The camera topology graph.
 *
 * Travel times are measured between the *edges of coverage*, not between the
 * masts: what matters for correlation is how long an object spends in the gap.
 * At a 1.4 m/s walking pace the 84 m gap from Camera 07 to Camera 08 takes about
 * a minute, and the 74 m gap on to Camera 09 about fifty seconds.
 */
export const TOPOLOGY: readonly CameraTopologyEdge[] = Object.freeze([
  {
    fromCameraId: CAM_07,
    toCameraId: CAM_08,
    distanceMeters: 84,
    minTravelSeconds: 25,
    expectedTravelSeconds: 60,
    maxTravelSeconds: 180,
    confidence: 0.9,
    bidirectional: true,
  },
  {
    fromCameraId: CAM_08,
    toCameraId: CAM_09,
    distanceMeters: 74,
    minTravelSeconds: 22,
    expectedTravelSeconds: 53,
    maxTravelSeconds: 160,
    confidence: 0.9,
    bidirectional: true,
  },
]);

const rule = (
  id: RuleId,
  name: string,
  when: Rule['when'],
  then: Rule['then'],
): Rule => ({
  id,
  name,
  enabled: true,
  when,
  then,
  createdAt: utcMillis(0),
  updatedAt: utcMillis(0),
});

export const RULES: readonly Rule[] = Object.freeze([
  rule(
    RULE_RESTRICTED_ENTRY,
    'Person in Restricted Zone A',
    {
      objectClasses: ['person'],
      zoneIds: [ZONE_RESTRICTED_A],
      cameraIds: [],
      timeWindow: null,
      direction: 'ANY',
      minDurationMillis: 0,
      confidenceThreshold: 0.55,
      cooldownMillis: 30_000,
      minTrackCount: 1,
    },
    { eventType: 'PersonEnteredZone', severity: 'HIGH', notify: true, createIncident: true },
  ),
  rule(
    RULE_LOITERING,
    'Person remaining in Restricted Zone A',
    {
      objectClasses: ['person'],
      zoneIds: [ZONE_RESTRICTED_A],
      cameraIds: [],
      timeWindow: null,
      direction: 'ANY',
      minDurationMillis: 10_000,
      confidenceThreshold: 0.55,
      cooldownMillis: 60_000,
      minTrackCount: 1,
    },
    { eventType: 'ObjectLoitering', severity: 'HIGH', notify: true, createIncident: true },
  ),
  rule(
    RULE_ASSET_APPROACH,
    'Person approaching the substation',
    {
      objectClasses: ['person'],
      zoneIds: [ZONE_ASSET],
      cameraIds: [],
      timeWindow: null,
      direction: 'ANY',
      minDurationMillis: 0,
      confidenceThreshold: 0.55,
      cooldownMillis: 30_000,
      minTrackCount: 1,
    },
    { eventType: 'PersonEnteredZone', severity: 'CRITICAL', notify: true, createIncident: true },
  ),
]);

/**
 * Three people walking east along the road at a normal pace.
 *
 * Laterally separated by two metres so they resolve as distinct tracks, and
 * staggered slightly in time the way a real group walks.
 */
export const groupOfThree = (walkSpeedMps = 1.4): readonly Actor[] =>
  [0, 1, 2].map((index) => {
    const north = (index - 1) * 2;
    const startDelay = index * 1.5;
    const distance = 300;

    return {
      id: `actor-${index + 1}`,
      objectClass: 'person',
      route: [
        { at: offset(SITE_ORIGIN, 0, north), timeSeconds: startDelay },
        {
          at: offset(SITE_ORIGIN, distance, north),
          timeSeconds: startDelay + distance / walkSpeedMps,
        },
      ],
    } satisfies Actor;
  });

export type Scenario = {
  readonly name: string;
  readonly description: string;
  readonly seed: number;
  readonly durationSeconds: number;
  readonly frameIntervalSeconds: number;
  readonly cameras: readonly SimulatedCamera[];
  readonly zones: readonly Zone[];
  readonly rules: readonly Rule[];
  readonly topology: readonly CameraTopologyEdge[];
  readonly actors: readonly Actor[];
  /** Scenario start, as a wall-clock UTC instant, so timelines read sensibly. */
  readonly startedAt: ReturnType<typeof utcMillis>;
};

/** 2024-06-12 at 02:14 UTC - the after-hours window the demo narrative uses. */
export const DEMO_START = utcMillis(Date.UTC(2024, 5, 12, 2, 14, 0));

export const perimeterIntrusionScenario = (seed = 20240612): Scenario => ({
  name: 'perimeter-intrusion',
  description:
    'Three people walk the perimeter road after hours, cross Restricted Zone A, ' +
    'pass through a coverage gap, and are reacquired near the substation.',
  seed,
  durationSeconds: 230,
  // 5 fps of inference: realistic for a multi-camera deployment, and enough for
  // the tracker to hold identity across the dropouts the sensors inject.
  frameIntervalSeconds: 0.2,
  cameras: CAMERAS,
  zones: ZONES,
  rules: RULES,
  topology: TOPOLOGY,
  actors: groupOfThree(),
  startedAt: DEMO_START,
});

/** A quiet site: no actors at all. Nothing should ever be raised. */
export const quietScenario = (seed = 1): Scenario => ({
  ...perimeterIntrusionScenario(seed),
  name: 'quiet-site',
  description: 'An empty site. The system must raise nothing.',
  durationSeconds: 120,
  actors: [],
});
