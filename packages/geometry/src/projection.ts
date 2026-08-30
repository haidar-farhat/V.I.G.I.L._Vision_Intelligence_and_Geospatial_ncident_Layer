import type {
  BoundingBox,
  CameraPose,
  LatLon,
  PositionEstimate,
  Vec2,
} from '@sentinel/shared-types';
import { PositionSource } from '@sentinel/shared-types';
import { clamp, normalizeDegrees, toDegrees, toRadians } from './vec.ts';
import { bearingDegrees, destinationPoint, haversineDistance } from './geodesy.ts';

/**
 * Image-space to ground-plane projection.
 *
 * Given a camera's pose and a point in the image, this answers "where on the
 * ground is that?" using a flat-earth, level-ground pinhole model. It is
 * deliberately the simplest model that is honest:
 *
 *  - it assumes the ground is a horizontal plane at the camera's mount height;
 *  - it ignores lens distortion (correctable later via `CameraIntrinsics`);
 *  - and it reports an uncertainty radius that grows the way the real error does.
 *
 * That last property is the important one. Projection error is dominated by
 * uncertainty in the ray's elevation angle, and because ground range is
 * `h / tan(theta)`, the error explodes as the ray flattens toward the horizon.
 * A detection 20 m away might be located to within a metre; the same camera's
 * detection near the horizon is uncertain by tens of metres. Rendering both as
 * identical dots on a map would be a lie, so the uncertainty is computed from the
 * actual derivative and travels with the position everywhere.
 */

/** Assumed angular error of a ray, combining detection jitter and pose error. */
export const DEFAULT_ANGULAR_UNCERTAINTY_DEG = 1.5;

/** Rays flatter than this are treated as unusable rather than projected. */
export const MIN_DEPRESSION_ANGLE_DEG = 0.5;

export type ProjectionOptions = {
  /** 1-sigma angular uncertainty in degrees. */
  readonly angularUncertaintyDeg?: number;
  /** Reject projections beyond the pose's stated range. */
  readonly enforceRange?: boolean;
};

/**
 * The point where an object meets the ground: the bottom edge, horizontal centre
 * of its bounding box. For a standing person or a vehicle this is the contact
 * patch, which is the only part of the box whose ground position is meaningful.
 */
export const groundContactPoint = (box: BoundingBox): Vec2 => ({
  x: box.x + box.w / 2,
  y: box.y + box.h,
});

/**
 * Ray direction for a normalised image point.
 *
 * Uses a rectilinear (tangent) model rather than a linear angle sweep, matching
 * how a real lens maps angle to sensor position. `u` and `v` are 0..1 with the
 * origin at the top-left of the frame.
 */
export const rayAngles = (
  pose: CameraPose,
  u: number,
  v: number,
): { readonly bearingDeg: number; readonly elevationDeg: number } => {
  const halfH = toRadians(pose.horizontalFov / 2);
  const halfV = toRadians(pose.verticalFov / 2);

  // Offsets from the image centre in the range -1..1.
  const dx = clamp(u, 0, 1) * 2 - 1;
  const dy = 1 - clamp(v, 0, 1) * 2;

  const yaw = Math.atan(dx * Math.tan(halfH));
  const pitchOffset = Math.atan(dy * Math.tan(halfV));

  return {
    bearingDeg: normalizeDegrees(pose.heading + toDegrees(yaw)),
    elevationDeg: pose.pitch + toDegrees(pitchOffset),
  };
};

export type GroundProjection = {
  readonly position: LatLon;
  readonly groundDistanceMeters: number;
  readonly bearingDeg: number;
  /** 1-sigma horizontal uncertainty in metres. */
  readonly uncertaintyMeters: number;
};

/**
 * Project a normalised image point onto the ground plane.
 *
 * Returns `null` when the ray cannot meet the ground: it points at or above the
 * horizon, or the intersection lies beyond the camera's useful range. Returning
 * null rather than a clamped guess is deliberate - a position the system cannot
 * actually determine must not appear on the map at all.
 */
export const projectToGround = (
  pose: CameraPose,
  u: number,
  v: number,
  options: ProjectionOptions = {},
): GroundProjection | null => {
  const { bearingDeg, elevationDeg } = rayAngles(pose, u, v);

  // Depression is positive when the ray points downward toward the ground.
  const depressionDeg = -elevationDeg;
  if (depressionDeg < MIN_DEPRESSION_ANGLE_DEG) return null;

  const depression = toRadians(depressionDeg);
  const groundDistance = pose.mountHeight / Math.tan(depression);
  if (!Number.isFinite(groundDistance) || groundDistance <= 0) return null;

  const enforceRange = options.enforceRange ?? true;
  if (enforceRange && groundDistance > pose.rangeMeters) return null;

  const sigmaAngle = toRadians(options.angularUncertaintyDeg ?? DEFAULT_ANGULAR_UNCERTAINTY_DEG);

  // d = h / tan(theta)  =>  |dd/dtheta| = h / sin^2(theta)
  const sinDepression = Math.sin(depression);
  const rangeSigma = (pose.mountHeight * sigmaAngle) / (sinDepression * sinDepression);
  // Lateral error is simply the arc subtended by the bearing uncertainty.
  const lateralSigma = groundDistance * sigmaAngle;

  return {
    position: destinationPoint(pose.position, bearingDeg, groundDistance),
    groundDistanceMeters: groundDistance,
    bearingDeg,
    uncertaintyMeters: Math.hypot(rangeSigma, lateralSigma),
  };
};

/**
 * Project a detection box to a map position, degrading gracefully.
 *
 * When the geometry does not permit a real projection the camera's own position
 * is returned instead, tagged `CAMERA_FALLBACK` with an uncertainty covering the
 * whole field of view. The operator still sees "something is happening at this
 * camera" - which is true - without the map implying a precision that does not
 * exist.
 */
export const projectDetection = (
  pose: CameraPose | null,
  box: BoundingBox,
  options: ProjectionOptions = {},
): PositionEstimate | null => {
  if (pose === null) return null;

  const contact = groundContactPoint(box);
  const projection = projectToGround(pose, contact.x, contact.y, options);

  if (projection === null) {
    return {
      point: pose.position,
      radiusMeters: pose.rangeMeters,
      source: PositionSource.CameraFallback,
    };
  }

  return {
    point: projection.position,
    radiusMeters: projection.uncertaintyMeters,
    source: PositionSource.GroundProjection,
  };
};

/**
 * Whether two position estimates are consistent with being the same object,
 * accounting for both uncertainties. Used by correlation, where treating an
 * uncertain position as exact would manufacture false associations.
 */
export const positionsOverlap = (a: PositionEstimate, b: PositionEstimate, sigmas = 2): boolean =>
  haversineDistance(a.point, b.point) <= (a.radiusMeters + b.radiusMeters) * sigmas;

/**
 * Nearest ground distance the camera can see: the bottom edge of the frame,
 * which points most steeply downward and therefore lands closest to the mast.
 */
export const nearGroundDistance = (pose: CameraPose): number => {
  const bottom = projectToGround(pose, 0.5, 1, { enforceRange: false });
  return bottom === null ? 0 : bottom.groundDistanceMeters;
};

/**
 * Farthest ground distance the camera can see: the top edge of the frame.
 *
 * Null when the top of the frame is above the horizon - the camera sees to
 * infinity in principle, and the pose's stated `rangeMeters` is what actually
 * bounds it.
 */
export const farGroundDistance = (pose: CameraPose): number | null => {
  const top = projectToGround(pose, 0.5, 0, { enforceRange: false });
  return top === null ? null : top.groundDistanceMeters;
};

/**
 * Ground-plane footprint of a camera's field of view.
 *
 * A downward-tilted camera does not see the ground at its own feet, so the true
 * footprint is an annular sector between the near and far ground distances, not
 * a pie slice from the mast. Drawing the pie slice would tell the operator the
 * camera covers ground it is physically blind to, which is exactly the kind of
 * false coverage assumption that gets a site burgled.
 *
 * The far edge is the lesser of the pose's stated range and where the top of the
 * frame actually meets the ground.
 */
export const fieldOfViewWedge = (pose: CameraPose, arcSegments = 24): readonly LatLon[] => {
  const segments = Math.max(2, Math.trunc(arcSegments));
  const halfFov = pose.horizontalFov / 2;

  const far = farGroundDistance(pose);
  const farRange = far === null ? pose.rangeMeters : Math.min(far, pose.rangeMeters);
  const nearRange = Math.min(nearGroundDistance(pose), farRange);

  const bearingAt = (t: number): number =>
    normalizeDegrees(pose.heading - halfFov + t * pose.horizontalFov);

  const points: LatLon[] = [];

  // Far arc, left to right.
  for (let i = 0; i <= segments; i += 1) {
    points.push(destinationPoint(pose.position, bearingAt(i / segments), farRange));
  }

  if (nearRange > 0.5) {
    // Near arc, right back to left, closing the annular sector.
    for (let i = segments; i >= 0; i -= 1) {
      points.push(destinationPoint(pose.position, bearingAt(i / segments), nearRange));
    }
  } else {
    // Camera effectively sees down to its own base: degenerate to a pie slice.
    points.push(pose.position);
  }

  return points;
};

/** Whether a bearing falls inside the camera's horizontal field of view. */
export const bearingInFov = (pose: CameraPose, bearingDeg: number): boolean => {
  const delta = Math.abs(
    ((normalizeDegrees(bearingDeg - pose.heading) + 180) % 360) - 180,
  );
  return delta <= pose.horizontalFov / 2;
};

/** Degrees of arc subtended by one metre at a given distance. Useful for sizing hints. */
export const angularSizeDeg = (sizeMeters: number, distanceMeters: number): number =>
  distanceMeters <= 0 ? 0 : toDegrees(2 * Math.atan(sizeMeters / (2 * distanceMeters)));

/**
 * The exact inverse of `projectToGround`: where does a point on the ground appear
 * in the image?
 *
 * Returns null when the point falls outside the frame or behind the camera.
 *
 * This is what allows the simulator to drive the *real* pipeline rather than a
 * parallel one. A synthetic person at a known position is projected into image
 * space here, handed to the tracker and zone engine as an ordinary detection, and
 * projected back onto the ground by the production code. Ground truth is known,
 * so the round-trip measures the pipeline's actual spatial error instead of
 * assuming it away - and any drift between the forward and inverse models shows
 * up immediately as a failing test.
 */
export type ImageCoordinate = {
  readonly u: number;
  readonly v: number;
  readonly distanceMeters: number;
  /** True when the point falls within the frame and the pose's range. */
  readonly inFrame: boolean;
};

/**
 * Image coordinates for a world point, **without** clipping to the frame.
 *
 * Coordinates outside 0..1 are meaningful and are returned as such: an object
 * standing half out of frame still has a well-defined position, and box
 * synthesis needs the unclipped value for the part that is off-screen. Callers
 * that want frame membership read `inFrame` or use `projectToImage`.
 */
export const imageCoordinates = (
  pose: CameraPose,
  point: LatLon,
  heightMeters = 0,
): ImageCoordinate | null => {
  const distance = haversineDistance(pose.position, point);
  if (distance <= 0) return null;

  const bearing = bearingDegrees(pose.position, point);
  const yawDeg = ((normalizeDegrees(bearing - pose.heading) + 180) % 360) - 180;
  const halfH = pose.horizontalFov / 2;

  // Beyond a quarter turn off-axis the tangent mapping is meaningless: the point
  // is beside or behind the camera, not merely out of frame.
  if (Math.abs(yawDeg) >= 90) return null;

  // Height above the ground plane raises the point in the frame.
  const elevationDeg = toDegrees(Math.atan2(heightMeters - pose.mountHeight, distance));
  const pitchOffsetDeg = elevationDeg - pose.pitch;
  const halfV = pose.verticalFov / 2;
  if (Math.abs(pitchOffsetDeg) >= 90) return null;

  // Invert the rectilinear (tangent) mapping used by rayAngles.
  const dx = Math.tan(toRadians(yawDeg)) / Math.tan(toRadians(halfH));
  const dy = Math.tan(toRadians(pitchOffsetDeg)) / Math.tan(toRadians(halfV));

  return {
    u: (dx + 1) / 2,
    v: (1 - dy) / 2,
    distanceMeters: distance,
    inFrame:
      Math.abs(yawDeg) <= halfH && Math.abs(pitchOffsetDeg) <= halfV && distance <= pose.rangeMeters,
  };
};

export const projectToImage = (
  pose: CameraPose,
  point: LatLon,
  heightMeters = 0,
): { readonly u: number; readonly v: number; readonly distanceMeters: number } | null => {
  const coordinate = imageCoordinates(pose, point, heightMeters);
  if (coordinate === null || !coordinate.inFrame) return null;
  return { u: coordinate.u, v: coordinate.v, distanceMeters: coordinate.distanceMeters };
};

/**
 * Synthesise the bounding box an object of a given real-world size would occupy.
 *
 * Both dimensions come from the same projective model - the corners of the
 * object's bounding volume are projected and the extent is measured in image
 * space. An earlier version mixed a projected height with an angular-size width
 * and fell back to the angular formula when the object's top left the frame,
 * which produced a discontinuity: boxes *grew* as the object receded past that
 * point. Anything sizing detections must use one model throughout.
 *
 * Note that on a steeply tilted camera the projected height is legitimately
 * non-monotonic with distance - perspective compression near the bottom of the
 * frame is severe - while width falls off cleanly. That is the real geometry,
 * not an artefact.
 */
export const boxForObject = (
  pose: CameraPose,
  point: LatLon,
  heightMeters: number,
  widthMeters: number,
): BoundingBox | null => {
  const base = imageCoordinates(pose, point, 0);
  if (base === null) return null;

  const top = imageCoordinates(pose, point, heightMeters);
  if (top === null) return null;

  // Width is measured by projecting the object's lateral extent, so it uses the
  // same tangent mapping as the height rather than a parallel approximation.
  const halfWidthBearing = toDegrees(Math.atan2(widthMeters / 2, base.distanceMeters));
  const left = imageCoordinates(
    pose,
    destinationPoint(pose.position, bearingDegrees(pose.position, point) - halfWidthBearing, base.distanceMeters),
    0,
  );
  const right = imageCoordinates(
    pose,
    destinationPoint(pose.position, bearingDegrees(pose.position, point) + halfWidthBearing, base.distanceMeters),
    0,
  );
  if (left === null || right === null) return null;

  const boxHeight = Math.abs(base.v - top.v);
  const boxWidth = Math.abs(right.u - left.u);

  return {
    x: Math.min(left.u, right.u),
    y: Math.min(base.v, top.v),
    w: boxWidth,
    h: boxHeight,
  };
};
