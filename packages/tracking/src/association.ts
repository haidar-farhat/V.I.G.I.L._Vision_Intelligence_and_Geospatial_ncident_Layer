import type {
  AssociationReason,
  CameraId,
  CameraTopologyEdge,
  Track,
  TrackAssociation,
} from '@sentinel/shared-types';
import { AssociationReasonCode } from '@sentinel/shared-types';
import { angleDifference } from '@sentinel/geometry';

/**
 * Cross-camera track association.
 *
 * When a track leaves camera A and something appears on camera B, this scores
 * the hypothesis that they are the same object. It never asserts identity: the
 * output is a score in 0..1 plus the reasons that produced it, and the UI shows
 * both together so an operator reads "87%, compatible route and travel time"
 * rather than a bare claim of sameness.
 *
 * The camera topology graph is what makes this tractable. Without an edge saying
 * "A connects to B in roughly 90 seconds", every pair of tracks in a time window
 * is equally plausible and the scores are noise.
 */

export type AssociationWeights = {
  readonly classMatch: number;
  readonly travelTime: number;
  readonly topologyPrior: number;
  readonly direction: number;
  readonly appearance: number;
};

/**
 * Default weights.
 *
 * Travel-time plausibility dominates because it is the most reliable signal
 * available without biometrics: physics constrains how fast a person can cross a
 * site, and that constraint holds regardless of lighting, pose or camera model.
 * Appearance is weighted lowest because it degrades badly across cameras with
 * different exposure and white balance, which is precisely when it would
 * otherwise be trusted most.
 */
export const DEFAULT_WEIGHTS: AssociationWeights = Object.freeze({
  classMatch: 0.3,
  travelTime: 0.3,
  topologyPrior: 0.2,
  direction: 0.1,
  appearance: 0.1,
});

export type AssociationOptions = {
  readonly weights?: AssociationWeights;
  /** Candidates scoring below this are discarded rather than shown. */
  readonly minScore?: number;
  /** Hard ceiling on the gap between tracks, regardless of topology. */
  readonly maxGapMillis?: number;
};

export const DEFAULT_MIN_SCORE = 0.45;
export const DEFAULT_MAX_GAP_MILLIS = 10 * 60 * 1000;

/** Cosine similarity, mapped from [-1, 1] onto [0, 1]. */
export const appearanceSimilarity = (
  a: readonly number[] | null,
  b: readonly number[] | null,
): number | null => {
  if (a === null || b === null || a.length === 0 || a.length !== b.length) return null;

  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    normA += x * x;
    normB += y * y;
  }
  if (normA === 0 || normB === 0) return null;

  const cosine = dot / (Math.sqrt(normA) * Math.sqrt(normB));
  return (Math.max(-1, Math.min(1, cosine)) + 1) / 2;
};

/**
 * How plausible a travel time is against a topology edge.
 *
 * 1.0 at the expected duration, tapering to 0 at the stated minimum and maximum.
 * Arriving faster than physically possible or later than the window allows both
 * score zero - the first is more suspicious than the second, but neither is
 * evidence that these are the same object.
 */
export const travelTimePlausibility = (edge: CameraTopologyEdge, gapSeconds: number): number => {
  if (gapSeconds < edge.minTravelSeconds || gapSeconds > edge.maxTravelSeconds) return 0;

  const expected = edge.expectedTravelSeconds;
  if (gapSeconds === expected) return 1;

  const span =
    gapSeconds < expected ? expected - edge.minTravelSeconds : edge.maxTravelSeconds - expected;
  if (span <= 0) return 1;

  return Math.max(0, 1 - Math.abs(gapSeconds - expected) / span);
};

export type TopologyLookup = (from: CameraId, to: CameraId) => CameraTopologyEdge | undefined;

/**
 * Build a lookup over a topology edge list, honouring bidirectional edges.
 */
export const topologyIndex = (edges: readonly CameraTopologyEdge[]): TopologyLookup => {
  const index = new Map<string, CameraTopologyEdge>();
  const key = (from: CameraId, to: CameraId): string => `${from}->${to}`;

  for (const edge of edges) {
    index.set(key(edge.fromCameraId, edge.toCameraId), edge);
    if (edge.bidirectional) {
      index.set(key(edge.toCameraId, edge.fromCameraId), {
        ...edge,
        fromCameraId: edge.toCameraId,
        toCameraId: edge.fromCameraId,
      });
    }
  }

  return (from, to) => index.get(key(from, to));
};

/**
 * Score one candidate association.
 *
 * Returns null when the hypothesis is impossible rather than merely weak: a
 * different object class, a negative time gap, or a gap beyond the hard ceiling.
 * Scoring those as "very low" rather than rejecting them would leave the operator
 * sifting through associations the system already knows are wrong.
 */
export const scoreAssociation = (
  from: Track,
  to: Track,
  topology: TopologyLookup,
  options: AssociationOptions = {},
): TrackAssociation | null => {
  const weights = options.weights ?? DEFAULT_WEIGHTS;
  const maxGap = options.maxGapMillis ?? DEFAULT_MAX_GAP_MILLIS;

  if (from.cameraId === to.cameraId) return null;
  if (from.objectClass !== to.objectClass) return null;

  const gapMillis = to.firstSeen - from.lastSeen;
  if (gapMillis < 0 || gapMillis > maxGap) return null;

  const reasons: AssociationReason[] = [];
  let score = 0;

  // --- object class -------------------------------------------------------
  score += weights.classMatch;
  reasons.push({
    code: AssociationReasonCode.ClassMatch,
    detail: `both tracks are ${from.objectClass}`,
    contribution: weights.classMatch,
  });

  // --- topology and travel time ------------------------------------------
  const edge = topology(from.cameraId, to.cameraId);
  const gapSeconds = gapMillis / 1000;

  if (edge === undefined) {
    reasons.push({
      code: AssociationReasonCode.TopologyEdgeUnknown,
      detail: 'no configured route between these cameras',
      contribution: 0,
    });
  } else {
    // The edge states the physical envelope of the route. A gap outside it means
    // the object could not have made this trip - too fast to walk, or so late it
    // is a different journey. That is a hard constraint, not a weak signal: a
    // candidate the system already knows is impossible must never reach an
    // operator's review queue with a plausible-looking score attached.
    if (gapSeconds < edge.minTravelSeconds || gapSeconds > edge.maxTravelSeconds) return null;

    const prior = weights.topologyPrior * edge.confidence;
    score += prior;
    reasons.push({
      code: AssociationReasonCode.TopologyEdgeKnown,
      detail: `configured route, ${Math.round(edge.distanceMeters)} m`,
      contribution: prior,
    });

    const plausibility = travelTimePlausibility(edge, gapSeconds);
    const contribution = weights.travelTime * plausibility;
    score += contribution;
    reasons.push({
      code:
        plausibility > 0
          ? AssociationReasonCode.TravelTimePlausible
          : AssociationReasonCode.TravelTimeImplausible,
      detail: `${gapSeconds.toFixed(1)} s elapsed, expected ~${edge.expectedTravelSeconds} s`,
      contribution,
    });
  }

  // --- direction of travel ------------------------------------------------
  if (from.headingDegrees !== null && to.headingDegrees !== null) {
    const delta = Math.abs(angleDifference(from.headingDegrees, to.headingDegrees));
    // Full credit when headings agree, tapering to zero at a right angle.
    const consistency = Math.max(0, 1 - delta / 90);
    const contribution = weights.direction * consistency;
    score += contribution;
    reasons.push({
      code:
        consistency > 0.5
          ? AssociationReasonCode.DirectionConsistent
          : AssociationReasonCode.DirectionInconsistent,
      detail: `headings differ by ${delta.toFixed(0)} degrees`,
      contribution,
    });
  }

  // --- appearance ---------------------------------------------------------
  const similarity = appearanceSimilarity(from.embedding, to.embedding);
  if (similarity !== null) {
    const contribution = weights.appearance * similarity;
    score += contribution;
    reasons.push({
      code:
        similarity > 0.5
          ? AssociationReasonCode.AppearanceSimilar
          : AssociationReasonCode.AppearanceDissimilar,
      detail: `appearance similarity ${(similarity * 100).toFixed(0)}%`,
      contribution,
    });
  }

  // Normalise against the weight that was actually applicable.
  //
  // The distinction that matters: a missing *signal* is normalised away, but a
  // missing *route* is not. A site with no appearance model installed, or tracks
  // too stationary to have a heading, should still be able to score highly on the
  // evidence it does have. But an unconfigured topology edge is evidence of
  // absence - there is no known way to get from A to B - so its weight stays in
  // the denominator and drags the score down. Normalising it away would make an
  // unknown route score *higher* than a known one, which is precisely backwards.
  const availableWeight =
    weights.classMatch +
    weights.topologyPrior +
    weights.travelTime +
    (from.headingDegrees !== null && to.headingDegrees !== null ? weights.direction : 0) +
    (similarity !== null ? weights.appearance : 0);

  const normalised = availableWeight <= 0 ? 0 : Math.min(1, score / availableWeight);
  const minScore = options.minScore ?? DEFAULT_MIN_SCORE;
  if (normalised < minScore) return null;

  return {
    fromTrackId: from.id,
    toTrackId: to.id,
    fromCameraId: from.cameraId,
    toCameraId: to.cameraId,
    departedAt: from.lastSeen,
    arrivedAt: to.firstSeen,
    score: normalised,
    reasons,
  };
};

/**
 * Best association for `arriving` among recently-departed tracks.
 *
 * One-to-one by construction: an arriving track gets at most one predecessor, so
 * a single object cannot be simultaneously matched to three departures and
 * inflate a group's apparent size.
 */
export const bestAssociation = (
  arriving: Track,
  departed: readonly Track[],
  topology: TopologyLookup,
  options: AssociationOptions = {},
): TrackAssociation | null => {
  let best: TrackAssociation | null = null;

  for (const candidate of departed) {
    const association = scoreAssociation(candidate, arriving, topology, options);
    if (association === null) continue;
    if (best === null || association.score > best.score) best = association;
  }

  return best;
};
