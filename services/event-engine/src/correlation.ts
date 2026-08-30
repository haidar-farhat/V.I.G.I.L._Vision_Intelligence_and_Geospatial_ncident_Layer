import { createHash } from 'node:crypto';
import type {
  CameraId,
  EventId,
  Incident,
  IncidentId,
  IncidentTimelineEntry,
  SecurityEvent,
  TrackAssociation,
  TrackId,
  UtcMillis,
  Zone,
  ZoneId,
} from '@sentinel/shared-types';
import { asId } from '@sentinel/shared-types';
import type { RiskContext } from './risk.ts';
import { assessRisk, distinctObjects } from './risk.ts';

/**
 * Event correlation.
 *
 * The single most important behaviour in the product: three cameras observing one
 * person must produce **one** incident, not three alerts. Alert fatigue is a
 * system failure mode, and a platform that raises an alert per detection is worse
 * than no platform at all, because it trains the operator to ignore it.
 *
 * An event joins an existing incident when it is plausibly the same situation:
 *
 *  - it shares a track with the incident (same object, same camera), or
 *  - one of its tracks is associated with a track already in the incident
 *    (the cross-camera hand-off), or
 *  - it happened close in time and touches a zone the incident already involves.
 *
 * Everything else opens a new incident. The window is bounded so an incident
 * cannot grow indefinitely and swallow a whole night's activity into one row.
 */

export type CorrelationOptions = {
  /** How long an incident stays open to new events after its last one. */
  readonly quietPeriodMillis?: number;
  /** Hard cap on an incident's total span, regardless of activity. */
  readonly maxSpanMillis?: number;
  /** Minimum association score for a cross-camera hand-off to merge incidents. */
  readonly minAssociationScore?: number;
};

export const DEFAULT_QUIET_PERIOD_MILLIS = 3 * 60 * 1000;
export const DEFAULT_MAX_SPAN_MILLIS = 30 * 60 * 1000;
export const DEFAULT_MIN_ASSOCIATION_SCORE = 0.6;

/** Mutable incident state held while an incident is still accepting events. */
type OpenIncident = {
  id: IncidentId;
  openedAt: UtcMillis;
  lastEventAt: UtcMillis;
  events: SecurityEvent[];
  trackIds: Set<TrackId>;
  cameraIds: Set<CameraId>;
  zoneIds: Set<ZoneId>;
  timeline: IncidentTimelineEntry[];
};

/**
 * Stable incident identity, derived from the first event that opened it.
 *
 * Deterministic so that replaying the same buffered events after a reconnect
 * reproduces the same incident rather than creating a parallel one.
 */
export const deriveIncidentId = (seed: SecurityEvent): IncidentId =>
  asId<IncidentId>(
    `INC-${createHash('sha256')
      .update(`${seed.nodeId} ${seed.cameraId} ${seed.id} ${seed.occurredAt}`)
      .digest('hex')
      .slice(0, 10)
      .toUpperCase()}`,
  );

export type CorrelationResult = {
  readonly incident: Incident;
  /** True when this event opened the incident rather than joining one. */
  readonly opened: boolean;
  /** Why the event joined, for the timeline and for debugging tuning. */
  readonly reason: CorrelationReason;
};

export const CorrelationReason = {
  NewSituation: 'NEW_SITUATION',
  SameTrack: 'SAME_TRACK',
  AssociatedTrack: 'ASSOCIATED_TRACK',
  SameZoneAndTime: 'SAME_ZONE_AND_TIME',
} as const;
export type CorrelationReason = (typeof CorrelationReason)[keyof typeof CorrelationReason];

/**
 * Disjoint-set over track ids linked by accepted associations.
 *
 * Association chains must be transitive. A group can walk from camera 07 to 08 to
 * 09 while camera 08 observes no zone and therefore produces no events - so its
 * track never enters an incident, and a pairwise-only check silently breaks the
 * chain, splitting one journey into two incidents. Union-find makes "is this the
 * same object we have been following?" a single lookup that survives any number
 * of intermediate cameras.
 */
class TrackGroups {
  readonly #parent = new Map<string, string>();

  find(id: string): string {
    const parent = this.#parent.get(id);
    if (parent === undefined) {
      this.#parent.set(id, id);
      return id;
    }
    if (parent === id) return id;

    const root = this.find(parent);
    this.#parent.set(id, root);
    return root;
  }

  union(a: string, b: string): void {
    const rootA = this.find(a);
    const rootB = this.find(b);
    if (rootA !== rootB) this.#parent.set(rootB, rootA);
  }

  connected(a: string, b: string): boolean {
    return this.find(a) === this.find(b);
  }
}

export class Correlator {
  readonly #open = new Map<IncidentId, OpenIncident>();
  readonly #options: Required<CorrelationOptions>;
  readonly #associations: TrackAssociation[] = [];
  readonly #groups = new TrackGroups();
  #riskContext: RiskContext;

  constructor(riskContext: RiskContext, options: CorrelationOptions = {}) {
    this.#riskContext = riskContext;
    this.#options = {
      quietPeriodMillis: options.quietPeriodMillis ?? DEFAULT_QUIET_PERIOD_MILLIS,
      maxSpanMillis: options.maxSpanMillis ?? DEFAULT_MAX_SPAN_MILLIS,
      minAssociationScore: options.minAssociationScore ?? DEFAULT_MIN_ASSOCIATION_SCORE,
    };
  }

  /** Zones may be added as the operator draws them. */
  setZones(zones: ReadonlyMap<ZoneId, Zone>): void {
    this.#riskContext = { ...this.#riskContext, zones };
  }

  /**
   * Record a cross-camera association.
   *
   * Supplied by the tracking layer, which scores hand-offs but does not decide
   * what they mean. Only associations at or above the threshold can merge two
   * incidents - a weak hypothesis is worth showing an operator, but not worth
   * silently restructuring their queue.
   */
  addAssociation(association: TrackAssociation): void {
    if (association.score >= this.#options.minAssociationScore) {
      this.#associations.push(association);
      this.#groups.union(String(association.fromTrackId), String(association.toTrackId));
    }
  }

  get openIncidentCount(): number {
    return this.#open.size;
  }

  /**
   * Feed one event and receive the incident it belongs to.
   *
   * Events must be supplied in non-decreasing time order per node, which the
   * worker buffer guarantees.
   */
  ingest(event: SecurityEvent): CorrelationResult {
    this.#expire(event.occurredAt);

    const match = this.#findIncident(event);

    if (match === null) {
      const incident = this.#openIncident(event);
      return {
        incident: this.#materialise(incident),
        opened: true,
        reason: CorrelationReason.NewSituation,
      };
    }

    const [open, reason] = match;
    this.#attach(open, event, reason);
    return { incident: this.#materialise(open), opened: false, reason };
  }

  /**
   * Find the incident this event belongs to.
   *
   * Checked in order of decreasing certainty: sharing a track is near-proof,
   * an association is a scored hypothesis, and zone-plus-time is circumstantial.
   */
  #findIncident(event: SecurityEvent): readonly [OpenIncident, CorrelationReason] | null {
    for (const open of this.#open.values()) {
      if (event.trackIds.some((id) => open.trackIds.has(id))) {
        return [open, CorrelationReason.SameTrack];
      }
    }

    for (const open of this.#open.values()) {
      for (const eventTrack of event.trackIds) {
        for (const openTrack of open.trackIds) {
          if (this.#groups.connected(String(eventTrack), String(openTrack))) {
            return [open, CorrelationReason.AssociatedTrack];
          }
        }
      }
    }

    for (const open of this.#open.values()) {
      const closeInTime = event.occurredAt - open.lastEventAt <= this.#options.quietPeriodMillis;
      const sharesZone = event.zoneIds.some((id) => open.zoneIds.has(id));
      if (closeInTime && sharesZone) return [open, CorrelationReason.SameZoneAndTime];
    }

    return null;
  }

  #openIncident(event: SecurityEvent): OpenIncident {
    const open: OpenIncident = {
      id: deriveIncidentId(event),
      openedAt: event.occurredAt,
      lastEventAt: event.occurredAt,
      events: [event],
      trackIds: new Set(event.trackIds),
      cameraIds: new Set([event.cameraId]),
      zoneIds: new Set(event.zoneIds),
      timeline: [
        {
          at: event.occurredAt,
          kind: 'EVENT',
          label: event.summary,
          eventId: event.id,
          cameraId: event.cameraId,
        },
      ],
    };

    this.#open.set(open.id, open);
    return open;
  }

  #attach(open: OpenIncident, event: SecurityEvent, reason: CorrelationReason): void {
    // Idempotent: a replayed event must not be counted twice.
    if (open.events.some((e) => e.id === event.id)) return;

    open.events.push(event);
    open.lastEventAt = Math.max(open.lastEventAt, event.occurredAt) as UtcMillis;
    for (const id of event.trackIds) open.trackIds.add(id);
    for (const id of event.zoneIds) open.zoneIds.add(id);
    open.cameraIds.add(event.cameraId);

    if (reason === CorrelationReason.AssociatedTrack) {
      open.timeline.push({
        at: event.occurredAt,
        kind: 'ASSOCIATION',
        label: `the same object appears to have been picked up by another camera`,
        eventId: null,
        cameraId: event.cameraId,
      });
    }

    open.timeline.push({
      at: event.occurredAt,
      kind: 'EVENT',
      label: event.summary,
      eventId: event.id,
      cameraId: event.cameraId,
    });
  }

  /** Close incidents that can no longer accept events. */
  #expire(now: UtcMillis): void {
    for (const [id, open] of this.#open) {
      const quiet = now - open.lastEventAt > this.#options.quietPeriodMillis;
      const tooLong = now - open.openedAt > this.#options.maxSpanMillis;
      if (quiet || tooLong) this.#open.delete(id);
    }
  }

  #materialise(open: OpenIncident): Incident {
    const events = [...open.events].sort((a, b) => a.occurredAt - b.occurredAt);
    const objectGroupOf = (trackId: TrackId): string => this.#groups.find(String(trackId));

    const risk = assessRisk(events, {
      ...this.#riskContext,
      assessedAt: open.lastEventAt,
      objectGroupOf,
    });

    const positioned = events.find((e) => e.position !== null);
    const distinctObjectCount = distinctObjects(events, objectGroupOf);

    return {
      id: open.id,
      title: titleFor(events, open.cameraIds.size, distinctObjectCount),
      severity: risk.severity,
      status: 'NEW',
      openedAt: open.openedAt,
      updatedAt: open.lastEventAt,
      closedAt: null,
      position: positioned?.position ?? null,
      cameraIds: [...open.cameraIds],
      zoneIds: [...open.zoneIds],
      trackIds: [...open.trackIds],
      distinctObjectCount,
      eventIds: events.map((e) => e.id),
      evidenceIds: [...new Set(events.flatMap((e) => e.evidenceIds))],
      risk,
      aiSummary: null,
      assessment: null,
      acknowledgedBy: null,
      acknowledgedAt: null,
    };
  }

  /** Every incident currently accepting events. */
  incidents(): readonly Incident[] {
    return [...this.#open.values()].map((open) => this.#materialise(open));
  }

  incident(id: IncidentId): Incident | undefined {
    const open = this.#open.get(id);
    return open === undefined ? undefined : this.#materialise(open);
  }

  timeline(id: IncidentId): readonly IncidentTimelineEntry[] {
    const open = this.#open.get(id);
    if (open === undefined) return [];
    return [...open.timeline].sort((a, b) => a.at - b.at);
  }

  /**
   * Associations recorded for an incident, for the investigation view.
   *
   * Includes hand-offs through cameras that produced no events of their own -
   * those are exactly the transitions the operator most needs to see, because
   * they are the part of the journey nothing else in the UI would show.
   */
  associationsFor(id: IncidentId): readonly TrackAssociation[] {
    const open = this.#open.get(id);
    if (open === undefined) return [];

    const roots = new Set<string>();
    for (const trackId of open.trackIds) roots.add(this.#groups.find(String(trackId)));

    return this.#associations.filter(
      (a) =>
        roots.has(this.#groups.find(String(a.fromTrackId))) ||
        roots.has(this.#groups.find(String(a.toTrackId))),
    );
  }

  eventsFor(id: IncidentId): readonly SecurityEvent[] {
    const open = this.#open.get(id);
    if (open === undefined) return [];
    return [...open.events].sort((a, b) => a.occurredAt - b.occurredAt);
  }
}

/**
 * A title an operator can triage from a list without opening it.
 *
 * States what happened and where, never who or why.
 */
export const titleFor = (
  events: readonly SecurityEvent[],
  cameraCount: number,
  objectCount: number,
): string => {
  const first = events[0];
  if (first === undefined) return 'Incident';

  // Counts objects, not track segments. One person walking past three cameras is
  // one person, and a title claiming otherwise is the first thing an operator
  // would notice was wrong.
  const subject =
    objectCount > 1
      ? `${objectCount} ${first.objectClass === 'person' ? 'people' : `${first.objectClass}s`}`
      : `a ${first.objectClass ?? 'object'}`;

  const detail = String(first.detail['zone'] ?? '');
  const where = detail === '' ? '' : ` in ${detail}`;
  const corroboration = cameraCount > 1 ? ` (${cameraCount} cameras)` : '';

  return `${capitalise(subject)}${where}${corroboration}`;
};

const capitalise = (value: string): string =>
  value.length === 0 ? value : value[0]!.toUpperCase() + value.slice(1);

/** Ids of every event in a list, for evidence bundles. */
export const eventIdsOf = (events: readonly SecurityEvent[]): readonly EventId[] =>
  events.map((e) => e.id);
