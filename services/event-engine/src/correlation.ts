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
import { assessRisk } from './risk.ts';

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

export class Correlator {
  readonly #open = new Map<IncidentId, OpenIncident>();
  readonly #options: Required<CorrelationOptions>;
  readonly #associations: TrackAssociation[] = [];
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
      for (const association of this.#associations) {
        const linksIn =
          event.trackIds.includes(association.toTrackId) &&
          open.trackIds.has(association.fromTrackId);
        const linksOut =
          event.trackIds.includes(association.fromTrackId) &&
          open.trackIds.has(association.toTrackId);

        if (linksIn || linksOut) return [open, CorrelationReason.AssociatedTrack];
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
    const risk = assessRisk(events, { ...this.#riskContext, assessedAt: open.lastEventAt });

    const positioned = events.find((e) => e.position !== null);

    return {
      id: open.id,
      title: titleFor(events, open.cameraIds.size),
      severity: risk.severity,
      status: 'NEW',
      openedAt: open.openedAt,
      updatedAt: open.lastEventAt,
      closedAt: null,
      position: positioned?.position ?? null,
      cameraIds: [...open.cameraIds],
      zoneIds: [...open.zoneIds],
      trackIds: [...open.trackIds],
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

  /** Associations recorded for an incident, for the investigation view. */
  associationsFor(id: IncidentId): readonly TrackAssociation[] {
    const open = this.#open.get(id);
    if (open === undefined) return [];
    return this.#associations.filter(
      (a) => open.trackIds.has(a.fromTrackId) || open.trackIds.has(a.toTrackId),
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
export const titleFor = (events: readonly SecurityEvent[], cameraCount: number): string => {
  const first = events[0];
  if (first === undefined) return 'Incident';

  const tracks = new Set(events.flatMap((e) => e.trackIds)).size;
  const subject =
    tracks > 1
      ? `${tracks} ${first.objectClass === 'person' ? 'people' : `${first.objectClass}s`}`
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
