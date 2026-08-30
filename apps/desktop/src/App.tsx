import { useMemo, useState } from 'react';
import { SiteMap } from './components/SiteMap.tsx';
import { IncidentAnalysis } from './components/IncidentAnalysis.tsx';
import { MapEditor } from './components/MapEditor.tsx';
import type { Snapshot, SnapshotIncident } from './data/types.ts';
import rawSnapshot from './data/snapshot.json';
import './styles/app.css';

/**
 * The command centre.
 *
 * Laid out so an operator can answer, without clicking anything: what is
 * happening, where, when, which cameras saw it, how serious it is, and why the
 * system decided that. The last of those is the one most products omit.
 *
 * Every number on this screen comes from a real run of the pipeline, not from a
 * fixture. See scripts/snapshot.mjs.
 */

const snapshot = rawSnapshot as unknown as Snapshot;

const SECTIONS = [
  'COMMAND',
  'CAMERAS',
  'MAP',
  'EVENTS',
  'INCIDENTS',
  'TRACKS',
  'RECORDINGS',
  'NODES',
  'RULES',
  'SYSTEM',
] as const;

const pad = (n: number): string => String(n).padStart(2, '0');

/** UTC, because that is what is stored. A real deployment renders operator-local. */
export const clock = (ms: number): string => {
  const d = new Date(ms);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
};

export const App = (): JSX.Element => {
  const [section, setSection] = useState<(typeof SECTIONS)[number]>('COMMAND');
  const [selectedIncidentId, setSelectedIncidentId] = useState<string | null>(
    snapshot.incidents[0]?.id ?? null,
  );

  const selected: SnapshotIncident | undefined = useMemo(
    () => snapshot.incidents.find((i) => i.id === selectedIncidentId),
    [selectedIncidentId],
  );

  const events = useMemo(
    () => [...snapshot.events].sort((a, b) => b.occurredAt - a.occurredAt),
    [],
  );

  const cameraName = useMemo(
    () => new Map(snapshot.cameras.map((c) => [c.id, c.name])),
    [],
  );

  return (
    <div className="shell">
      <header className="titlebar">
        <div className="brand">
          SENTINEL<span>·</span>VISION
        </div>

        <nav className="nav">
          {SECTIONS.map((name) => (
            <button
              key={name}
              type="button"
              aria-current={section === name}
              onClick={() => setSection(name)}
            >
              {name}
            </button>
          ))}
        </nav>

        <div className="titlebar-right">
          <span>STANDALONE</span>
          <span>WAN BLOCKED</span>
          <span className="secure">
            <i className="dot" /> SYSTEM SECURE
          </span>
        </div>
      </header>

      {section === 'MAP' ? (
        <MapEditor snapshot={snapshot} />
      ) : section === 'COMMAND' ? (
        <main className="command">
          {/* ---------------------------------------------------- cameras */}
          <section className="panel">
            <div className="panel-head">
              Live cameras
              <span className="count">
                {snapshot.cameras.filter((c) => c.reporting).length}/{snapshot.cameras.length}
              </span>
            </div>
            <div className="panel-body">
              {snapshot.cameras.map((camera) => (
                <div className="camera" key={camera.id}>
                  <i className={`state${camera.reporting ? '' : ' idle'}`} />
                  <div>
                    <div className="name">{camera.name}</div>
                    <div className="meta">
                      {camera.id} · {camera.horizontalFov}° fov · {camera.rangeMeters} m
                    </div>
                  </div>
                  <div className="tracks">{camera.trackCount > 0 ? camera.trackCount : '—'}</div>
                </div>
              ))}

              <div className="section">
                <h3>Pipeline</h3>
                <div className="kv">
                  <span>frames</span>
                  <span>{snapshot.pipeline.frames.toLocaleString()}</span>
                </div>
                <div className="kv">
                  <span>detections</span>
                  <span>{snapshot.pipeline.detections.toLocaleString()}</span>
                </div>
                <div className="kv">
                  <span>mean position error</span>
                  <span>{snapshot.pipeline.positionError.meanMeters.toFixed(2)} m</span>
                </div>
                <div className="kv">
                  <span>within stated 2σ</span>
                  <span>
                    {(snapshot.pipeline.positionError.withinStatedUncertainty * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="note">
                  Positions are projected onto the ground plane and carry the
                  uncertainty that projection actually implies. An uncalibrated
                  camera produces a wide ellipse, never a confident dot.
                </p>
              </div>
            </div>
          </section>

          {/* -------------------------------------------------------- map */}
          <section className="panel">
            <div className="panel-head">
              Site map
              <span className="count">
                {snapshot.cameras.length} cameras · {snapshot.zones.length} zones
              </span>
            </div>
            <SiteMap snapshot={snapshot} selectedIncidentId={selectedIncidentId} />
          </section>

          {/* -------------------------------------------------- incidents */}
          <section className="panel">
            <div className="panel-head">
              Incidents
              <span className="count">{snapshot.incidents.length}</span>
            </div>
            <div className="panel-body">
              {snapshot.incidents.length === 0 ? (
                <div className="empty">
                  No open incidents.
                  <br />
                  The site is quiet.
                </div>
              ) : (
                snapshot.incidents.map((incident) => (
                  <button
                    type="button"
                    className="incident"
                    key={incident.id}
                    aria-selected={incident.id === selectedIncidentId}
                    onClick={() => setSelectedIncidentId(incident.id)}
                  >
                    <div className="incident-top">
                      <span className={`badge ${incident.severity}`}>{incident.severity}</span>
                      <span className="incident-id">{incident.id}</span>
                    </div>
                    <div className="incident-title">{incident.title}</div>
                    <div className="incident-meta">
                      {clock(incident.openedAt)} UTC · {incident.eventCount} events ·{' '}
                      {incident.cameraIds.length} cameras · risk {incident.risk.score}
                    </div>
                  </button>
                ))
              )}

              <div className="section">
                <h3>Why one incident</h3>
                <p className="note">
                  {snapshot.events.length} events across {snapshot.cameras.filter((c) => c.reporting).length}{' '}
                  cameras were correlated into {snapshot.incidents.length} incident.
                  The same {snapshot.incidents[0]?.distinctObjectCount ?? 0} people
                  produced {snapshot.incidents[0]?.trackSegmentCount ?? 0} track
                  segments; the system counts people, not segments.
                </p>
              </div>
            </div>
          </section>

          {/* ----------------------------------------------------- events */}
          <section className="panel">
            <div className="panel-head">
              Priority events
              <span className="count">{events.length}</span>
            </div>
            <div className="panel-body">
              {events.map((event) => (
                <button type="button" className="event" key={event.id}>
                  <i className={`sev ${event.severity}`} />
                  <span className="time">{clock(event.occurredAt)}</span>
                  <span>
                    <span className="what">{event.summary}</span>
                    <br />
                    <span className="where">
                      {cameraName.get(event.cameraId) ?? event.cameraId} ·{' '}
                      {(event.confidence * 100).toFixed(0)}% confidence
                    </span>
                  </span>
                </button>
              ))}
            </div>
          </section>

          {/* --------------------------------------------------- timeline */}
          <section className="panel">
            <div className="panel-head">
              Incident timeline
              <span className="count">{selected?.id ?? '—'}</span>
            </div>
            <div className="panel-body pad">
              {selected === undefined ? (
                <div className="empty">Select an incident.</div>
              ) : (
                selected.timeline.map((entry, index) => (
                  <div className="timeline-row" key={`${entry.at}-${index}`}>
                    <span className="t">{clock(entry.at)}</span>
                    <span className={`k ${entry.kind}`}>{entry.kind}</span>
                    <span className="l">{entry.label}</span>
                  </div>
                ))
              )}
            </div>
          </section>

          {/* --------------------------------------------------- analysis */}
          <section className="panel">
            <div className="panel-head">
              Assessment
              <span className="count">{selected === undefined ? '—' : selected.severity}</span>
            </div>
            <div className="panel-body pad">
              {selected === undefined ? (
                <div className="empty">Select an incident.</div>
              ) : (
                <IncidentAnalysis incident={selected} />
              )}
            </div>
          </section>
        </main>
      ) : (
        <main className="command" style={{ gridTemplateColumns: '1fr', gridTemplateRows: '1fr' }}>
          <section className="panel">
            <div className="panel-head">{section}</div>
            <div className="panel-body">
              <div className="empty">
                <strong>{section}</strong> is designed but not built.
                <br />
                <br />
                See STATUS.md for what is implemented and what is not. This screen
                exists so the gap is visible rather than hidden behind a plausible
                mock-up.
              </div>
            </div>
          </section>
        </main>
      )}

      <footer className="statusbar">
        <span>
          <b>MODE</b> STANDALONE
        </span>
        <span>
          <b>WAN</b> BLOCKED / NOT REQUIRED
        </span>
        <span>
          <b>CLOUD</b> DISABLED
        </span>
        <span>
          <b>TELEMETRY</b> DISABLED
        </span>
        <span>
          <b>SCENARIO</b> {snapshot.scenario.name} (seed {snapshot.scenario.seed})
        </span>
        <span style={{ marginLeft: 'auto' }}>
          <b>SNAPSHOT</b> {new Date(snapshot.generatedAt).toISOString().slice(0, 19)}Z
        </span>
      </footer>
    </div>
  );
};
