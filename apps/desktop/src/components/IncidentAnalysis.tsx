import type { SnapshotIncident } from '../data/types.ts';

/**
 * The assessment panel.
 *
 * Two things are shown that most products hide:
 *
 * 1. **Why the risk score is what it is.** Every point is attributable to a stated
 *    reason. A number nobody can account for invites either blind trust or blanket
 *    dismissal, and both are failures.
 *
 * 2. **Which AI statements are observation and which are inference**, with the
 *    evidence each rests on and the model and prompt version that produced them.
 *    An operator has to be able to tell recorded fact from the model's reasoning
 *    at a glance, so the partition is structural rather than a matter of wording.
 */

type Props = {
  readonly incident: SnapshotIncident;
};

export const IncidentAnalysis = ({ incident }: Props): JSX.Element => {
  const report = incident.report;

  return (
    <div className="analysis">
      <h4>Risk</h4>

      <div className="risk-total">
        <span className="n">{incident.risk.score}</span>
        <span className="of">/ 100 · {incident.risk.severity}</span>
      </div>

      {incident.risk.contributions.map((contribution, index) => (
        <div className="risk-row" key={`${contribution.code}-${index}`}>
          <span className={`pts${contribution.points < 0 ? ' neg' : ''}`}>
            {contribution.points >= 0 ? '+' : ''}
            {contribution.points}
          </span>
          <span className="why">
            <b>{contribution.code.replace(/_/g, ' ')}</b>
            {contribution.detail}
          </span>
        </div>
      ))}

      {report === null ? (
        <>
          <h4>Analysis</h4>
          <div className="empty">No analysis has been generated for this incident.</div>
        </>
      ) : report.insufficientEvidence ? (
        <>
          <h4>Analysis</h4>
          <div className="summary">Insufficient evidence.</div>
          <p className="note">
            The analyst declined rather than speculating. That is the correct
            behaviour, not a failure.
          </p>
        </>
      ) : (
        <>
          <h4>Analysis</h4>
          <div className="summary">{report.summary}</div>

          <h4>Observed</h4>
          {report.observed.map((claim, index) => (
            <div className="claim" key={`observed-${index}`}>
              <p>{claim.text}</p>
              <span className="cite">evidence: {claim.evidence.join(', ')}</span>
              <span className="conf">{(claim.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}

          <h4>Inferred</h4>
          {report.inferred.map((claim, index) => (
            <div className="claim" key={`inferred-${index}`}>
              <p>{claim.text}</p>
              <span className="cite">
                {claim.evidence.length === 0 ? 'derived from the events above' : `evidence: ${claim.evidence.join(', ')}`}
              </span>
              <span className="conf">{(claim.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}

          <h4>Unknown</h4>
          <ul className="unknown">
            {report.unknown.map((item, index) => (
              <li key={`unknown-${index}`}>{item}</li>
            ))}
          </ul>

          <h4>For the operator</h4>
          <ul className="questions">
            {report.operatorQuestions.map((question, index) => (
              <li key={`question-${index}`}>{question}</li>
            ))}
          </ul>

          <div className="provenance">
            model {report.modelId}
            <br />
            prompt {report.promptVersion}
            <br />
            {incident.distinctObjectCount} objects · {incident.trackSegmentCount} track segments ·{' '}
            {incident.eventCount} events · {incident.cameraIds.length} cameras
            <br />
            Every statement above is bound to evidence that was supplied to the
            analyst. Reports citing anything else are rejected before display.
          </div>
        </>
      )}
    </div>
  );
};
