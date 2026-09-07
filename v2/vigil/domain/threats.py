"""Labels a site treats as dangerous, and the rule that acts on them.

A false "armed" call is the most expensive mistake this system can make:
somebody will act on it. So this module is built to under-claim.

**The vocabulary is empty until a site fills it.** Nothing is a threat by
default. The model this product ships with names the eighty COCO classes, and
several of them — `knife`, `scissors`, `baseball bat` — are ordinary objects
in ordinary rooms. Treating them as weapons out of the box would mean a
kitchen raising a critical alert every evening, and an operator learning
within a week to ignore the word. `SUGGESTED` is a starting point a site
*opts into*, not a default.

**A threat claim carries more than an ordinary one.** It needs a higher
confidence than a normal detection, it must hold across several frames, and
the event records the model's own label, its digest and the conditions, so
somebody who was not there can check the claim afterwards.

**What it never does.** Infer a weapon from shape, posture or behaviour. The
only thing that can say `knife` is a detector trained to say `knife`, and
this module quotes it rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .events import Event, EventType, Rule, RuleContext, Severity
from .relations import RelationKind
from .zones import PresenceChange


@dataclass(frozen=True, slots=True)
class Threat:
    """One label a site treats as dangerous, and how much attention it deserves."""

    label: str
    display: str
    severity: Severity = Severity.HIGH

    def describe(self) -> str:
        return f"{self.display} ({self.severity})"


#: A starting point, not a default. A site adopts what applies to it, and only
#: for a model that can actually name those classes. The names are the labels
#: a detector emits, lower-cased.
SUGGESTED: tuple[Threat, ...] = (
    Threat("gun", "a gun", Severity.CRITICAL),
    Threat("pistol", "a pistol", Severity.CRITICAL),
    Threat("handgun", "a handgun", Severity.CRITICAL),
    Threat("rifle", "a rifle", Severity.CRITICAL),
    Threat("shotgun", "a shotgun", Severity.CRITICAL),
    Threat("firearm", "a firearm", Severity.CRITICAL),
    Threat("weapon", "a weapon", Severity.CRITICAL),
    Threat("knife", "a knife", Severity.HIGH),
    Threat("machete", "a machete", Severity.HIGH),
    Threat("axe", "an axe", Severity.HIGH),
    Threat("crowbar", "a crowbar", Severity.MEDIUM),
    Threat("baseball bat", "a baseball bat", Severity.MEDIUM),
)


class ThreatVocabulary:
    """What this site calls dangerous. Empty means nothing is."""

    def __init__(self, threats: Iterable[Threat] = ()):
        self._by_label: dict[str, Threat] = {t.label.strip().lower(): t for t in threats}

    def __bool__(self) -> bool:
        return bool(self._by_label)

    def __len__(self) -> int:
        return len(self._by_label)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_label))

    def of(self, label: str | None) -> Threat | None:
        return self._by_label.get((label or "").strip().lower())

    def describe(self) -> str:
        if not self._by_label:
            return "no label is treated as a threat on this site"
        return ", ".join(t.describe() for t in sorted(self._by_label.values(), key=lambda t: t.label))

    def unknown_to(self, model_labels: Sequence[str]) -> tuple[str, ...]:
        """Threat labels this model cannot produce, so a site is not told it is protected.

        A site that configures `knife` against a model naming only vehicles
        has configured nothing, and would never learn it from the absence of
        alerts.
        """
        known = {str(name).strip().lower() for name in model_labels}
        return tuple(label for label in self.labels if label not in known)

    @classmethod
    def from_labels(cls, labels: Iterable[str], *, catalogue: Mapping[str, Threat] | None = None) -> "ThreatVocabulary":
        """Build from bare label names, taking the severity from `SUGGESTED` when it knows one."""
        known = catalogue if catalogue is not None else {t.label: t for t in SUGGESTED}
        chosen = []
        for raw in labels:
            label = str(raw).strip().lower()
            if not label:
                continue
            chosen.append(known.get(label, Threat(label, f"a {label}", Severity.HIGH)))
        return cls(chosen)

    @classmethod
    def suggested(cls) -> "ThreatVocabulary":
        return cls(SUGGESTED)


class ThreatRule(Rule):
    """A dangerous thing was seen — and, when it can be told, who has it.

    Fires on presence rather than on every frame, so the zone's own entry
    hold applies before anything is claimed. When a relation says a person is
    carrying it, the sentence says so and the severity rises one step,
    because a knife on a bench and a knife in a hand are not the same fact.
    """

    id = "threat"
    description = "A label the site treats as dangerous was detected."
    event_type = EventType.ZONE_ENTRY
    severity = Severity.HIGH

    #: A threat claim needs more than an ordinary detection.
    min_confidence = 0.6
    #: And it must have held: one frame of a confident wrong label is still wrong.
    min_observations = 3

    def __init__(self, vocabulary: ThreatVocabulary | None = None):
        self._vocabulary = vocabulary or ThreatVocabulary()

    @property
    def vocabulary(self) -> ThreatVocabulary:
        return self._vocabulary

    def on_presence_change(self, change: PresenceChange, context: RuleContext) -> list[Event]:
        if not self._vocabulary or change.kind != "ENTERED" or context.track is None:
            return []
        events = self._for_track(context, context.class_label, change.presence.observations, direct=True)
        # The person is what entered; what they carry enters with them.
        for relation in context.relations:
            if relation.kind is RelationKind.CARRIED and relation.subject == context.track.id:
                carried = context.name_of(relation.object)
                events.extend(self._for_track(context, carried, relation.observations, direct=False,
                                              relation=relation))
        return events

    def _for_track(self, context: RuleContext, label: str | None, observations: int, *, direct: bool,
                   relation=None) -> list[Event]:
        threat = self._vocabulary.of(label)
        if threat is None:
            return []
        if not context.detector.classifies:
            # A detector that cannot classify cannot have said "knife".
            return []
        confidence = context.track.confidence if context.track is not None else 0.0
        if confidence < self.min_confidence or observations < self.min_observations:
            return []
        where = f" in {context.zone.name}" if context.zone is not None else ""
        if direct:
            summary = f"{threat.display.capitalize()} was detected{where}"
            severity = threat.severity
        else:
            summary = f"Somebody is carrying {threat.display}{where}"
            severity = _one_step_up(threat.severity)
        conditions = [
            f"the detector labelled it {label!r} at {confidence:.2f}, above the {self.min_confidence:.2f} "
            "a threat claim requires",
            f"it held for {observations} observations, above the {self.min_observations} a threat claim requires",
            f"this site treats {label!r} as {threat.describe()}",
        ]
        if relation is not None:
            conditions.extend(relation.conditions)
            conditions.append("who is carrying it is inferred from one camera and may be wrong")
        return [self._build(context, summary=summary, conditions=tuple(conditions), confidence=confidence,
                            observations=observations, severity=severity)]


def _one_step_up(severity: Severity) -> Severity:
    order = list(Severity)
    return order[min(len(order) - 1, order.index(severity) + 1)]
