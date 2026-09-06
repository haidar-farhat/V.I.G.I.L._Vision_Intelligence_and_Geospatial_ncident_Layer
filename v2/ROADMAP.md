# What comes next: distance, things, and what they are doing together

The spine works: video in, tracks projected onto the ground, zones, rules,
incidents, evidence. What it cannot yet say is the sort of thing an operator
actually says out loud —

> *a car stopped eleven metres from the gate, three people got out, one of
> them is carrying something, and they are walking towards the door.*

Three capabilities stand between here and that sentence. They are listed in
the order they can be built, which is also the order of how much they can be
trusted. Each names what it needs, what it will refuse to claim, and how it
will be proven.

Nothing here is built yet unless [CAPABILITIES.md](CAPABILITIES.md) says so.
That file is generated from the manifest; this one is intent.

---

## 1 · Distance, said with its error

**What it adds.** Every measurement a person would reach for: how far an
object is from the camera, from a zone edge, from another object; how fast
the gap is closing; how long until it reaches the zone at the current speed.
On the plan, in the track table, in the incident's own words, and in the
exported report.

**Why it is first.** It needs no new model. The geometry is already there —
`project_point` returns a position *and* a 1-σ uncertainty, and the tracker
already measures ground speed over a window. This is arithmetic on facts the
system already has.

**What it must never do.** Print a distance without its error. Two positions
each known to ±1.4 m are eleven metres apart *give or take about two*, and a
plain "11 m" invites somebody to act on a precision nobody measured. Every
distance is carried as `(metres, ±metres)` and is rendered that way.

**Proven by.** Golden tests against hand-computed geometry; a camera run
where a person walks a measured path and the reported distance is checked
against a tape measure. That last one is the only real proof and it needs
somebody at the site with a tape.

## 2 · What things are doing together

**What it adds.** Relations between tracks, each with the measurement that
produced it:

| Relation | Read as | Rests on |
|---|---|---|
| `INSIDE` | three people **in** the car | a person's box is mostly within a vehicle's, held over many frames |
| `CARRIED` | a person **carrying** something | a small object's box overlaps a person's, off the ground, moving with them |
| `NEAR` | two people **together** | their ground positions are within a few metres, allowing for both errors |
| `APPROACHING` | walking **towards** the gate | the gap to a zone or an object is closing over a window |

**Why it is second.** It needs no new model either — it is geometry over the
tracks that already exist. It is what turns a list of boxes into a situation.

**What it must never do.** Call an overlap a fact. One camera cannot tell
*inside* from *in front of*: a person walking past a car occludes it exactly
as a person sitting in it does. So a relation is always `INFERRED`, never
`OBSERVED`; it carries the numbers it was drawn from ("0.78 of the person's
box lay within the vehicle's, for 21 frames"), and the interface says
"probably in the car", never "in the car". Two cameras, or a depth sensor,
would let it say more; until then it says less.

**Proven by.** Golden tests over synthetic box sequences for each relation
and each way it should *not* fire (a person passing in front of a car; a bag
on the ground beside somebody; two people who happen to be four metres
apart). Then a camera run with a real car and real people.

## 3 · Naming what a thing is, including a dangerous one

**What it adds.** A threat vocabulary: labels the site treats as dangerous,
each with a display name and a severity — so a detector that can say `knife`
produces an event that says *a person carrying a knife entered the yard* at
`CRITICAL`, rather than *an object entered the yard* at `MEDIUM`.

**The split that matters.** The *mechanism* — the vocabulary, the severity
escalation, the wording, the way a relation from §2 turns "a knife" into "a
person carrying a knife" — is buildable and testable now. The *detection* is
not: the model this ships with names the eighty COCO classes and **not one of
them is a weapon**. With those weights nothing will ever be labelled a
threat, and the console will say so rather than implying a capability the
site does not have.

**What it must never do.** Assert a weapon on anything but a detector's own
label and confidence. A false "armed" call is the most expensive mistake this
system can make: somebody will act on it. So the threat rule demands a higher
confidence floor than an ordinary class, requires the label to hold across
several frames, and every such event carries the model's digest and the
frames it was drawn from, so the claim can be checked afterwards by somebody
who was not there.

**Needs.** An operator-supplied model trained on the classes that site cares
about. The pipeline already accepts any ONNX model and reads its class names
from the file, so this is a matter of weights and a watch list, not of code.

**Proven by.** The mechanism, by tests with a stand-in model that names a
threat class. The detection, by an evaluation on real footage of the actual
weights a site intends to use — which is the operator's evidence to gather,
and which this system should help them record rather than assume.

## 4 · Telling one object from another across cameras

Appearance features, so the north gate and the yard can agree that the same
person walked between them. Today that rests on time and place alone and every
link says so. This is what makes §2's relations survive a camera change.

**Needs.** A small embedding model, and the same honesty: an appearance match
is a similarity with a threshold, not an identity.

## 5 · What is deliberately not on this list

- **Faces and plates.** Built in v1, deferred here ([DECISIONS.md](DECISIONS.md)
  D-08): each is a privacy switch, a model dependency and a thousand lines,
  and none of it is worth anything until the spine has run for an hour on a
  physical IP camera.
- **Behaviour classification** ("fighting", "loitering with intent"). The
  system can say what it measured — speed, dwell, proximity, direction — and
  a rule can act on that. It will not put a name on an intention.
- **Anything that needs the Internet.** Unchanged and non-negotiable.

---

### The order, and why

1 before 2 because a relation is measured in metres. 2 before 3 because *a
person carrying a knife* is a relation and a label together, and the relation
is the part that can be built now. 3's mechanism before 3's weights because
the mechanism can be tested and the weights are the operator's. 4 last
because it is the only one that needs a model this project would have to
choose, and choosing it blind is how a system ends up with a capability
nobody measured.
