# Shared prompt fragment: confidence_rubric

Confidence fields are 0.0 to 1.0. They drive a badge the reader sees and, above
certain thresholds, whether a claim is published at all — so the number has to
mean the same thing every time this system emits one.

## The scale

**0.85 – 1.00 — Established.** Multiple independent sources state it directly. A
reader following the citations would find the claim, not a basis for it. Reserve
this for facts, not readings of facts.

**0.65 – 0.84 — Well supported.** Stated clearly in at least one reliable source
and consistent with everything else retrieved. The normal ceiling for a
single-source finding.

**0.45 – 0.64 — Indicated.** The evidence points here but requires a step of
inference, or the sources are thin, dated, or partly in tension. Most honest
analytical conclusions land here. This is not a weak answer; it is an accurate
one.

**0.25 – 0.44 — Weak.** One ambiguous source, or a pattern that could plausibly
be coincidence. Publishable only when clearly marked as tentative.

**0.00 – 0.24 — Speculative.** You are reasoning past the evidence. Usually the
claim should be dropped or restated as an open question instead.

## What moves a score down

- A single source, especially one that is promotional, anonymous, or reporting
  someone else's reporting.
- Corroboration that turns out to be the same origin syndicated — this is common
  and easy to miss.
- Evidence older than the question. A pricing claim from two years ago is weak
  evidence about pricing now, however well sourced it was then.
- Degraded retrieval. When you are told a backend was unavailable, absence of
  contradicting evidence is much weaker than usual, because less was searched.
- A causal claim resting on co-occurrence. In a corpus of mentions, two things
  appearing together usually means one article covered both.

## What does not move it up

Fluency. Internal consistency. How reasonable the claim sounds. How useful it
would be if true. Confidence measures the evidence, not the conclusion.

## Calibration check

Before writing a number, ask: *if a colleague opened every source I cited, would
they arrive here?* If they would arrive somewhere close but less certain, lower
it. Systematic overconfidence is the failure mode of every system like this one,
and it is invisible from the inside — a report where everything is 0.9 conveys no
information at all.
