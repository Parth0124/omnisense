"""Stage 3 -- Language: what language `content.text` is in, or an honest `und`.

`docs/signal-model.md` §3.3 states the policy this module implements in two
sentences: detection runs on `content.text` **after cleaning**, and a detector
confidence below `LANGUAGE_CONFIDENCE_FLOOR` records `und` rather than a guess.
Both halves matter for a reason that is easy to under-rate.

**After cleaning, because markup is a language.** An HTML document is mostly
`div`, `class`, `href`, `src`, `nav` -- English-shaped ASCII tokens, in volume,
regardless of what the article says. Feed a detector the raw document and a
German news page reliably comes back as English with high confidence. The
detector is not wrong; it is describing the markup, which is why this stage sits
after stage 1 and reads the *cleaned* body rather than `ctx.raw_bytes`.

**`und` rather than a guess, because `und` is actionable.** A low-confidence
guess and a refusal look the same in the field and behave completely differently
downstream: §3.3 excludes `und` from language-filtered retrieval, whereas a
wrong-but-confident `en` puts a Portuguese review inside an English-only evidence
set, where an agent will quote it. `Language.detected()` in `models/signal.py`
implements the floor; this module is the call site it was written for. Note it
keeps the *measured* confidence on the `und` result rather than zeroing it --
that number is what makes the floor tunable later from stored data.

**Seeding is a correctness requirement, not a tidiness one.** `langdetect`
classifies by drawing random samples of n-grams from the text and applying random
priors across iterations (`Detector._detect_block`). With no seed those draws
come from whatever state the process-global `random` happens to be in, so the
same body classifies as `pt` on one worker and `es` on the next, and reprocessing
a Signal changes its stored language for no reason anyone can reconstruct.
`docs/signal-model.md` §5.1 promises that stages 1-3 "reproduce byte-identical
output for the same input and version" -- that promise is the whole basis for
`scripts/reindex.py` rebuilding derived stores from PostgreSQL without drift, and
an unseeded detector silently breaks it. `LangdetectDetector` therefore owns a
seeded `DetectorFactory` instead of touching the library's module-level default.

The detector is injected rather than imported at the call site, for the same
reason every model-bearing stage takes its provider as a constructor argument
(Design Doc §15): a test substitutes a five-line fake, and swapping `langdetect`
for a better classifier later touches one class, not the pipeline.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from importlib import metadata
from typing import Any, Final, Protocol, runtime_checkable

from models.enums import StageName
from models.signal import Language
from services.signal_engine.pipeline import EnrichmentContext

__all__ = [
    "DETECTOR_SEED",
    "MIN_ALPHA_CHARS",
    "LangdetectDetector",
    "LanguageDetector",
    "LanguageStage",
    "dominant_script",
    "normalize_language_code",
]


DETECTOR_SEED: Final = 0
"""The fixed seed handed to `langdetect`. Part of the stage contract.

Changing it changes stored languages for short and code-switched text across the
whole corpus, so it is a `pipeline_version` bump and a backfill -- exactly like
changing a model id -- not a tuning knob.
"""

MIN_ALPHA_CHARS: Final = 8
"""Below this many letters, detection is not attempted and the answer is `und`.

Not an arbitrary guard. `langdetect` builds a profile from character n-grams, and
on `"ok"`, `"+1"`, `"lol"` or a bare URL there are almost no n-grams to weigh --
so it returns whichever language has the highest prior *with high reported
probability*, which sails straight past the confidence floor. Counting letters
rather than characters is what makes an emoji-only or numeric body resolve here
instead of raising inside the detector.
"""


# --------------------------------------------------------------------------- #
# The detector seam
# --------------------------------------------------------------------------- #


@runtime_checkable
class LanguageDetector(Protocol):
    """A deterministic text -> language-probability function.

    `id` is written into `Language.detector` and is therefore stored per Signal.
    It carries the library version because reproducing a stored result means
    reproducing the *profile set* that produced it: `langdetect` ships its
    language profiles inside the wheel, so a version bump can change an answer
    without any code in this repository changing.
    """

    id: str

    def probabilities(self, text: str) -> Sequence[tuple[str, float]]:
        """Candidate languages, highest probability first. May be empty.

        Empty means "no opinion" and is a legitimate answer, distinct from
        raising -- which means the detector broke and is the stage's failure.
        """
        ...


def _langdetect_version() -> str:
    """Version of the installed `langdetect`, or a marker when it is unknown."""
    try:
        return metadata.version("langdetect")
    except metadata.PackageNotFoundError:  # pragma: no cover -- source checkout
        return "unknown"


class LangdetectDetector:
    """`langdetect`, pinned to a fixed seed and its own profile factory.

    Two things are deliberate here.

    **A private `DetectorFactory` rather than `langdetect.detect_langs`.** The
    module-level helper reads `DetectorFactory.seed`, a *class* attribute, so
    seeding it is a global mutation that any other importer can undo. An instance
    with its own seed and its own loaded profiles cannot be perturbed from
    elsewhere in the process.

    **Profiles are loaded lazily, once, under a lock.** Loading is ~55 JSON
    profiles off disk and costs tens of milliseconds; doing it at import would
    charge that to every process that merely imports the signal engine, including
    the API. The lock is not for the event loop -- `probabilities` is sync and
    never awaits -- it is for the thread pool a worker may drive stages from.

    Known wart, documented because it is invisible and will eventually surprise
    someone: `langdetect` implements its seeding by calling `random.seed()` on the
    **process-global** `random` module on every detection. Anything else in the
    process that samples from `random` (jitter on a retry backoff, reservoir
    sampling in a metric) therefore gets its stream reset once per Signal. Code
    that needs independent randomness must hold its own `random.Random` instance.
    """

    def __init__(self, *, seed: int = DETECTOR_SEED) -> None:
        self.id = f"langdetect/{_langdetect_version()}"
        self._seed = seed
        self._factory: Any | None = None
        self._lock = threading.Lock()

    def probabilities(self, text: str) -> Sequence[tuple[str, float]]:
        """Run the detector. Raises `LangDetectException` when it finds nothing.

        The exception is deliberately not caught. A stage that swallowed its own
        failure and returned a default would report `ok` to the pipeline and
        leave `lineage.stages[]` claiming an enrichment that never ran
        (`services/signal_engine/pipeline.py`). Stage 3 is degradable, so the
        pipeline turns the raise into `und` plus a recorded failure by itself --
        which is the same field value with an audit trail attached.
        """
        detector = self._new_detector()
        detector.append(text)
        return [(item.lang, float(item.prob)) for item in detector.get_probabilities()]

    def _new_detector(self) -> Any:
        """One `Detector` per call: `langdetect`'s detectors are single-use."""
        return self._ensure_factory().create()

    def _ensure_factory(self) -> Any:
        if self._factory is not None:
            return self._factory
        with self._lock:
            if self._factory is None:
                from langdetect.detector_factory import PROFILES_DIRECTORY, DetectorFactory

                factory = DetectorFactory()
                factory.load_profile(PROFILES_DIRECTORY)
                factory.seed = self._seed
                self._factory = factory
        return self._factory


# --------------------------------------------------------------------------- #
# Code and script normalization
# --------------------------------------------------------------------------- #

#: `langdetect` emits two tags that are not bare ISO 639-1. Both are legal BCP-47
#: once the region subtag is cased correctly, and casing is not cosmetic: a
#: retrieval filter comparing `language.code` to `"zh-CN"` silently matches
#: nothing if the stored value is `"zh-cn"`.
_REGION_TAGS: Final[dict[str, str]] = {"zh-cn": "zh-CN", "zh-tw": "zh-TW"}


def normalize_language_code(code: str) -> str:
    """Render a detector tag as BCP-47. Unknown shapes pass through lower-cased."""
    lowered = code.strip().lower()
    if not lowered:
        return "und"
    mapped = _REGION_TAGS.get(lowered)
    if mapped is not None:
        return mapped
    language, separator, region = lowered.partition("-")
    if separator and len(region) == 2:
        return f"{language}-{region.upper()}"
    return lowered


#: ISO 15924 codes for the ranges that actually appear in ingested text, in the
#: order they are tested. Deliberately incomplete: an unmatched character counts
#: toward nothing rather than toward a wrong script, because `script` is
#: advisory metadata and a wrong value is worse than an absent one.
_SCRIPT_RANGES: Final[tuple[tuple[int, int, str], ...]] = (
    (0x0041, 0x024F, "Latn"),
    (0x0370, 0x03FF, "Grek"),
    (0x0400, 0x052F, "Cyrl"),
    (0x0530, 0x058F, "Armn"),
    (0x0590, 0x05FF, "Hebr"),
    (0x0600, 0x06FF, "Arab"),
    (0x0750, 0x077F, "Arab"),
    (0x0900, 0x097F, "Deva"),
    (0x0980, 0x09FF, "Beng"),
    (0x0A00, 0x0A7F, "Guru"),
    (0x0A80, 0x0AFF, "Gujr"),
    (0x0B80, 0x0BFF, "Taml"),
    (0x0C00, 0x0C7F, "Telu"),
    (0x0C80, 0x0CFF, "Knda"),
    (0x0D00, 0x0D7F, "Mlym"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x10A0, 0x10FF, "Geor"),
    (0x1200, 0x137F, "Ethi"),
    (0x3040, 0x309F, "Kana"),
    (0x30A0, 0x30FF, "Kana"),
    (0x3400, 0x4DBF, "Hani"),
    (0x4E00, 0x9FFF, "Hani"),
    (0xAC00, 0xD7AF, "Hang"),
    (0x1100, 0x11FF, "Hang"),
)


def dominant_script(text: str) -> str | None:
    """Most-used ISO 15924 script in `text`, or `None` when there are no letters.

    Measured from the characters rather than inferred from the detected language,
    and the difference is the point: romanized Japanese is `ja` written in `Latn`,
    and a table keyed on the language code would confidently report `Jpan` for a
    body containing no Japanese characters at all. Reporting what is actually
    there keeps the field usable for the one thing it is for -- deciding whether a
    tokenizer or an OpenSearch analyzer can handle the body.

    Kana plus Han is reported as `Jpan`, the composite code, because Japanese
    prose mixes the two by definition and calling it `Hani` would route it to a
    Chinese analyzer.
    """
    counts: dict[str, int] = {}
    for character in text:
        if not character.isalpha():
            continue
        script = _script_of(ord(character))
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return None
    if "Kana" in counts and "Hani" in counts:
        counts["Jpan"] = counts.pop("Kana") + counts.pop("Hani")
    # Ties broken by the script name so the answer is stable across runs; a
    # dict-order tiebreak would depend on which character happened to come first.
    return max(sorted(counts), key=lambda name: counts[name])


def _script_of(codepoint: int) -> str | None:
    for start, end, script in _SCRIPT_RANGES:
        if start <= codepoint <= end:
            return script
    return None


def _alpha_count(text: str) -> int:
    """Letters only -- what a character-n-gram detector actually has to work with.

    Counted per character rather than by splitting on whitespace, because CJK
    prose has no spaces: a word count would read a perfectly detectable Japanese
    sentence as one token and send it to `und`.
    """
    return sum(1 for character in text if character.isalpha())


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


class LanguageStage:
    """Stage 3. Satisfies `Stage`; **degradable** -- failure leaves `und`.

    Holds one detector and no per-record state, so a single instance is shared
    across concurrent records as `SignalPipeline` requires.
    """

    name = StageName.LANGUAGE
    version = "1.0.0"

    def __init__(self, detector: LanguageDetector | None = None) -> None:
        self._detector = detector if detector is not None else LangdetectDetector()

    @property
    def model_id(self) -> str | None:
        """`None`: stage 3 is deterministic and calls no model.

        `langdetect` is a fixed n-gram table shipped in a wheel, not a model that
        can be re-served or upgraded independently, and `docs/signal-model.md`
        §5.1 records `model` only for the stages whose output cannot be
        reproduced without it. The detector's identity is not lost -- it is
        stored on the Signal as `language.detector`, which is where a consumer
        looking at a language would actually think to check.
        """
        return None

    @property
    def detector_id(self) -> str:
        """Identity of the detector in use, for startup logging."""
        return self._detector.id

    async def apply(self, ctx: EnrichmentContext) -> None:
        """Detect the language of the cleaned body and write `signal.language`.

        Two non-failures are handled here rather than allowed to raise, because
        both are *correct answers* rather than broken enrichment:

        - **Too little text.** A media-only post has an empty `content.text` by
          design (`Content.text` "may be empty for media-only posts"). Letting
          the detector raise `LangDetectException` on it would demote a perfectly
          good photo Signal to `partial` and dock its confidence for having no
          words in it.
        - **No opinion.** A detector that returns no candidates has answered:
          `und`, confidence 0.0.

        Everything else -- the detector throwing, the profiles failing to load --
        propagates. That is not this stage's call to make.
        """
        signal = ctx.require_signal()
        text = signal.content.text

        if _alpha_count(text) < MIN_ALPHA_CHARS:
            signal.language = Language(
                code="und",
                confidence=0.0,
                detector=self._detector.id,
                script=dominant_script(text),
            )
            return

        candidates = self._detector.probabilities(text)
        if not candidates:
            signal.language = Language(
                code="und", confidence=0.0, detector=self._detector.id, script=dominant_script(text)
            )
            return

        code, confidence = max(candidates, key=lambda item: item[1])
        # `Language.detected` -- not a hand-rolled comparison against the floor.
        # The policy lives in one place so that raising or lowering
        # LANGUAGE_CONFIDENCE_FLOOR cannot leave a second copy of it behind here.
        signal.language = Language.detected(
            normalize_language_code(code),
            _clamp_confidence(confidence),
            self._detector.id,
            dominant_script(text),
        )


def _clamp_confidence(value: float) -> float:
    """Force a probability into `[0.0, 1.0]`.

    `Score` is a validated range on the model, so a detector reporting 1.0000001
    -- floating-point noise from summing per-language probabilities -- would
    raise a `ValidationError` on assignment and turn a successful detection into
    a stage failure. Clamping is the honest fix: the value is a probability by
    construction, and the excess is representation error, not information.
    """
    if value != value:  # NaN: no ordering, so comparisons below would all be False.
        return 0.0
    return min(1.0, max(0.0, value))
