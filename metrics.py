"""Scoring logic for naija-asr-eval.

Word alignment, word error rate (WER), bootstrap confidence intervals and a
proper-noun (name) analysis. Pure Python + numpy: no audio or model code, so it
runs and is tested on any laptop.

Typical use:
    score = score_clip("my name is Sobande", "My name is show bandit.")
    score.counts.wer            # 0.5
    format_alignment(score.alignment)
"""

from __future__ import annotations

import math
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Sequence

import numpy as np

# Abbreviations whose trailing "." does not end a sentence, so the next word is
# not a sentence start. Lowercase, without the dot. Includes common Nigerian titles.
ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "rev", "hon", "gen", "col",
    "capt", "lt", "sgt", "sen", "gov", "engr", "alh", "barr", "pst", "arc", "mt",
})

# Title words are capitalised mid-sentence but are not names, so they are never
# counted as proper nouns ("Dr." is often transcribed as "doctor").
TITLE_WORDS = ABBREVIATIONS | frozenset({
    "mister", "missus", "miss", "madam", "doctor", "professor", "chief", "alhaji",
    "alhaja", "pastor", "engineer", "barrister", "honourable", "honorable",
    "reverend", "oba", "prince", "princess", "sir", "lady",
})

_SENTENCE_END = (".", "!", "?")
_CLOSING_CHARS = "\"')]}”’"  # quotes/brackets that may follow a full stop
_EDGE_PUNCTUATION = string.punctuation + "“”‘’–—"


# --------------------------------------------------------------------------- #
# Tokens and normalisation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Token:
    """One word of a transcript.

    raw:      the word exactly as written, e.g. "Sobande,"
    norm:     the normalised form used for matching, e.g. "sobande"
    position: index of the word in the whitespace-split original text, so a
              normalised word can always be traced back to its original casing.
    """
    raw: str
    norm: str
    position: int


def normalize_token(token: str) -> str:
    """Lowercase, fold accents (é -> e, ọ -> o) and remove punctuation.

    Applied to one word at a time, never to the whole string, so word positions
    in the reference stay aligned with the original (cased) text.
    """
    decomposed = unicodedata.normalize("NFKD", token.lower())
    # NFKD splits "é" into "e" + a combining accent mark; isalnum() drops the
    # mark, and also drops punctuation, apostrophes and hyphens.
    return "".join(ch for ch in decomposed if ch.isalnum())


def _clean_text(text: object) -> str:
    """Turn missing values (None, or NaN from an empty pandas CSV cell) into ""."""
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    if not isinstance(text, str):
        raise TypeError(f"Expected a transcript string, got {type(text).__name__}")
    return text


def tokenize(text: object) -> list[Token]:
    """Split on whitespace and normalise each word.

    Words that normalise to nothing (e.g. a lone "-" or "...") are dropped, but
    every kept Token remembers its original position.
    """
    tokens = []
    for position, raw in enumerate(_clean_text(text).split()):
        norm = normalize_token(raw)
        if norm:
            tokens.append(Token(raw=raw, norm=norm, position=position))
    return tokens


# --------------------------------------------------------------------------- #
# Proper-noun heuristic
# --------------------------------------------------------------------------- #

def _strip_edges(word: str) -> str:
    return word.strip(_EDGE_PUNCTUATION)


def _ends_sentence(raw_word: str) -> bool:
    """True if this word ends a sentence, so the next word is a sentence start."""
    word = raw_word.rstrip(_CLOSING_CHARS)
    if not word.endswith(_SENTENCE_END):
        return False
    if word.endswith("."):
        core = _strip_edges(word)
        if core.lower() in ABBREVIATIONS:
            return False          # "Mr." , "Dr." , "Engr."
        if len(core) == 1 and core.isupper():
            return False          # an initial, as in "J. K. Rowling"
    return True


def _is_capitalised(word: str) -> bool:
    first_letter = next((ch for ch in word if ch.isalpha()), "")
    return first_letter.isupper()


# AfriSpeech references often drop apostrophes: "Im happy that Im still progressing".
_PRONOUN_I_WITHOUT_APOSTROPHE = frozenset({"I", "Im", "Ive", "Ill", "Id"})


def _is_pronoun_i(core: str) -> bool:
    """"I" and its contractions, with a straight, curly or missing apostrophe."""
    return core in _PRONOUN_I_WITHOUT_APOSTROPHE or core[:2] in ("I'", "I’")


def find_proper_noun_positions(reference: object) -> set[int] | None:
    """Positions of words that are LIKELY proper nouns. A heuristic, not NER.

    A word counts if it is capitalised, does not start a sentence, is not the
    pronoun "I", is not a title (Mr., Chief, Doctor...) and is not a single-letter
    initial. A full stop after an abbreviation such as "Mr." does not start a
    new sentence.

    Known misses: names at the start of a sentence are never counted, and
    capitalised non-names ("Monday", "God", "English") are counted.

    Returns None when the transcript has no lowercase letters (e.g. LibriSpeech,
    which is ALL CAPS), because capitalisation then carries no information.
    """
    text = _clean_text(reference)
    if not any(ch.islower() for ch in text):
        return None

    positions = set()
    at_sentence_start = True
    for position, raw in enumerate(text.split()):
        core = _strip_edges(raw)
        if not normalize_token(raw):
            # Punctuation-only token such as "-" or a detached ".".
            if _ends_sentence(raw):
                at_sentence_start = True
            continue

        is_candidate = (
            _is_capitalised(core)
            and not at_sentence_start
            and not _is_pronoun_i(core)
            and core.lower().rstrip(".") not in TITLE_WORDS
            and len(normalize_token(core)) > 1
        )
        if is_candidate:
            positions.add(position)
        at_sentence_start = _ends_sentence(raw)
    return positions


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

MATCH, SUB, DEL, INS = "match", "sub", "del", "ins"


@dataclass(frozen=True)
class AlignedPair:
    """One step of the alignment. ref is None for insertions, hyp is None for deletions."""
    op: str
    ref: Token | None
    hyp: Token | None


def align(ref: Sequence[Token], hyp: Sequence[Token]) -> list[AlignedPair]:
    """Minimum-edit-distance (Levenshtein) word alignment.

    Several alignments can have the same minimum cost. Our tie-break walks left
    to right and prefers match > substitution > deletion > insertion, which
    keeps substitutions as early as possible:
        "Sobande" vs "show bandit"  ->  Sobande->show (sub), +bandit (ins)
    """
    n, m = len(ref), len(hyp)
    # cost[i][j] = fewest edits to turn ref[i:] into hyp[j:] (suffix costs, so
    # we can then walk forward from the start of both sentences).
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n:
                cost[i][j] = m - j                    # only insertions left
            elif j == m:
                cost[i][j] = n - i                    # only deletions left
            else:
                diagonal = cost[i + 1][j + 1] + (ref[i].norm != hyp[j].norm)
                cost[i][j] = min(diagonal, cost[i + 1][j] + 1, cost[i][j + 1] + 1)

    pairs = []
    i = j = 0
    while i < n or j < m:
        if i < n and j < m:
            same = ref[i].norm == hyp[j].norm
            if cost[i][j] == cost[i + 1][j + 1] + (not same):
                pairs.append(AlignedPair(MATCH if same else SUB, ref[i], hyp[j]))
                i += 1
                j += 1
                continue
        if i < n and (j == m or cost[i][j] == cost[i + 1][j] + 1):
            pairs.append(AlignedPair(DEL, ref[i], None))
            i += 1
        else:
            pairs.append(AlignedPair(INS, None, hyp[j]))
            j += 1
    return pairs


# --------------------------------------------------------------------------- #
# Error counts and per-clip scoring
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ErrorCounts:
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def ref_words(self) -> int:
        # Every reference word is either matched, substituted or deleted.
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        """errors / reference words. Can exceed 1.0 because insertions are unbounded."""
        if self.ref_words == 0:
            return math.nan
        return self.errors / self.ref_words

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        return ErrorCounts(
            self.hits + other.hits,
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
        )

    @classmethod
    def from_alignment(cls, pairs: Iterable[AlignedPair]) -> ErrorCounts:
        ops = Counter(pair.op for pair in pairs)
        return cls(ops[MATCH], ops[SUB], ops[DEL], ops[INS])


@dataclass(frozen=True)
class ProperNounResult:
    word: str        # the reference word, e.g. "Sobande"
    correct: bool    # True only if the model produced exactly this word here
    heard_as: str    # model words between the nearest correct neighbours, e.g. "show bandit"


@dataclass(frozen=True)
class ClipScore:
    counts: ErrorCounts
    alignment: tuple[AlignedPair, ...]
    # None means the heuristic does not apply (all-caps reference).
    proper_nouns: tuple[ProperNounResult, ...] | None


def _heard_as(pairs: Sequence[AlignedPair], index: int) -> str:
    """Model output between the nearest correctly matched words around pairs[index]."""
    if pairs[index].op == MATCH:
        return pairs[index].hyp.norm
    start = index
    while start > 0 and pairs[start - 1].op != MATCH:
        start -= 1
    end = index
    while end < len(pairs) - 1 and pairs[end + 1].op != MATCH:
        end += 1
    return " ".join(p.hyp.norm for p in pairs[start:end + 1] if p.hyp is not None)


def score_clip(reference: object, hypothesis: object) -> ClipScore:
    """Align one reference/hypothesis pair and compute its errors.

    An empty or missing hypothesis is valid: every reference word becomes a deletion.
    An empty reference is not: WER is undefined, so we raise instead of hiding it.
    """
    ref_tokens = tokenize(reference)
    if not ref_tokens:
        raise ValueError(f"Reference transcript has no words: {reference!r}")
    hyp_tokens = tokenize(hypothesis)
    pairs = align(ref_tokens, hyp_tokens)

    noun_positions = find_proper_noun_positions(reference)
    if noun_positions is None:
        proper_nouns = None
    else:
        proper_nouns = tuple(
            ProperNounResult(
                word=_strip_edges(pair.ref.raw),
                correct=pair.op == MATCH,
                heard_as=_heard_as(pairs, k),
            )
            for k, pair in enumerate(pairs)
            if pair.ref is not None and pair.ref.position in noun_positions
        )
    return ClipScore(ErrorCounts.from_alignment(pairs), tuple(pairs), proper_nouns)


def counts_with_normalizer(reference: object, hypothesis: object, normalize: Callable[[str], str]) -> ErrorCounts:
    """Error counts after a whole-sentence normaliser, e.g. Whisper's English normaliser.

    That normaliser rewrites across words ("thirty five" -> "35", "colour" -> "color"),
    so reference word positions are not preserved and no proper-noun analysis is
    possible here. Used only as a robustness check on WER. A reference that
    normalises to nothing gives ref_words == 0 (its insertions still count).
    """
    def words(text: object) -> list[Token]:
        return [Token(w, w, i) for i, w in enumerate(normalize(_clean_text(text)).split())]
    return ErrorCounts.from_alignment(align(words(reference), words(hypothesis)))


def format_alignment(pairs: Iterable[AlignedPair]) -> str:
    """Readable one-line alignment: 'my name is [sobande->show] [+bandit]'."""
    parts = []
    for p in pairs:
        if p.op == MATCH:
            parts.append(p.ref.norm)
        elif p.op == SUB:
            parts.append(f"[{p.ref.norm}->{p.hyp.norm}]")
        elif p.op == DEL:
            parts.append(f"[-{p.ref.norm}]")
        else:
            parts.append(f"[+{p.hyp.norm}]")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Corpus-level aggregation
# --------------------------------------------------------------------------- #

def total_counts(scores: Iterable[ClipScore]) -> ErrorCounts:
    return sum((s.counts for s in scores), ErrorCounts())


def corpus_wer(scores: Iterable[ClipScore]) -> float:
    """Total errors / total reference words. Long clips weigh more than short ones,
    unlike the mean of per-clip WERs, which lets a 3-word clip count as much as a
    40-word clip."""
    return total_counts(scores).wer


def proper_noun_counts(scores: Iterable[ClipScore]) -> tuple[int, int]:
    """(correct, total) likely proper nouns across clips where the heuristic applies."""
    correct = total = 0
    for s in scores:
        if s.proper_nouns is None:
            continue
        total += len(s.proper_nouns)
        correct += sum(r.correct for r in s.proper_nouns)
    return correct, total


def top_substitutions(scores: Iterable[ClipScore], n: int = 25) -> list[tuple[str, str, int]]:
    """Most common (reference word, model word, count) substitution pairs."""
    pairs = Counter(
        (p.ref.norm, p.hyp.norm)
        for s in scores for p in s.alignment if p.op == SUB
    )
    return [(ref, hyp, count) for (ref, hyp), count in pairs.most_common(n)]


def worst_clips(scored: Iterable[tuple[str, ClipScore]], n: int = 30) -> list[tuple[str, ClipScore]]:
    """The n clips with the highest WER (ties broken by more errors), from (clip_id, score) pairs."""
    return sorted(scored, key=lambda item: (item[1].counts.wer, item[1].counts.errors), reverse=True)[:n]


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Estimate:
    value: float
    ci_low: float
    ci_high: float


def _as_count_arrays(numerators: Sequence[float], denominators: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    if num.ndim != 1 or num.shape != den.shape:
        raise ValueError("numerators and denominators must be 1-D and the same length")
    if len(num) == 0:
        raise ValueError("need at least one clip")
    if den.sum() <= 0:
        raise ValueError("denominators sum to zero, so the ratio is undefined")
    return num, den


def _resampled_ratios(num: np.ndarray, den: np.ndarray, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    # Each row is one bootstrap resample: len(num) clip indices drawn with replacement.
    idx = rng.integers(0, len(num), size=(n_resamples, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num[idx].sum(axis=1) / den[idx].sum(axis=1)


def _interval(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2 * 100
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_ratio(
    numerators: Sequence[float],
    denominators: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Ratio of sums with a percentile bootstrap CI, resampling whole clips.

    With numerators = errors per clip and denominators = reference words per clip
    this is corpus WER. With wrong names / names per clip it is the name error rate.
    """
    num, den = _as_count_arrays(numerators, denominators)
    rng = np.random.default_rng(seed)
    samples = _resampled_ratios(num, den, n_resamples, rng)
    low, high = _interval(samples, confidence)
    return Estimate(float(num.sum() / den.sum()), low, high)


def bootstrap_wer(
    errors: Sequence[int],
    ref_words: Sequence[int],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Corpus WER with a 95% bootstrap CI over clips."""
    return bootstrap_ratio(errors, ref_words, n_resamples, confidence, seed)


def bootstrap_wer_difference(
    errors_a: Sequence[int],
    ref_words_a: Sequence[int],
    errors_b: Sequence[int],
    ref_words_b: Sequence[int],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """WER(a) - WER(b), e.g. Nigerian minus LibriSpeech, with a bootstrap CI.

    The groups contain different clips, so each is resampled independently.
    If the interval excludes 0, the gap is larger than resampling noise.
    """
    num_a, den_a = _as_count_arrays(errors_a, ref_words_a)
    num_b, den_b = _as_count_arrays(errors_b, ref_words_b)
    rng = np.random.default_rng(seed)
    diffs = _resampled_ratios(num_a, den_a, n_resamples, rng) - _resampled_ratios(num_b, den_b, n_resamples, rng)
    low, high = _interval(diffs, confidence)
    value = num_a.sum() / den_a.sum() - num_b.sum() / den_b.sum()
    return Estimate(float(value), low, high)


# --------------------------------------------------------------------------- #
# Speaker-level (cluster) bootstrap
# --------------------------------------------------------------------------- #

def sum_by_cluster(
    numerators: Sequence[float], denominators: Sequence[float], clusters: Sequence[Hashable]
) -> tuple[np.ndarray, np.ndarray]:
    """Add up numerators and denominators per cluster (e.g. per speaker)."""
    num, den = _as_count_arrays(numerators, denominators)
    labels = list(clusters)
    if len(labels) != len(num):
        raise ValueError("clusters must have one label per clip")
    _, cluster_index = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
    n_clusters = cluster_index.max() + 1
    return (np.bincount(cluster_index, weights=num, minlength=n_clusters),
            np.bincount(cluster_index, weights=den, minlength=n_clusters))


def bootstrap_wer_by_speaker(
    errors: Sequence[int],
    ref_words: Sequence[int],
    speakers: Sequence[Hashable],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Corpus WER with a CI that resamples whole speakers instead of clips.

    Clips from one speaker are not independent: a speaker who is hard to
    understand is hard in every clip. Resampling speakers keeps each speaker's
    clips together, so with few speakers the interval is honestly wider. The
    point estimate is unchanged, because the ratio of sums over speakers equals
    the ratio of sums over clips.
    """
    num, den = sum_by_cluster(errors, ref_words, speakers)
    return bootstrap_ratio(num, den, n_resamples, confidence, seed)


def bootstrap_wer_difference_by_speaker(
    errors_a: Sequence[int],
    ref_words_a: Sequence[int],
    speakers_a: Sequence[Hashable],
    errors_b: Sequence[int],
    ref_words_b: Sequence[int],
    speakers_b: Sequence[Hashable],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """WER(a) - WER(b) with each group's speakers resampled independently."""
    num_a, den_a = sum_by_cluster(errors_a, ref_words_a, speakers_a)
    num_b, den_b = sum_by_cluster(errors_b, ref_words_b, speakers_b)
    return bootstrap_wer_difference(num_a, den_a, num_b, den_b, n_resamples, confidence, seed)
