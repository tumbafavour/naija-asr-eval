"""Unit tests for metrics.py. Run with:  pytest -v"""

import math
import random

import pytest

from metrics import (
    DEL, INS, MATCH, SUB,
    ErrorCounts, align, bootstrap_ratio, bootstrap_wer, bootstrap_wer_difference,
    corpus_wer, find_proper_noun_positions, format_alignment, normalize_token,
    proper_noun_counts, score_clip, tokenize, top_substitutions, worst_clips,
)


# ----------------------------------------------------------------- required cases

def test_sobande_example_one_substitution_one_insertion():
    score = score_clip("my name is Sobande", "my name is show bandit")
    assert score.counts.wer == pytest.approx(0.5)
    assert score.counts.substitutions == 1
    assert score.counts.insertions == 1
    assert score.counts.deletions == 0
    assert score.counts.hits == 3
    assert format_alignment(score.alignment) == "my name is [sobande->show] [+bandit]"


def test_wer_can_exceed_100_percent():
    # A typical Whisper hallucination: the model repeats itself.
    score = score_clip("thank you", "thank you thank you thank you so much")
    assert score.counts.insertions == 6
    assert score.counts.wer == pytest.approx(3.0)
    assert score.counts.wer > 1.0


def test_corpus_wer_weights_by_length_not_mean_of_clips():
    short_bad = score_clip("hello", "yellow")                       # 1 error / 1 word = 100%
    long_good = score_clip("one two three four five six seven eight nine",
                           "one two three four five six seven eight nine")  # 0 / 9
    assert corpus_wer([short_bad, long_good]) == pytest.approx(1 / 10)
    mean_of_clips = (short_bad.counts.wer + long_good.counts.wer) / 2
    assert mean_of_clips == pytest.approx(0.5)                      # what we must NOT report


def test_word_after_mr_dr_mrs_is_not_a_sentence_start():
    ref = "I met Mr. Okafor and Dr. Bello. Then Mrs. Eze arrived."
    positions = find_proper_noun_positions(ref)
    words = ref.split()
    found = {words[p] for p in positions}
    assert found == {"Okafor", "Bello.", "Eze"}
    # "Then" follows a real full stop, so it is a sentence start, not a name.
    assert "Then" not in found
    # Titles themselves are not names.
    assert not found & {"Mr.", "Dr.", "Mrs."}


def test_empty_model_output_counts_every_reference_word_as_deletion():
    for empty in ["", "   ", None, float("nan")]:
        score = score_clip("the quick brown fox", empty)
        assert score.counts.deletions == 4
        assert score.counts.substitutions == score.counts.insertions == 0
        assert score.counts.wer == pytest.approx(1.0)


# ----------------------------------------------------------------- normalisation

def test_normalisation_is_per_token_and_keeps_positions():
    tokens = tokenize("Hello, -- my NAME is Adébáyọ̀!")
    assert [t.norm for t in tokens] == ["hello", "my", "name", "is", "adebayo"]
    # "--" disappears but the remaining words keep their original positions.
    assert [t.position for t in tokens] == [0, 2, 3, 4, 5]
    assert tokens[-1].raw == "Adébáyọ̀!"


@pytest.mark.parametrize("raw, expected", [
    ("Don't", "dont"), ("Lagos.", "lagos"), ("(Abuja)", "abuja"),
    ("well-known", "wellknown"), ("2023", "2023"), ("Ọ̀yọ́", "oyo"), ("...", ""),
])
def test_normalize_token(raw, expected):
    assert normalize_token(raw) == expected


def test_case_and_punctuation_differences_are_not_errors():
    score = score_clip("My name is Sobande.", "my name is sobande")
    assert score.counts.errors == 0


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        score_clip("  ...  ", "anything")


def test_non_string_transcript_raises():
    with pytest.raises(TypeError):
        score_clip(123, "hello")


# ----------------------------------------------------------------- alignment

def test_perfect_match():
    score = score_clip("good morning lagos", "good morning lagos")
    assert score.counts == ErrorCounts(hits=3)
    assert score.counts.wer == 0.0


def test_deletion_and_insertion_ops():
    ref, hyp = tokenize("a b c d"), tokenize("a c d e")
    ops = [p.op for p in align(ref, hyp)]
    assert ops == [MATCH, DEL, MATCH, MATCH, INS]


def test_tie_break_prefers_substitutions_left_to_right():
    # Two optimal alignments exist (2 subs, or del+match+ins); we pick 2 subs.
    ops = [p.op for p in align(tokenize("a b"), tokenize("b a"))]
    assert ops == [SUB, SUB]
    # A name split into several words: the substitution comes first.
    ops = [p.op for p in align(tokenize("olu wa da"), tokenize("oluwada"))]
    assert ops == [SUB, DEL, DEL]


def test_counts_add_up():
    score = score_clip("the cat sat on the mat", "a cat sat on mat today")
    c = score.counts
    assert c.hits + c.substitutions + c.deletions == 6          # every ref word accounted for
    assert c.hits + c.substitutions + c.insertions == 6         # every hyp word accounted for


def test_total_errors_match_jiwer_on_random_sentences():
    jiwer = pytest.importorskip("jiwer")
    rng = random.Random(42)
    vocab = ["ade", "bola", "chi", "dayo", "eze", "funmi", "go", "home", "to", "the"]
    for _ in range(300):
        ref = " ".join(rng.choices(vocab, k=rng.randint(1, 12)))
        hyp = " ".join(rng.choices(vocab, k=rng.randint(0, 12)))
        ours = score_clip(ref, hyp).counts
        if hyp:
            theirs = jiwer.process_words(ref, hyp)
            expected_errors = theirs.substitutions + theirs.deletions + theirs.insertions
        else:
            expected_errors = len(ref.split())
        assert ours.errors == expected_errors, (ref, hyp)


# ----------------------------------------------------------------- proper nouns

def test_sentence_initial_capitals_and_pronoun_i_are_not_names():
    ref = "Yesterday I went to Ibadan with Tunde. I'm tired."
    words = ref.split()
    found = {words[p] for p in find_proper_noun_positions(ref)}
    assert found == {"Ibadan", "Tunde."}


def test_known_limitation_name_at_sentence_start_is_missed():
    # Documented heuristic limitation: we cannot tell this "Tunde" is a name.
    assert find_proper_noun_positions("Tunde went home.") == set()


def test_initials_do_not_end_sentence_and_are_not_names():
    ref = "We saw Prof. A. Musa today."
    words = ref.split()
    found = {words[p] for p in find_proper_noun_positions(ref)}
    assert found == {"Musa"}


def test_sentence_end_inside_quotes():
    ref = 'She asked "where is Kemi?" Then Ngozi left.'
    words = ref.split()
    found = {words[p] for p in find_proper_noun_positions(ref)}
    # 'Kemi?"' ends a sentence even though a quote follows the "?", so "Then" is
    # a sentence start; "Ngozi" comes after it and is detected.
    assert found == {'Kemi?"', "Ngozi"}


def test_all_caps_reference_returns_none():
    assert find_proper_noun_positions("HE HOPED THERE WOULD BE STEW FOR DINNER") is None
    score = score_clip("HE HOPED THERE WOULD BE STEW", "he hoped there would be stew")
    assert score.proper_nouns is None


def test_proper_noun_result_shows_what_the_model_heard():
    score = score_clip("my name is Sobande and I live in Lagos", "my name is show bandit and i live in lagos")
    results = {r.word: r for r in score.proper_nouns}
    assert results["Sobande"].correct is False
    assert results["Sobande"].heard_as == "show bandit"
    assert results["Lagos"].correct is True
    assert proper_noun_counts([score]) == (1, 2)


def test_proper_noun_counts_skip_all_caps_clips():
    cased = score_clip("I visited Enugu", "i visited enugu")
    caps = score_clip("I VISITED ENUGU", "i visited a new goo")
    assert proper_noun_counts([cased, caps]) == (1, 1)


# ----------------------------------------------------------------- tables

def test_top_substitutions_counts_pairs():
    scores = [
        score_clip("call Sobande now", "call show now"),
        score_clip("tell Sobande", "tell show"),
        score_clip("see Eze", "see easy"),
    ]
    assert top_substitutions(scores, n=2) == [("sobande", "show", 2), ("eze", "easy", 1)]


def test_worst_clips_sorted_by_wer():
    scored = [
        ("good", score_clip("a b c d", "a b c d")),
        ("bad", score_clip("a b", "x y")),
        ("medium", score_clip("a b c d", "a b x d")),
    ]
    assert [clip_id for clip_id, _ in worst_clips(scored, n=2)] == ["bad", "medium"]


# ----------------------------------------------------------------- bootstrap

def test_bootstrap_point_estimate_is_corpus_wer_and_inside_ci():
    errors, words = [1, 0, 3, 2, 5, 0, 1], [10, 8, 12, 9, 20, 5, 7]
    est = bootstrap_wer(errors, words, seed=1)
    assert est.value == pytest.approx(sum(errors) / sum(words))
    assert est.ci_low <= est.value <= est.ci_high
    assert est.ci_low < est.ci_high


def test_bootstrap_is_reproducible_with_seed():
    errors, words = [1, 0, 3, 2, 5], [10, 8, 12, 9, 20]
    assert bootstrap_wer(errors, words, seed=7) == bootstrap_wer(errors, words, seed=7)


def test_bootstrap_identical_clips_have_zero_width_ci():
    est = bootstrap_wer([2, 2, 2, 2], [10, 10, 10, 10])
    assert est.ci_low == pytest.approx(0.2)
    assert est.ci_high == pytest.approx(0.2)


def test_bootstrap_more_clips_narrower_ci():
    rng = random.Random(0)
    words = [rng.randint(5, 30) for _ in range(400)]
    errors = [rng.randint(0, w // 2) for w in words]
    small = bootstrap_wer(errors[:40], words[:40])
    large = bootstrap_wer(errors, words)
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        bootstrap_wer([1, 2], [10])
    with pytest.raises(ValueError):
        bootstrap_wer([], [])
    with pytest.raises(ValueError):
        bootstrap_ratio([0, 0], [0, 0])


def test_bootstrap_difference():
    est = bootstrap_wer_difference([3] * 50, [10] * 50, [1] * 50, [10] * 50)
    assert est.value == pytest.approx(0.2)
    assert est.ci_low == pytest.approx(0.2) and est.ci_high == pytest.approx(0.2)

    rng = random.Random(3)
    words = [rng.randint(5, 25) for _ in range(100)]
    noisy_a = [rng.randint(0, w) for w in words]
    est = bootstrap_wer_difference(noisy_a, words, noisy_a, words)
    assert est.ci_low < 0 < est.ci_high          # same data: no real gap
    assert not math.isnan(est.value)
