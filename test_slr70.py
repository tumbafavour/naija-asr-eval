"""Unit tests for slr70.py. Run with:  pytest -v"""

import math
import random
from collections import Counter

import pytest

from metrics import score_clip
from slr70 import Utterance, balanced_sample, parse_file_id, parse_line_index, round_robin_sample

# Lines in the real line_index format (the first is a real SLR70 line).
SAMPLE_TSV = (
    "﻿ngm_02436_01770933554\tOgbonna Nwuke's father worked with the Okpara administration\n"
    "ngf_00295_00153749982\tIkechi Uko is from Abia State Nigeria\n"
    "\n"
    "ngf_00295_00999999999\tMaw may refer to a fish air bladder\n"
)


def make_utterances(speakers_per_gender: dict[str, dict[int, int]]) -> list[Utterance]:
    """{"female": {speaker_number: clip_count}, ...} -> fake utterances."""
    prefix = {"female": "ngf", "male": "ngm"}
    out = []
    for gender, speakers in speakers_per_gender.items():
        for number, count in speakers.items():
            for k in range(count):
                file_id = f"{prefix[gender]}_{number:05d}_{k:011d}"
                out.append(Utterance(file_id, f"{prefix[gender]}_{number:05d}", gender, f"sentence {k}"))
    return out


# ----------------------------------------------------------------- parsing

def test_parse_file_id():
    assert parse_file_id("ngm_02436_01770933554") == ("ngm_02436", "male")
    assert parse_file_id("ngf_00295_00153749982") == ("ngf_00295", "female")


@pytest.mark.parametrize("bad", ["", "ngx_00001_1", "ngm_02436", "ngm-02436-0177", "NGM_02436_0177"])
def test_parse_file_id_rejects_unknown_patterns(bad):
    with pytest.raises(ValueError):
        parse_file_id(bad)


def test_parse_line_index_reads_ids_speakers_genders_and_text():
    utterances, problems = parse_line_index(SAMPLE_TSV)
    assert problems == []
    assert [u.file_id for u in utterances] == [
        "ngm_02436_01770933554", "ngf_00295_00153749982", "ngf_00295_00999999999"]
    first = utterances[0]
    assert first.speaker == "ngm_02436" and first.gender == "male"
    assert first.transcript == "Ogbonna Nwuke's father worked with the Okpara administration"
    assert {u.speaker for u in utterances[1:]} == {"ngf_00295"}


def test_parse_line_index_skips_and_reports_bad_lines():
    text = (
        "ngm_00001_00000000001\tA good line\n"
        "not_an_id\tSome text\n"                  # bad ID
        "ngm_00001_00000000002\t   \n"            # empty transcript
        "ngm_00001_00000000003\n"                 # no tab at all
        "ngm_00001_00000000001\tA duplicate\n"    # duplicate ID
    )
    utterances, problems = parse_line_index(text)
    assert [u.file_id for u in utterances] == ["ngm_00001_00000000001"]
    assert len(problems) == 4
    assert "line 2" in problems[0] and "duplicate" in problems[3]


def test_tab_inside_transcript_is_kept():
    utterances, _ = parse_line_index("ngf_00001_00000000001\tLagos\tIsland is busy\n")
    assert utterances[0].transcript == "Lagos\tIsland is busy"


# ----------------------------------------------------------------- sampling

def test_round_robin_spreads_clips_over_speakers():
    # One very talkative speaker (50 clips) and four quiet ones (5 each).
    pool = make_utterances({"male": {1: 50, 2: 5, 3: 5, 4: 5, 5: 5}})
    chosen = round_robin_sample(pool, 10, random.Random(0))
    per_speaker = Counter(u.speaker for u in chosen)
    assert len(chosen) == 10
    assert len(per_speaker) == 5 and set(per_speaker.values()) == {2}


def test_round_robin_max_per_speaker_is_ceiling():
    pool = make_utterances({"female": {s: 20 for s in range(1, 8)}})       # 7 speakers
    chosen = round_robin_sample(pool, 30, random.Random(1))
    per_speaker = Counter(u.speaker for u in chosen)
    assert max(per_speaker.values()) == math.ceil(30 / 7)
    assert max(per_speaker.values()) - min(per_speaker.values()) <= 1


def test_round_robin_uses_talkative_speakers_once_quiet_ones_run_out():
    pool = make_utterances({"male": {1: 10, 2: 1, 3: 1}})
    chosen = round_robin_sample(pool, 6, random.Random(0))
    assert Counter(u.speaker for u in chosen) == {"ngm_00001": 4, "ngm_00002": 1, "ngm_00003": 1}


def test_round_robin_returns_everything_when_pool_is_small():
    pool = make_utterances({"male": {1: 2, 2: 1}})
    assert len(round_robin_sample(pool, 10, random.Random(0))) == 3


def test_round_robin_never_repeats_a_clip():
    pool = make_utterances({"female": {1: 3, 2: 3}})
    chosen = round_robin_sample(pool, 6, random.Random(0))
    assert len({u.file_id for u in chosen}) == 6


def test_balanced_sample_splits_genders_evenly():
    pool = make_utterances({"female": {s: 30 for s in range(1, 21)}, "male": {s: 30 for s in range(1, 11)}})
    chosen = balanced_sample(pool, 300, seed=42)
    assert Counter(u.gender for u in chosen) == {"female": 150, "male": 150}


def test_balanced_sample_odd_n_gives_extra_to_female():
    pool = make_utterances({"female": {1: 10}, "male": {2: 10}})
    assert Counter(u.gender for u in balanced_sample(pool, 7, seed=0)) == {"female": 4, "male": 3}


def test_balanced_sample_is_reproducible_and_seed_sensitive():
    pool = make_utterances({"female": {s: 20 for s in range(1, 9)}, "male": {s: 20 for s in range(1, 9)}})
    first = [u.file_id for u in balanced_sample(pool, 40, seed=42)]
    assert first == [u.file_id for u in balanced_sample(pool, 40, seed=42)]
    assert first == [u.file_id for u in balanced_sample(list(reversed(pool)), 40, seed=42)]   # input order irrelevant
    assert first != [u.file_id for u in balanced_sample(pool, 40, seed=7)]


def test_balanced_sample_reports_shortfall_by_returning_fewer():
    pool = make_utterances({"female": {1: 100}, "male": {2: 3}})
    chosen = balanced_sample(pool, 20, seed=0)
    assert Counter(u.gender for u in chosen) == {"female": 10, "male": 3}


# ----------------------------------------------------------------- scoring an SLR70 line

def test_slr70_line_runs_through_the_same_name_analysis():
    utterances, _ = parse_line_index("ngf_00295_00153749982\tIkechi Uko is from Abia State Nigeria\n")
    score = score_clip(utterances[0].transcript, "Ike chi uko is from aba state nigeria")
    names = {r.word: r.correct for r in score.proper_nouns}
    # "Ikechi" starts the sentence, so the heuristic cannot see it (a documented limitation).
    assert names == {"Uko": True, "Abia": False, "State": True, "Nigeria": True}
