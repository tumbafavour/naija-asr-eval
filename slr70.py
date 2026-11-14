"""Helpers for Google's Crowdsourced Nigerian English corpus (OpenSLR SLR70).

Pure Python: parses the line_index TSV files and draws a gender-balanced,
speaker-diverse sample. Downloading and decoding audio live in the notebook.

A line of line_index_male.tsv looks like:
    ngm_02436_01770933554<TAB>Ogbonna Nwuke's father worked with the Okpara administration
The file ID is <gender prefix>_<speaker number>_<utterance number>.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

GENDER_BY_PREFIX = {"ngf": "female", "ngm": "male"}
_FILE_ID = re.compile(r"^(ng[fm])_(\d+)_(\d+)$")


@dataclass(frozen=True)
class Utterance:
    file_id: str      # "ngm_02436_01770933554", also the audio file's name without ".wav"
    speaker: str      # "ngm_02436"
    gender: str       # "female" or "male"
    transcript: str


def parse_file_id(file_id: str) -> tuple[str, str]:
    """'ngm_02436_01770933554' -> ('ngm_02436', 'male'). Raises ValueError for anything else."""
    match = _FILE_ID.match(file_id)
    if not match:
        raise ValueError(f"Unrecognised SLR70 file ID: {file_id!r}")
    prefix, speaker_number, _ = match.groups()
    return f"{prefix}_{speaker_number}", GENDER_BY_PREFIX[prefix]


def parse_line_index(text: str) -> tuple[list[Utterance], list[str]]:
    """Parse a line_index TSV. Returns (utterances, problems).

    Bad lines (unrecognised ID, missing transcript, duplicate ID) are skipped and
    described in `problems`, so one bad line cannot stop the run and none are
    dropped silently. The corpus page itself warns it "may still contain errors".
    """
    utterances, problems, seen = [], [], set()
    for line_number, line in enumerate(text.lstrip("﻿").splitlines(), start=1):
        if not line.strip():
            continue
        # partition splits at the FIRST tab only, so a stray tab inside a transcript survives.
        file_id, tab, transcript = line.partition("\t")
        file_id, transcript = file_id.strip(), transcript.strip()
        try:
            speaker, gender = parse_file_id(file_id)
        except ValueError:
            problems.append(f"line {line_number}: unrecognised file ID {file_id!r}")
            continue
        if not tab or not transcript:
            problems.append(f"line {line_number}: no transcript for {file_id}")
            continue
        if file_id in seen:
            problems.append(f"line {line_number}: duplicate file ID {file_id}")
            continue
        seen.add(file_id)
        utterances.append(Utterance(file_id, speaker, gender, transcript))
    return utterances, problems


def round_robin_sample(utterances: Sequence[Utterance], n: int, rng: random.Random) -> list[Utterance]:
    """Pick up to n utterances, taking one per speaker per round.

    Speakers and each speaker's utterances are shuffled with `rng`, then we cycle
    through the speakers. Every speaker contributes before anyone contributes a
    second clip, so the sample is spread over as many voices as possible instead
    of being dominated by the speakers who recorded most. Returns fewer than n
    only when fewer than n utterances exist.
    """
    by_speaker: dict[str, list[Utterance]] = defaultdict(list)
    for utterance in utterances:
        by_speaker[utterance.speaker].append(utterance)
    # Sort before shuffling so the result depends only on the seed, not on input order.
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    for speaker in speakers:
        clips = sorted(by_speaker[speaker], key=lambda u: u.file_id)
        rng.shuffle(clips)
        by_speaker[speaker] = clips

    chosen: list[Utterance] = []
    round_index = 0
    while len(chosen) < n:
        took_any = False
        for speaker in speakers:
            clips = by_speaker[speaker]
            if round_index < len(clips):
                chosen.append(clips[round_index])
                took_any = True
                if len(chosen) == n:
                    break
        if not took_any:
            break  # every speaker has run out
        round_index += 1
    return chosen


def balanced_sample(utterances: Sequence[Utterance], n: int, seed: int) -> list[Utterance]:
    """n // 2 clips per gender (female gets the extra one when n is odd), speaker-diverse within each."""
    rng = random.Random(seed)
    genders = sorted(set(GENDER_BY_PREFIX.values()))          # ["female", "male"]
    chosen: list[Utterance] = []
    for index, gender in enumerate(genders):
        quota = n // 2 + (1 if index < n % 2 else 0)
        pool = [u for u in utterances if u.gender == gender]
        chosen += round_robin_sample(pool, quota, rng)
    return chosen
