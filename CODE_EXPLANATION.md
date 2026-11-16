# Code explanation

This document explains how naija-asr-eval works and why it is built the way it is. Read it alongside the code. Links point to the exact lines.

Contents:

1. Project Overview
2. Architecture and Design Decisions
3. Code Walkthrough
4. Technical Concepts
5. Line-by-Line Explanation of the Complex Parts
6. Error Handling and Edge Cases
7. Performance Considerations
8. How to Modify and Extend
9. Learning Notes and Likely Interview Questions

---

## 1. Project Overview

**Question.** How much worse do open speech-recognition (ASR) models do on Nigerian-accented English than on standard English, and how do they fail on Nigerian names?

**What the code does**, end to end:

```
metadata files ──► filter + fixed-seed sample ──► fetch audio for chosen clips only ──► 16 kHz FLAC cache
                                                                                           │
results/*.csv + chart ◄── score (metrics.py) ◄── predictions/<model>.csv ◄── Whisper tiny/base/small
```

**Where each part runs:**

| Part | Runs on | Why |
|---|---|---|
| `metrics.py`, `slr70.py` and their tests | Any laptop, plain `pip` | Pure Python + numpy, so the logic can be tested without a GPU |
| `naija_asr_eval.ipynb` | Google Colab (free T4 GPU) | Downloads data, runs Whisper, calls the two modules to score |

**Headline result (SLR70 vs LibriSpeech, whisper-small):** 14.8% vs 4.3% WER. That is about 3.5× worse on Nigerian English at every model size. The gap survives a stricter normaliser and speaker-level confidence intervals. The full results are in [README.md](README.md).

---

## 2. Architecture and Design Decisions

Each decision lists the alternatives considered and why this one won.

### 2.1 Pure-Python scoring modules, separate from the notebook

- **Choice:**
  - [metrics.py](metrics.py) holds all scoring.
  - [slr70.py](slr70.py) holds SLR70 parsing and sampling.
  - Neither imports audio or model libraries. The notebook imports them.
- **Alternatives:**
  - Put everything in the notebook. This is simpler, but nothing can be unit-tested, and notebook cells are easy to run out of order.
  - Build a full Python package (`src/`, `pyproject.toml`). That is cleaner for a large project, but more structure than a beginner needs to explain.
- **Trade-off:** Colab needs the two `.py` files uploaded, or the repo cloned. In return, the exact code that produced the numbers is the code the 64 tests check. An optional cell in section 0 of the notebook can even run those tests on Colab.

### 2.2 Writing the word alignment ourselves instead of using `jiwer`

- **Choice:** [`align`](metrics.py:191) implements word-level Levenshtein alignment, about 40 lines.
- **Alternative:** `jiwer`, the standard WER library.
- **Why:**
  - The proper-noun analysis needs to know which model word lined up with which reference word. The substitution table needs the same.
  - We also want a fixed, documented tie-break, and code you can explain line by line.
- **Safety net:** `test_total_errors_match_jiwer_on_random_sentences` checks our error totals against jiwer on 300 random sentence pairs.

### 2.3 Normalising one word at a time

- **Choice:** [`normalize_token`](metrics.py:62) lowercases, removes accents (é → e, ọ → o) and strips punctuation, applied to each word separately.
- **Alternative:** normalise the whole sentence, as Whisper's normaliser does: "thirty five" → "35", "colour" → "color".
- **Why:** the proper-noun check reads capitals from the original reference. If normalisation merged or split words, word *i* of the normalised text would no longer be word *i* of the original.
- **Trade-off:** pure formatting differences count as errors ("thirty" vs "30").
- **How we handled it:** a robustness check rescores everything with Whisper's normaliser through [`counts_with_normalizer`](metrics.py:329). The gap barely moved: 10.5 → 9.1 points for whisper-small.
- **Why remove accents?** Whisper almost never writes Yoruba tone marks. We want to know whether it recognised "Adébáyọ̀", not whether it spelled it the Yoruba way.

### 2.4 Corpus-level WER, not the average of per-clip WERs

- **Choice:** [`corpus_wer`](metrics.py:365) is total errors ÷ total reference words.
- **Why:** averaging per-clip WERs gives a 3-word clip the same weight as a 40-word clip. Example: one 1-word clip with 1 error plus one 9-word clip with 0 errors.
  - Corpus WER = 1/10 = 10%.
  - Mean of per-clip WERs = (100% + 0%)/2 = 50%.
- This is tested in `test_corpus_wer_weights_by_length_not_mean_of_clips`.

### 2.5 Bootstrap confidence intervals, resampling clips or speakers

- **Choice:** percentile bootstrap with 2,000 resamples, in [`bootstrap_ratio`](metrics.py:433).
- **Alternatives:**
  - A normal-approximation formula. WER is a ratio of two random sums, so the formula is awkward.
  - BCa bootstrap. Slightly more accurate, but harder to explain.
- **Two versions:**
  - **Clip-level** treats clips as independent.
  - **Speaker-level** ([`bootstrap_wer_by_speaker`](metrics.py:504)) resamples whole speakers. One speaker's clips share a voice, so they are not independent. SLR70 has only 31 speakers, so the speaker-level interval is the honest one. It is wider, as expected: 12.4–17.3% vs 13.0–16.8% for whisper-small.
- **The gap** ([`bootstrap_wer_difference`](metrics.py:463)) resamples the two groups independently, because they contain different clips and speakers.

### 2.6 A capitalisation heuristic for proper nouns, instead of an NER model

- **Choice:** [`find_proper_noun_positions`](metrics.py:135). A word counts as a likely name if it:
  - is capitalised
  - is not at the start of a sentence
  - is not "I" or a contraction of it
  - is not a title
  - is not a single-letter initial
- **Alternative:** a named-entity recogniser such as spaCy.
  - Off-the-shelf NER is trained mostly on Western news, so it is itself unreliable on Nigerian names. That would mean measuring one biased model with another.
  - It also adds a heavy dependency.
- **Trade-off:** the heuristic is transparent and testable but noisy. It counts "Road" and "American", and misses names at the start of a sentence. The README says plainly that it is a heuristic, and one test documents a known miss.

### 2.7 Sampling: a fixed seed, and limits per speaker

- **AfriSpeech** ([`sample_capped`](naija_asr_eval.ipynb)): shuffle with `SEED`, keep at most 3 clips per speaker, then take the first 75.
- **SLR70** ([`round_robin_sample`](slr70.py:70)): **round-robin across speakers**, one clip from every speaker before anyone gives a second.
  - Why not the same cap of 3? SLR70 has only 19 female and 12 male speakers. A cap of 3 would give 57 and 36 clips, well short of 150 per gender.
  - Round-robin reaches any target size with the widest possible spread of voices.
- **Alternative considered:** reservoir sampling while streaming. It is memory-efficient, but it doesn't let you inspect the data before sampling, which you asked for.
- **Reproducibility:** inside `round_robin_sample`, speakers are **sorted before they are shuffled**, so the sample depends only on the seed, never on file order. This is tested.

### 2.8 Load metadata first, and decode only the chosen audio

Each dataset ships small metadata files next to large audio files. The notebook reads the metadata first, prints the real values, filters and samples, and only then touches audio.
- **AfriSpeech:** audio is stored as one archive per accent and split, so we stream only the accents we need. Decoding is switched off (`Audio(decode=False)`), and a stream stops once all its clips are found.
- **SLR70:** we read each WAV's duration from its header inside the zip, then decode only the 300 chosen files.
- **Alternative:** download and decode everything. That is about 200 hours of AfriSpeech audio, which is impractical on free Colab.

### 2.9 Resumable work: save per batch, per dataset

- **Choice:** predictions are appended to `results/predictions/<model>.csv` after every batch of 16. Samples are saved per dataset (`manifest_<dataset>.csv`).
- **Alternative:** save at the end. A Colab disconnect would then lose everything.
- **Benefits:**
  - A disconnect loses at most one batch.
  - Turning AfriSpeech on later reuses the SLR70 and LibriSpeech samples and transcriptions. This was tested: the samples stay byte-identical, and only new clips are transcribed.

### 2.10 Model settings

- **`language="en", task="transcribe"`:** without these, Whisper may decide that accented speech is another language, or translate it.
- **Greedy decoding:** Whisper's default. Beam search is slightly better but slower, and greedy matches the Whisper paper's Table D.1.1, which we use as a sanity check.
- **Clips over 30 s excluded:** Whisper reads 30-second windows. Longer audio needs chunking, which adds its own errors at the boundaries.
- **Only tiny, base and small:** they fit comfortably on a free T4. See section 8 for adding bigger models.

### 2.11 Never pool SLR70 with AfriSpeech

The two corpora differ in recording setup, sentence sources and how speakers were recruited. A pooled number would mix those differences with accent. "All AfriSpeech" pools only AfriSpeech's own four groups.

### 2.12 Privacy

- No audio ever enters the repo. `.gitignore` blocks audio formats, and the last notebook cell refuses to package `results/` if it contains audio or archives.
- Speaker IDs are replaced with per-run labels (`slr70-spk001`, …) by `anonymise_speakers`.

---

## 3. Code Walkthrough

### 3.1 `metrics.py`: scoring

| Function | What it does |
|---|---|
| [`Token`](metrics.py:49) | One word: `raw` (as written), `norm` (normalised), `position` (index in the original sentence) |
| [`tokenize`](metrics.py:85) | Splits on whitespace and normalises each word. Drops words that normalise to nothing (e.g. "--") but keeps every kept word's position |
| [`find_proper_noun_positions`](metrics.py:135) | Returns the positions of likely proper nouns, or `None` for all-caps text (LibriSpeech) |
| [`align`](metrics.py:191) | Minimum-edit-distance alignment: a list of `match` / `sub` / `del` / `ins` steps |
| [`ErrorCounts`](metrics.py:237) | Hits, substitutions, deletions, insertions; computes `ref_words`, `errors` and `wer`; supports `+` |
| [`score_clip`](metrics.py:301) | The main entry point: tokenises, aligns, counts errors, checks each likely name. Returns a `ClipScore` |
| [`counts_with_normalizer`](metrics.py:329) | Error counts after any whole-sentence normaliser (used with Whisper's) |
| [`format_alignment`](metrics.py:342) | A readable line such as `my name is [sobande->show] [+bandit]` |
| [`corpus_wer`](metrics.py:365), [`top_substitutions`](metrics.py:383), [`worst_clips`](metrics.py:392) | Group-level tables |
| [`bootstrap_ratio`](metrics.py:433), [`bootstrap_wer`](metrics.py:452), [`bootstrap_wer_difference`](metrics.py:463) | Clip-level confidence intervals |
| [`sum_by_cluster`](metrics.py:490), [`bootstrap_wer_by_speaker`](metrics.py:504), [`bootstrap_wer_difference_by_speaker`](metrics.py:524) | Speaker-level confidence intervals |

### 3.2 `slr70.py`: SLR70 data

| Function | What it does |
|---|---|
| [`parse_file_id`](slr70.py:31) | `ngm_02436_01770933554` → speaker `ngm_02436`, gender `male` |
| [`parse_line_index`](slr70.py:40) | Reads a TSV and returns `(utterances, problems)`. Bad lines are skipped and reported, never silently dropped |
| [`round_robin_sample`](slr70.py:70) | Takes clips one per speaker per round |
| [`balanced_sample`](slr70.py:107) | Gives each gender half the quota, then calls `round_robin_sample` within each gender |

### 3.3 `naija_asr_eval.ipynb`: the pipeline

| Section | Purpose |
|---|---|
| 0 | Get the code (upload or clone). Optionally run the unit tests on Colab |
| Configuration | Every experimental constant in one place, including the `DATASETS` switch |
| HF login | Uses the token from Colab Secrets; only required for AfriSpeech |
| 1a, 1b | Inspect metadata **before** filtering: speakers per gender, malformed lines, accent labels, country/domain values |
| 2 | Filter and sample each dataset. Saves `manifest_<dataset>.csv` |
| 3 | Fetch, decode and resample only the chosen clips, cached as FLAC in Google Drive |
| 4 | Transcribe with each model, appending to CSV after every batch |
| 5 | Score every clip: summary, gap, substitutions, worst clips, proper nouns |
| 5b | Robustness: Whisper's normaliser and speaker-level CIs → `robustness.csv` |
| 6 | Chart |
| 7 | Print failure examples for the README |
| 8 | Optional: score your own recordings |
| 9, 10 | Citations; check for audio and zip `results/` for download |

### 3.4 Tests

- [test_metrics.py](test_metrics.py): 45 tests, covering every requirement in the project brief plus edge cases.
- [test_slr70.py](test_slr70.py): 19 tests.

The tests were checked by **mutation testing**: deliberately breaking the code and confirming the right tests fail. Breaks used:
- removing "Mr."/"Dr." from the abbreviation list
- averaging per-clip WERs instead of computing corpus WER
- changing the tie-break
- replacing round-robin with plain random sampling
- splitting on every tab instead of the first
- skipping the sort before shuffling
- making the speaker bootstrap ignore speakers

---

## 4. Technical Concepts

**Word error rate (WER).** (S + D + I) / N, where N is the number of reference words.
- S = substitutions (a wrong word in place of the right one)
- D = deletions (a word missed)
- I = insertions (an extra word)

Because insertions are unbounded, WER can exceed 100%.

**Levenshtein (edit) distance.** The minimum number of substitutions, deletions and insertions that turns one sequence into another. It is computed with **dynamic programming**: a table where each cell holds the best answer for a smaller sub-problem, filled from the smallest sub-problems up. The cost is O(n × m) time for n and m words.

**Alignment and ties.** Several alignments can share the minimum cost. For "Sobande" vs "show bandit", both of these cost 2:
- substitute Sobande→show, then insert bandit
- insert show, then substitute Sobande→bandit

The WER is the same either way, but the S/D/I split and the substitution pairs differ, so the tie-break must be fixed and documented.

**Normalisation.** Making two strings comparable by removing differences we don't count as errors: case, punctuation, accents. Choosing what to normalise is a **measurement decision**, which is why there is a robustness check.

**Ratio estimator.** Corpus WER is a ratio of two sums, Σerrors / Σwords. Resampling speakers instead of clips doesn't change it, because adding errors per speaker first and then summing gives the same totals.

**Bootstrap.** We have one sample, but want to know how much the result would change with a different sample. So we simulate new samples by **drawing from our own data with replacement**, recompute the statistic each time, and read the 2.5th and 97.5th percentiles: that's the 95% percentile interval.

**Cluster (speaker-level) bootstrap.** When observations come in groups that are not independent (clips from one speaker), resample the groups. Otherwise the interval is too narrow, because it pretends there is more independent information than there is.

**Zero-shot evaluation.** Using a pretrained model as it is, without fine-tuning on the test domain. It measures what users get out of the box.

**Whisper basics.**
- An encoder–decoder transformer that reads a 30-second log-Mel spectrogram (a picture of the sound's frequencies over time) of 16 kHz audio and generates text tokens.
- Tiny, base and small have about 39M, 74M and 244M parameters.
- `fp16` (half-precision numbers) halves memory and speeds up the GPU with negligible accuracy loss.

**Resampling audio.** Datasets are recorded at 44.1 or 48 kHz, but Whisper expects 16 kHz. `librosa.resample` converts the rate with anti-aliasing filtering, so high frequencies don't fold back as noise.

**Streaming datasets.** `load_dataset(..., streaming=True)` reads rows on demand instead of downloading the whole dataset. `Audio(decode=False)` keeps the raw bytes, so rows we skip cost no decoding.

**Atomic rename.** Write a download to `name.part`, and only when it's complete rename it to `name`. A half-finished file can then never look finished.

**Seeded randomness.** `random.Random(seed)` and `numpy.random.default_rng(seed)` give the same "random" sequence every run, so samples and confidence intervals are reproducible.

---

## 5. Line-by-Line Explanation of the Complex Parts

### 5.1 `align`: alignment with a left-to-right tie-break ([metrics.py:191](metrics.py:191))

The excerpts in this section are trimmed slightly for reading, for example by joining short lines with `;`. The linked files hold the exact code.

```python
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
```

- `cost[i][j]` means: **the fewest edits needed to turn the rest of the reference (`ref[i:]`) into the rest of the hypothesis (`hyp[j:]`)**. These are *suffix* costs, so the table is filled from the end backwards.
- **Edges:** if the reference is used up (`i == n`), every remaining hypothesis word is an insertion. If the hypothesis is used up (`j == m`), every remaining reference word is a deletion.
- **Inside the table,** there are three choices:
  - move diagonally: a match costs 0, a substitution costs 1. `(a != b)` is `True`/`False`, which Python treats as 1/0.
  - delete `ref[i]` (go down)
  - insert `hyp[j]` (go right)

```python
while i < n or j < m:
    if i < n and j < m:
        same = ref[i].norm == hyp[j].norm
        if cost[i][j] == cost[i + 1][j + 1] + (not same):
            pairs.append(AlignedPair(MATCH if same else SUB, ref[i], hyp[j])); i += 1; j += 1; continue
    if i < n and (j == m or cost[i][j] == cost[i + 1][j] + 1):
        pairs.append(AlignedPair(DEL, ref[i], None)); i += 1
    else:
        pairs.append(AlignedPair(INS, None, hyp[j])); j += 1
```

- We walk forward from the start of both sentences. At each step we take a move **only if it still leads to the minimum total**: that's what `cost[i][j] == cost[next] + step_cost` checks.
- When several moves qualify, the order of the `if`s decides: **match/substitution first, then deletion, then insertion**. That pushes substitutions as early as possible, so "Sobande" vs "show bandit" becomes `[sobande->show] [+bandit]`.
- Computing suffix costs is what makes a simple left-to-right walk possible.

### 5.2 Sentence starts and abbreviations ([metrics.py:107](metrics.py:107), [metrics.py:135](metrics.py:135))

```python
def _ends_sentence(raw_word):
    word = raw_word.rstrip(_CLOSING_CHARS)          # 'Kemi?"' -> 'Kemi?'
    if not word.endswith(_SENTENCE_END):            # . ! ?
        return False
    if word.endswith("."):
        core = _strip_edges(word)
        if core.lower() in ABBREVIATIONS:           # "Mr." "Dr." "Engr." do not end a sentence
            return False
        if len(core) == 1 and core.isupper():       # "J." in "J. K. Rowling" is an initial
            return False
    return True
```

In `find_proper_noun_positions`, a flag `at_sentence_start` starts as `True`. After each word it is set to `_ends_sentence(word)`. A word is a candidate only if it is capitalised **and** `at_sentence_start` is `False`.

So in "I met **Mr.** Okafor", the word "Mr." doesn't set the flag, and "Okafor" counts. In "He left. Then Ade came.", "left." sets the flag, so "Then" is skipped and "Ade" counts.

All-caps text returns `None`. With no lowercase letters, capitals carry no information, so "not applicable" is more honest than a wrong number.

### 5.3 What the model "heard" ([metrics.py:288](metrics.py:288))

```python
start = index
while start > 0 and pairs[start - 1].op != MATCH:
    start -= 1
end = index
while end < len(pairs) - 1 and pairs[end + 1].op != MATCH:
    end += 1
return " ".join(p.hyp.norm for p in pairs[start:end + 1] if p.hyp is not None)
```

From the name's position, we walk left and right until we hit a correctly matched word. Everything the model wrote in between is what it produced instead of the name. That's how "Sobande" reports "show bandit" rather than only "show".

### 5.4 Vectorised bootstrap ([metrics.py:420](metrics.py:420))

```python
idx = rng.integers(0, len(num), size=(n_resamples, len(num)))
return num[idx].sum(axis=1) / den[idx].sum(axis=1)
```

- `idx` is a 2,000 × n grid of random clip numbers. Each row is one bootstrap resample, drawn with replacement.
- `num[idx]` looks up the errors for every chosen clip in one step (numpy fancy indexing). `.sum(axis=1)` totals each row.
- The result is 2,000 corpus WERs with no Python loop. `_interval` then takes the 2.5th and 97.5th percentiles.

### 5.5 Speaker-level totals ([metrics.py:490](metrics.py:490))

```python
_, cluster_index = np.unique(np.asarray(labels, dtype=str), return_inverse=True)
return (np.bincount(cluster_index, weights=num, minlength=n_clusters),
        np.bincount(cluster_index, weights=den, minlength=n_clusters))
```

- `np.unique(..., return_inverse=True)` gives each clip the number of its speaker, so ["b", "a", "b"] becomes [1, 0, 1].
- `np.bincount(..., weights=...)` adds up the errors (and words) for each speaker.
- The normal clip bootstrap then runs on those per-speaker totals, which is exactly resampling speakers.

### 5.6 Round-robin sampling ([slr70.py:70](slr70.py:70))

```python
speakers = sorted(by_speaker)
rng.shuffle(speakers)
for speaker in speakers:
    clips = sorted(by_speaker[speaker], key=lambda u: u.file_id)
    rng.shuffle(clips)
...
while len(chosen) < n:
    took_any = False
    for speaker in speakers:
        clips = by_speaker[speaker]
        if round_index < len(clips):
            chosen.append(clips[round_index]); took_any = True
            if len(chosen) == n: break
    if not took_any: break        # every speaker has run out
    round_index += 1
```

- **Sorting before shuffling** makes the result depend only on the seed, not on the order the TSV happened to list rows.
- Round 0 takes each speaker's first (randomly chosen) clip, round 1 their second, and so on.
- A speaker with few clips simply drops out once they've run out, and the others continue.
- The `took_any` check stops the loop if the whole pool is used up before reaching `n`. The caller then reports the shortfall.

### 5.7 Resumable transcription (notebook section 4)

```python
done = completed_clip_ids(out_path)                 # IDs already in the CSV
todo = [c for c in clip_ids if c not in done]
for start in range(0, len(todo), BATCH_SIZE):
    ...
    append_rows(out_path, rows)                     # write this batch immediately
```

The CSV is the record of what's finished. After a disconnect, rerunning the cell skips everything already written. `load_predictions` keeps the last row per clip (`drop_duplicates("clip_id", keep="last")`), so a batch written twice by an interrupted run can't double-count.

### 5.8 Safe downloads (notebook section 1a)

```python
partial = target.with_name(target.name + ".part")
for base in mirrors:
    ... copy the response into `partial` ...
    if target.suffix == ".zip" and not zipfile.is_zipfile(partial):
        continue                                    # corrupt or an error page: try the next mirror
    partial.replace(target)                         # rename only when complete
```

The download tries mirrors in order, checks that a zip is valid, and renames the file only when it's complete. A file that already exists is reused.

---

## 6. Error Handling and Edge Cases

| Situation | What happens | Where |
|---|---|---|
| Empty model output | Every reference word counts as a deletion (WER = 100%) | `score_clip`; tested |
| Model output is `NaN` (an empty CSV cell read by pandas) | Treated as `""` | `_clean_text` |
| Empty reference | `ValueError`, because WER is undefined and must not be hidden | `score_clip`; tested |
| Transcript isn't text (e.g. a number) | `TypeError` | `_clean_text`; tested |
| Model repeats itself (hallucination) | Insertions are counted, so WER can exceed 100% | tested |
| All-caps reference (LibriSpeech) | Proper-noun result is `None` ("not applicable") | tested |
| "Mr." / "Dr." / "Engr." / initials | The next word is not treated as a sentence start | tested |
| AfriSpeech's missing apostrophes ("Im", "Ive") | Treated as the pronoun "I", not a name | found in real data; tested |
| Malformed SLR70 line (bad ID, no tab, empty text, duplicate) | Skipped and reported | `parse_line_index`; tested |
| Tab inside a transcript | Kept: only the first tab separates ID from text | tested |
| Not enough eligible clips | Sample everything available and print `SHORT by N` (Hausa: 51 of 75) | notebook section 2 |
| Clip longer than 30 s | Excluded before sampling | notebook section 2 |
| A transcript with no audio in the zip | Excluded and counted | `build_slr70_manifest` |
| One audio file won't decode | Reported and dropped; the run continues | `fetch_from_stream`, `fetch_from_zip` |
| A download mirror fails, or a zip is corrupt | Try the next mirror; atomic `.part` rename | `download_file` |
| AfriSpeech split named "dev" or "validation" | Both are tried | `open_afrispeech_stream` |
| A file hash like `12e45…` | Read as text (`dtype=str`), never as a number | manifest loading |
| Colab disconnects | Resume from the CSVs; at most one batch is lost | sections 2–4 |
| No `HF_TOKEN` | Fine for SLR70; a clear error if AfriSpeech is on | HF login cell |
| Audio accidentally in `results/` | Packaging refuses | section 10 |
| Bootstrap inputs of different lengths, or empty | `ValueError` | `_as_count_arrays`; tested |
| A normaliser empties a reference | `ref_words = 0`; its insertions still count in the totals | `counts_with_normalizer`; tested |

---

## 7. Performance Considerations

**Alignment** is O(n × m) per clip, where n and m are the reference and model word counts. Sentences are 5–40 words, so scoring 1,800 clip–model pairs takes about a second.

**Bootstrap** builds a 2,000 × 300 index array (600,000 integers, about 5 MB) and sums it with numpy. One interval takes milliseconds.

**Data loading** costs the most, and is kept down by:
- reading small metadata files first
- streaming only the AfriSpeech accents we need, and stopping early
- reading SLR70 durations from WAV headers instead of decoding
- decoding only the chosen clips

**Measured on the first Colab run (T4):**

| Step | Time |
|---|---|
| Sampling, including the 1.2 GB SLR70 download and reading 3,359 WAV headers | ~5 min |
| Fetching and resampling 600 clips | 38 s |
| Transcribing 600 clips with all three models | ~2 min |
| Scoring | ~1 s |

**Transcription:** batches of 16 in fp16 on the GPU. The model is deleted and GPU memory released before the next model loads.

**Caching:**
- 16 kHz FLAC in Google Drive is several times smaller than the original 44.1/48 kHz WAV (a third of the samples, then lossless compression), and survives disconnects.
- The SLR70 zips live on Colab's temporary disk, which is fast and keeps Drive from filling up. They're downloaded again only if a clip is missing.

**Memory:** at most one batch of audio is held at a time, and the LibriSpeech pool keeps only metadata, never audio.

---

## 8. How to Modify and Extend

**Add a model.** Add its Hugging Face ID to `MODELS`, for example `"openai/whisper-medium"`. On a T4, lower `BATCH_SIZE` to about 8 for medium. Add a colour to `MODEL_COLORS`, continuing the blue ramp: darker for bigger.

**Enable AfriSpeech** (once approved):
1. Add `HF_TOKEN` to Colab Secrets.
2. Set `DATASETS = ["slr70", "afrispeech"]`.
3. Run all. Only the new clips are processed.

**Change sample sizes.** Change `SLR70_CLIPS`, `CLIPS_PER_GROUP`, `LIBRISPEECH_CLIPS` or the per-speaker caps. Delete the matching `manifest_<dataset>.csv`, or set `FORCE_RESAMPLE = True`, to draw a new sample.

**Add a dataset:**
1. Write a `build_<name>_manifest()` that returns the same columns as `MANIFEST_COLUMNS`.
2. Add it to `MANIFEST_BUILDERS`, `KNOWN_DATASETS` and the group lists.
3. Add a branch in `fetch_all_audio`.
4. Put any pure parsing or sampling logic in its own module with tests, as `slr70.py` does.

**Compare genders.** SLR70's manifest has a `gender` column. Group by it and call `bootstrap_wer_by_speaker` on each half.

**Improve the name measure:**
- Hand-label which detected proper nouns are Nigerian names or places (a CSV of `clip_id, name, is_nigerian`), then report that subset separately.
- Or compare against an NER model as a second opinion.
- A speaker-level interval for the name error rate needs only `sum_by_cluster` plus `bootstrap_ratio`.

**Try another normaliser.** Pass any `str → str` function to `counts_with_normalizer`.

**Out of scope here:** fine-tuning. AfriSpeech's authors have already shown it helps. This project measures what users get from the models out of the box.

---

## 9. Learning Notes and Likely Interview Questions

### Numbers to know by heart

- Whisper-small: **14.8%** WER on Nigerian English (SLR70) vs **4.3%** on LibriSpeech, about **3.5×** worse.
- Gap from tiny to small: **20.3 → 10.5** points. Smaller in absolute terms, about the same ratio.
- With Whisper's normaliser: **12.6% vs 3.5%**, gap **9.1 points** (speaker CI 6.7–11.6).
- Proper nouns: whisper-small got **33%** wrong; tiny and base about half.
- Data: 300 SLR70 clips from **31 speakers** (19 female, 12 male); 300 LibriSpeech clips from 40 speakers.
- Validation: our LibriSpeech numbers are within about 0.6 points of the Whisper paper's for all three models.

### Likely questions and how to answer them

1. **"Why is WER the right metric, and what does it miss?"** It's the standard for ASR, so results compare with the literature. But it treats every word equally: getting "the" wrong costs the same as getting someone's name wrong. That's exactly why I added the proper-noun analysis.

2. **"Why corpus WER instead of averaging per clip?"** Averaging lets short clips dominate. Use the 10% vs 50% example from section 2.4.

3. **"How do you know the gap isn't just formatting?"** I rescored with Whisper's own normaliser, which treats "thirty" and "30" as equal. The gap went from 10.5 to 9.1 points and stayed far from zero.

4. **"How do you know your pipeline is correct?"**
   - With Whisper's normaliser, my LibriSpeech numbers match the Whisper paper's table.
   - My alignment's error totals match jiwer on 300 random cases.
   - 64 unit tests, checked by deliberately breaking the code.

5. **"Explain your confidence intervals."** A bootstrap: resample with replacement 2,000 times and take the middle 95%. Clips from the same speaker aren't independent, so I also resample whole speakers. With only 31 speakers that's the honest interval, and it's wider.

6. **"Is the gap caused by accent?"** Not only. SLR70 and LibriSpeech also differ in recording conditions and text. That's why the AfriSpeech comparison *between* Nigerian accents, from one corpus, matters, and why I never pool corpora.

7. **"Does a bigger model solve the problem?"** It reduces errors for everyone. But Nigerian English stays about 3.5× harder at every size, so the relative disadvantage doesn't shrink. That's a fairness point, not just an accuracy point.

8. **"Why not use NER for names?"** NER models are trained mostly on Western text, so they're unreliable on Nigerian names too. A transparent heuristic with known, documented errors is easier to reason about. The next step is hand-labelling.

9. **"What's the tie-break in your alignment, and why does it matter?"** Explain "Sobande" → "show bandit": two equal-cost alignments, the same WER, a different S/D/I split. The tie-break must be fixed so the substitution counts are reproducible.

10. **"How would you extend this into a thesis?"** Some directions:
    - per-accent analysis with AfriSpeech
    - a hand-labelled Nigerian-name test set
    - testing whether lexicon or contextual biasing (giving the model a list of expected names) fixes name errors
    - testing fine-tuning, or larger models
    - looking at the effect on downstream tasks (e.g. name errors in transcribed clinical or customer-service calls)

11. **"What did the data surprise you with?"**
    - AfriSpeech strips apostrophes ("Im"), which broke my pronoun rule. I found it by inspecting the data and added a test.
    - Hausa has only 51 eligible held-out clips, which says something about resource gaps itself.
    - SLR70 has only 31 speakers.

12. **"What's novel here?"** Be modest and precise. Benchmarks already exist (AfriSpeech). My contribution is a reproducible zero-shot comparison with honest uncertainty, and an error analysis of how names fail.

### What you should be able to do without notes

- Explain WER, and compute it by hand for "my name is Sobande" vs "my name is show bandit": 1 substitution + 1 insertion over 4 words = 0.5.
- Draw the alignment table for a two-word example and walk through the left-to-right trace.
- Explain the bootstrap in one sentence, and why speaker-level intervals are wider.
- State every limitation in the README before anyone asks.
- Describe what happens when Colab disconnects, and why no audio is in the repo.
