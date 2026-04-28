"""Pick a diversity-balanced multi-clip selection set from BEAT2's split CSV.

Filters BEAT2's `train_test_split.csv` to a requested split, buckets clips
by (speaker, emotion), and round-robin samples N clips so the output spans
as many speaker x emotion cells as available. Emits a YAML-ready list
ready to paste into `ref_clip_audio_paths:` of a Stage 2 or Stage 3 config.

Usage:
    python scripts/pick_eval_clips.py                        # default: 16 clips from test
    python scripts/pick_eval_clips.py --split test --n 16
    python scripts/pick_eval_clips.py --split val --n 8

Output goes to stdout. Pipe it or copy-paste the `ref_clip_audio_paths:`
block into your config.
"""

import argparse
import csv
import os
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Pick a diversity-balanced clip set from BEAT2's split CSV."
    )
    parser.add_argument(
        "--csv",
        default=os.path.expandvars(
            "$WORK/data/beat2_raw/beat_english_v2.0.0/train_test_split.csv"
        ),
        help="Path to BEAT2 train_test_split.csv.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "additional"],
        help="Which split to draw clips from (default: test).",
    )
    parser.add_argument(
        "--n", type=int, default=16,
        help="Number of clips to pick (default: 16).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for reproducible within-bucket ordering.",
    )
    parser.add_argument(
        "--wav-prefix",
        default="$WORK/data/beat2_raw/beat_english_v2.0.0/wave16k",
        help="Directory prefix used to compose full .wav paths in the YAML output.",
    )
    args = parser.parse_args()

    buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            uid = row[0].strip()
            split = row[1].strip()
            if split != args.split:
                continue
            parts = uid.split("_")
            if len(parts) < 3:
                continue
            speaker = parts[1]
            try:
                emotion = int(parts[2])
            except ValueError:
                continue
            buckets[(speaker, emotion)].append(uid)

    if not buckets:
        raise SystemExit(f"No clips found in split={args.split} from {args.csv}")

    rng = random.Random(args.seed)
    for key in buckets:
        buckets[key].sort()
        rng.shuffle(buckets[key])

    # Round-robin: cycle through buckets sorted by size descending so densely
    # populated cells contribute first when N exceeds bucket count.
    sorted_keys = sorted(buckets.keys(), key=lambda k: (-len(buckets[k]), k))
    indices = {k: 0 for k in sorted_keys}
    chosen: list[tuple[tuple[str, int], str]] = []
    while len(chosen) < args.n:
        progressed = False
        for k in sorted_keys:
            if len(chosen) >= args.n:
                break
            i = indices[k]
            if i < len(buckets[k]):
                chosen.append((k, buckets[k][i]))
                indices[k] = i + 1
                progressed = True
        if not progressed:
            break

    coverage: dict[tuple[str, int], int] = defaultdict(int)
    for key, _ in chosen:
        coverage[key] += 1

    print(f"# Picked {len(chosen)} clips from split={args.split} (target n={args.n})")
    print(f"# Total available cells in split: {len(buckets)}")
    print("# Coverage (speaker, emotion -> count):")
    for (sp, em), c in sorted(coverage.items()):
        print(f"#   {sp:>10}  emo={em}  picked={c}  available={len(buckets[(sp, em)])}")
    print()
    print("ref_clip_audio_paths:")
    for _, uid in chosen:
        print(f'  - "{args.wav_prefix}/{uid}.wav"')


if __name__ == "__main__":
    main()
