#!/usr/bin/env python3
"""
Find which expected video outputs are missing after running sample.py.

Checks for each prompt index i:
  ./generation/<name>/<group_of_prompt[i]>/<i>/*.mp4   (and optionally *.npy)

Usage examples:
  # If you ran with --descriptions
  python find_missing_videos.py --name MARDM_SiT_XL --descriptions --descriptions_dir descriptions

  # If you ran with --text_path
  python find_missing_videos.py --name MARDM_SiT_XL --text_path ../motion_generation/text_prompt.txt

  # If you ran with --text_prompt
  python find_missing_videos.py --name MARDM_SiT_XL --text_prompt "a person walks..."
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_descriptions_dir(desc_dir: Path) -> tuple[list[str], list[str]]:
    """
    Mirrors sample.py: reads desc_dir/*.txt in sorted order.
    Returns:
      prompt_list: list[str]
      group_of_prompt: list[str] (file stem per prompt)
    """
    prompt_list: list[str] = []
    group_of_prompt: list[str] = []

    for fpath in sorted(desc_dir.glob("*.txt")):
        group = fpath.stem
        for line in fpath.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            prompt_list.append(line)
            group_of_prompt.append(group)

    return prompt_list, group_of_prompt


def load_text_path(text_path: Path) -> tuple[list[str], list[str]]:
    """
    Mirrors sample.py logic for --text_path:
      - prompt is infos[0] where infos = line.split('#')
      - group is always "text_path"
    """
    prompt_list: list[str] = []
    group_of_prompt: list[str] = []

    for raw in text_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            continue
        infos = raw.split("#")
        prompt_list.append(infos[0])
        group_of_prompt.append("text_path")

    return prompt_list, group_of_prompt


def prompt_has_outputs(base_dir: Path, group: str, idx: int, require_npy: bool) -> tuple[bool, bool]:
    """
    Returns (has_mp4, has_npy) by globbing in:
      base_dir / group / str(idx)
    """
    s_path = base_dir / group / str(idx)
    if not s_path.exists():
        return (False, False)

    has_mp4 = any(s_path.glob("*.mp4"))
    has_npy = any(s_path.glob("*.npy"))
    if require_npy:
        return (has_mp4, has_npy)
    return (has_mp4, True)  # treat npy as "don't care"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Same as sample.py --name (subdir under ./generation).")
    p.add_argument("--generation_root", default="./generation", help="Root output folder used by sample.py.")
    p.add_argument("--descriptions", action="store_true")
    p.add_argument("--descriptions_dir", default="descriptions")
    p.add_argument("--text_path", default="")
    p.add_argument("--text_prompt", default="")
    p.add_argument("--require_npy", action="store_true", help="Also require a .npy output (in addition to .mp4).")
    p.add_argument("--out_csv", default="missing_outputs.csv")
    args = p.parse_args()

    # Reconstruct prompt_list and group_of_prompt exactly as sample.py would.
    if args.descriptions:
        prompt_list, group_of_prompt = load_descriptions_dir(Path(args.descriptions_dir))
    elif args.text_prompt:
        prompt_list = [args.text_prompt]
        group_of_prompt = ["single"]
    elif args.text_path:
        prompt_list, group_of_prompt = load_text_path(Path(args.text_path))
    else:
        raise SystemExit("Need one of: --descriptions, --text_path, or --text_prompt")

    base_dir = Path(args.generation_root) / args.name

    missing_mp4: list[tuple[int, str, str]] = []
    missing_npy: list[tuple[int, str, str]] = []
    missing_either: list[tuple[int, str, str]] = []

    for i, (prompt, group) in enumerate(zip(prompt_list, group_of_prompt)):
        has_mp4, has_npy_ok = prompt_has_outputs(base_dir, group, i, require_npy=args.require_npy)
        if not has_mp4:
            missing_mp4.append((i, group, prompt))
        if args.require_npy and not has_npy_ok:
            missing_npy.append((i, group, prompt))
        if (not has_mp4) or (args.require_npy and not has_npy_ok):
            missing_either.append((i, group, prompt))

    total = len(prompt_list)
    done = total - len(missing_either)

    print(f"Checked: {total} prompts")
    print(f"Output base: {base_dir}")
    print(f"Require .npy: {args.require_npy}")
    print(f"Complete: {done}")
    print(f"Missing mp4: {len(missing_mp4)}")
    if args.require_npy:
        print(f"Missing npy: {len(missing_npy)}")
    print(f"Missing (per criteria): {len(missing_either)}")

    # Write CSV with all missing entries (per criteria)
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "group", "expected_dir", "missing_mp4", "missing_npy", "prompt"])
        for i, group, prompt in missing_either:
            expected_dir = str(base_dir / group / str(i))
            has_mp4, has_npy_ok = prompt_has_outputs(base_dir, group, i, require_npy=False)
            mp4_missing = not has_mp4
            npy_missing = not any((base_dir / group / str(i)).glob("*.npy"))
            w.writerow([i, group, expected_dir, mp4_missing, npy_missing, prompt])

    print(f"Wrote: {out_csv.resolve()}")

    # Show first few missing indices (handy for re-running with sharding)
    if missing_either:
        print("First few missing indices:", [i for i, _, _ in missing_either[:20]])


if __name__ == "__main__":
    main()
