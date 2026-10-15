#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize validated ExoPath shell scripts.

For each .sh file containing a ``python ... run.py`` command:

1. Force ``--model ExoPath``.
2. Remove ``--e_layers VALUE`` because ExoPath does not use it.
3. Rebuild ``--model_id`` from the real dataset name, feature mode,
   look-back length and prediction length.

Important:
- ``--data custom`` is retained for Electricity/Weather/Traffic because it is
  the data-loader key.
- The model ID does NOT use ``custom``. For custom datasets the dataset name is
  inferred from ``--data_path``:
      electricity.csv -> electricity
      weather.csv     -> weather
      traffic.csv     -> traffic

Examples:
    ETTh1_M_96_192
    electricity_MS_96_96
    weather_M_96_720
    traffic_M_96_336

Backups with suffix ``.bak`` are created by default.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional


def flag_pattern(name: str) -> re.Pattern:
    return re.compile(rf"(?<!\S){re.escape(name)}(?:=|\s+)([^\s\\]+)")


PATTERNS = {
    "--data": flag_pattern("--data"),
    "--data_path": flag_pattern("--data_path"),
    "--features": flag_pattern("--features"),
    "--seq_len": flag_pattern("--seq_len"),
    "--pred_len": flag_pattern("--pred_len"),
    "--model": flag_pattern("--model"),
    "--model_id": flag_pattern("--model_id"),
}

E_LAYERS_RE = re.compile(r"(?<!\S)--e_layers(?:=|\s+)[^\s\\]+")
MODEL_RE = re.compile(r"(?<!\S)--model(?:=|\s+)[^\s\\]+")
MODEL_ID_RE = re.compile(r"(?<!\S)--model_id(?:=|\s+)[^\s\\]+")


def get_flag(text: str, name: str) -> Optional[str]:
    match = PATTERNS[name].search(text)
    return match.group(1).strip("'\"") if match else None


def dataset_name(text: str) -> Optional[str]:
    data = get_flag(text, "--data")
    data_path = get_flag(text, "--data_path")

    if data and data.lower() != "custom":
        return data

    if not data_path:
        return None

    stem = Path(data_path).stem
    lower = stem.lower()

    aliases = {
        "electricity": "electricity",
        "weather": "weather",
        "traffic": "traffic",
        "solar_al": "Solar",
        "etth1": "ETTh1",
        "etth2": "ETTh2",
        "ettm1": "ETTm1",
        "ettm2": "ETTm2",
    }
    return aliases.get(lower, stem)


def compact_spaces(text: str) -> str:
    output = []
    for line in text.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        body = re.sub(r"[ \t]{2,}", " ", body)
        body = re.sub(r"[ \t]+\\$", r" \\", body)
        output.append(body.rstrip() + newline)
    return "".join(output)


def normalize(text: str) -> tuple[str, Optional[str], str]:
    if "run.py" not in text:
        return text, None, "no run.py command"

    dataset = dataset_name(text)
    features = get_flag(text, "--features") or "M"
    seq_len = get_flag(text, "--seq_len")
    pred_len = get_flag(text, "--pred_len")

    if not all([dataset, seq_len, pred_len]):
        return text, None, "missing dataset/features/seq_len/pred_len"

    model_id = f"{dataset}_{features}_{seq_len}_{pred_len}"

    new = E_LAYERS_RE.sub("", text)

    if MODEL_RE.search(new):
        new = MODEL_RE.sub("--model ExoPath", new)
    else:
        new = re.sub(
            r"(python(?:\d+(?:\.\d+)*)?\s+-u\s+run\.py)",
            r"\1 --model ExoPath",
            new,
            count=1,
        )

    if MODEL_ID_RE.search(new):
        new = MODEL_ID_RE.sub(f"--model_id {model_id}", new)
    else:
        new = re.sub(
            r"(python(?:\d+(?:\.\d+)*)?\s+-u\s+run\.py)",
            rf"\1 --model_id {model_id}",
            new,
            count=1,
        )

    return compact_spaces(new), model_id, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_root", type=Path, default=Path("script_val"))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    root = args.script_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"script_root does not exist: {root}")

    scanned = 0
    changed = 0

    for path in sorted(root.rglob("*.sh")):
        if path.name.endswith(".bak"):
            continue

        scanned += 1
        old = path.read_text(encoding="utf-8", errors="ignore")
        new, model_id, reason = normalize(old)

        if new == old:
            print(f"[skip] {path}: {reason}")
            continue

        changed += 1
        print(
            f"[edit] {path}: "
            f"--model ExoPath, --model_id {model_id}, remove --e_layers"
        )

        if args.dry_run:
            continue

        if not args.no_backup:
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(path, backup)

        path.write_text(new, encoding="utf-8")

    print(f"\nscanned={scanned}, changed={changed}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
