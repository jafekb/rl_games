#!/usr/bin/env python3
"""Read scalar tags from a TensorBoard events file."""

import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS_WANTED = {
    "episode/mean_grad_norm",
    "episode/mean_q",
    "episode/steps_survived",
    "episode/mean_huber_loss",
    "episode/mean_td_error",
}


def summarize(path: Path, *, compact: bool = False) -> dict[str, dict[str, float]]:
    acc = EventAccumulator(str(path))
    acc.Reload()
    scalars = acc.Tags().get("scalars", [])
    out = {}
    for tag in TAGS_WANTED:
        if tag not in scalars:
            out[tag] = {}
            continue
        events = acc.Scalars(tag)
        vals = [e.value for e in events if e.value == e.value]
        out[tag] = (
            {"n": len(events), "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}
            if vals
            else {}
        )
        if not compact and events:
            print(f"\n{tag} ({len(events)} points)")
            print("-" * 60)
            for e in events:
                print(f"  step={e.step}  value={e.value}")
            if vals:
                mn, mx, avg = min(vals), max(vals), sum(vals) / len(vals)
                print(f"  -> min={mn:.6g}  max={mx:.6g}  mean={avg:.6g}")
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("--compact", "-c")]
    compact = len(args) < len(sys.argv) - 1
    path = Path(args[0]) if args else Path()
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    if compact:
        summary = summarize(path, compact=True)
        for tag in TAGS_WANTED:
            s = summary.get(tag) or {}
            if s:
                print(
                    f"{tag}: n={s['n']} min={s['min']:.6g} max={s['max']:.6g} mean={s['mean']:.6g}"
                )
            else:
                print(f"{tag}: (not found)")
    else:
        summarize(path, compact=False)


if __name__ == "__main__":
    main()
