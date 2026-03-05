#!/usr/bin/env python3
"""Read scalar tags from a TensorBoard events file."""

import argparse
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS_WANTED = [
    "episode/steps_survived",
    "episode/terminal_reward",
    "episode/rolling_win_rate",
    "episode/mean_q",
    "episode/mean_huber_loss",
    "episode/mean_td_error",
    "episode/mean_grad_norm",
]


def _ema(vals: list[float], alpha: float) -> list[float]:
    """TensorBoard EMA: smoothed[i] = alpha*smoothed[i-1] + (1-alpha)*val[i], init to val[0]."""
    if not vals:
        return []
    out = [vals[0]]
    for v in vals[1:]:
        out.append(alpha * out[-1] + (1 - alpha) * v)
    return out


def summarize(
    path: Path,
    *,
    compact: bool = False,
    smooth: float = 0.99,
    tail: int = 0,
) -> dict[str, dict[str, float]]:
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
        if not vals:
            out[tag] = {}
            continue

        smoothed = _ema(vals, smooth)
        tail_smoothed = smoothed[-tail:] if tail > 0 else smoothed

        out[tag] = {
            "n": len(vals),
            "ema_final": smoothed[-1],
            "tail_mean": sum(tail_smoothed) / len(tail_smoothed),
            "tail_min": min(tail_smoothed),
            "tail_max": max(tail_smoothed),
        }

        if not compact:
            print(f"\n{tag} ({len(vals)} points)")
            print("-" * 60)
            for e, sv in zip(events, smoothed):
                print(f"  step={e.step}  raw={e.value:.6g}  smoothed={sv:.6g}")
            s = out[tag]
            print(f"  -> ema_final={s['ema_final']:.6g}")
            if tail > 0:
                print(
                    f"  -> tail{tail} (smoothed): mean={s['tail_mean']:.6g}  "
                    f"min={s['tail_min']:.6g}  max={s['tail_max']:.6g}"
                )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Read scalar tags from a TensorBoard events file.")
    parser.add_argument("path", nargs="?", default=".", help="Directory containing TB event files.")
    parser.add_argument(
        "-c",
        "--compact",
        action="store_true",
        help="One-line summary per tag instead of full dump.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.99,
        metavar="ALPHA",
        help="EMA smoothing factor, same as TensorBoard (default: 0.99).",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=2000,
        metavar="N",
        help="Show stats over the last N smoothed values (default: 500).",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    summary = summarize(path, compact=args.compact, smooth=args.smooth, tail=args.tail)

    if args.compact:
        for tag in TAGS_WANTED:
            s = summary.get(tag) or {}
            if not s:
                print(f"{tag}: (not found)")
                continue
            tail_str = (
                f"  tail{args.tail}(mean={s['tail_mean']:.4g} "
                f"min={s['tail_min']:.4g} max={s['tail_max']:.4g})"
            )
            print(f"{tag}: ema={s['ema_final']:.4g}{tail_str}  [n={s['n']}]")


if __name__ == "__main__":
    main()
