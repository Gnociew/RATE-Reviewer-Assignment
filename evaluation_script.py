import argparse
import json
import os
from glob import glob
import numpy as np


def compute_pairwise_weighted_kendall_loss(
    examples: list[dict],
    positive_key: str = "positive",
    negative_key: str = "negative",
    pred_field: str = "pred_score",
    ref_field: str = "score",
) -> dict:
    max_loss = 0.0
    loss = 0.0
    used = 0
    skipped = 0

    for ex in examples:
        pos = (ex or {}).get(positive_key) or {}
        neg = (ex or {}).get(negative_key) or {}

        if pred_field not in pos or pred_field not in neg or ref_field not in pos or ref_field not in neg:
            skipped += 1
            continue

        pred_diff = float(pos[pred_field]) - float(neg[pred_field])
        true_diff = float(pos[ref_field]) - float(neg[ref_field])

        w = float(np.abs(true_diff))
        if w == 0:
            skipped += 1
            continue

        max_loss += w
        used += 1

        prod = pred_diff * true_diff
        if prod == 0:
            loss += w / 2
        elif prod < 0:
            loss += w

    return {
        "loss": (loss / max_loss) if max_loss > 0 else float("nan"),
        "used": used,
        "skipped": skipped,
        "max_loss": max_loss,
        "raw_loss": loss,
    }


def compute_pairwise_accuracy(
    examples: list[dict],
    positive_key: str = "positive",
    negative_key: str = "negative",
    pred_field: str = "pred_score",
    ref_field: str = "score",
) -> dict:
    correct = 0
    total = 0
    skipped = 0

    for ex in examples:
        pos = (ex or {}).get(positive_key) or {}
        neg = (ex or {}).get(negative_key) or {}

        if pred_field not in pos or pred_field not in neg or ref_field not in pos or ref_field not in neg:
            skipped += 1
            continue

        pred_diff = float(pos[pred_field]) - float(neg[pred_field])
        true_diff = float(pos[ref_field]) - float(neg[ref_field])

        # Ignore pairs where ground truth scores are equal (no preference)
        if true_diff == 0:
            skipped += 1
            continue

        total += 1
        
        # Check if prediction aligns with ground truth
        # true_diff > 0 means positive > negative
        # true_diff < 0 means positive < negative
        if (pred_diff > 0 and true_diff > 0) or (pred_diff < 0 and true_diff < 0):
            correct += 1
        elif pred_diff == 0:
             # Tie in prediction is usually counted as 0.5 correct or wrong. 
             # Here we count it as wrong (0) for strict accuracy, or 0.5?
             # Let's stick to strict inequality for accuracy: correct if sign matches.
             pass

    return {
        "accuracy": (correct / total) if total > 0 else float("nan"),
        "correct": correct,
        "total": total,
        "skipped": skipped
    }


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _label_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _format_loss(v) -> str:
    try:
        f = float(v)
    except Exception:
        return "nan"
    if f != f:
        return "nan"
    return f"{f:.4f}"


def _discover_predictions(prediction_dir: str, algo: str) -> dict[str, dict[str, str]]:
    prediction_dir = os.path.abspath(prediction_dir)
    files = glob(os.path.join(prediction_dir, f"{algo}_*.json"))

    by_split: dict[str, dict[str, str]] = {"pc": {}, "rc": {}}
    for path in sorted(files):
        base = os.path.basename(path)
        if not base.startswith(f"{algo}_") or not base.endswith(".json"):
            continue

        name = base[: -len(".json")]

        for split in ("pc", "rc"):
            prefix = f"{algo}_{split}"
            if not name.startswith(prefix):
                continue

            rest = name[len(prefix) :]
            strategy = "default"
            if rest:
                if not rest.startswith("_"):
                    continue
                # remove leading underscore
                raw_strat = rest[1:]
                if not raw_strat:
                    strategy = "default"
                else:
                    # Check if it matches new format: {pooling}_{adapter}
                    # We can try to split by underscore.
                    # But strategies like "mean" or "max" don't have underscores.
                    # If there are multiple underscores, it's ambiguous, but let's assume standard behavior.
                    # The user mentioned: specter2_pc_mean_classification.json
                    # algo=specter2, split=pc, rest=_mean_classification
                    # raw_strat=mean_classification
                    strategy = raw_strat

            by_split[split][strategy] = path
            break

    return by_split


def _sorted_strategies(strategies: set[str]) -> list[str]:
    if not strategies:
        return []
    others = sorted([s for s in strategies if s != "default"])
    if "default" in strategies:
        return ["default"] + others
    return others


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pred_paths",
        type=str,
        nargs="+",
        help="Prediction JSON files (pc/rc). Each item must contain positive/negative with score + pred_score.",
    )
    group.add_argument(
        "--prediction_dir",
        type=str,
        help="Directory containing prediction files like {algo}_pc_*.json and {algo}_rc_*.json",
    )
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--pred_field", type=str, default="pred_score")
    parser.add_argument("--ref_field", type=str, default="score")
    return parser.parse_args(argv)


def _eval_one_file(path: str, pred_field: str, ref_field: str) -> tuple[str, dict, dict]:
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Prediction file must be a JSON list: {path}")

    # Adapter for flat format (id + score only)
    if data and "positive_score" in data[0]:
        nested_data = []
        for item in data:
            nested_item = {
                "positive": {
                    pred_field: item.get("positive_score"),
                    ref_field: item.get("positive_ref_score")
                },
                "negative": {
                    pred_field: item.get("negative_score"),
                    ref_field: item.get("negative_ref_score")
                }
            }
            nested_data.append(nested_item)
        data = nested_data

    res_loss = compute_pairwise_weighted_kendall_loss(
        data,
        positive_key="positive",
        negative_key="negative",
        pred_field=pred_field,
        ref_field=ref_field,
    )
    res_acc = compute_pairwise_accuracy(
        data,
        positive_key="positive",
        negative_key="negative",
        pred_field=pred_field,
        ref_field=ref_field,
    )
    return _label_from_path(path), res_loss, res_acc


if __name__ == "__main__":
    args = parse_args()

    if args.pred_paths is not None:
        for path in args.pred_paths:
            label, res_loss, res_acc = _eval_one_file(path, args.pred_field, args.ref_field)
            print(f"{label}:  loss: {_format_loss(res_loss['loss'])}  acc: {_format_loss(res_acc['accuracy'])}")

    else:
        if not args.algo:
            raise ValueError("--algo is required when using --prediction_dir")

        mapping = _discover_predictions(args.prediction_dir, args.algo)
        strategies = set(mapping["pc"].keys()) | set(mapping["rc"].keys())
        if not strategies:
            raise ValueError(f"No prediction files found for algo={args.algo} in {args.prediction_dir}")

        for strat in _sorted_strategies(strategies):
            pc_path = mapping["pc"].get(strat)
            rc_path = mapping["rc"].get(strat)
            if pc_path:
                label, res_loss, res_acc = _eval_one_file(pc_path, args.pred_field, args.ref_field)
                print(f"{label}:  loss: {_format_loss(res_loss['loss'])}  acc: {_format_loss(res_acc['accuracy'])}")
            if rc_path:
                label, res_loss, res_acc = _eval_one_file(rc_path, args.pred_field, args.ref_field)
                print(f"{label}:  loss: {_format_loss(res_loss['loss'])}  acc: {_format_loss(res_acc['accuracy'])}")
