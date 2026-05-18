"""
Unified runner for all LLM data fusion experiments.

Usage:
  python run.py --dataset book
  python run.py --dataset flight --mode di
  python run.py --dataset flight_blind
  python run.py --dataset both --mode dd
"""

import os
import argparse
import datetime

import config
from data.loader               import load_book_claims, load_movie_claims, load_flight_claims, FLIGHT_ATTRS
from data.preprocessor         import preprocess
from evaluation.truth_loader   import load_book_truth, load_movie_truth, load_flight_truth
from evaluation.metrics        import compute_metrics, compute_per_key_metrics, compute_flight_metrics
from llm.prompt_builder        import build_book_prompt, build_movie_prompt, build_flight_prompt
from llm.client                import call_llm
from llm.parser                import parse_response
from llm.parser_flight         import parse_flight_response, parse_flight_di_response, parse_time


MODELS = [
    ("openai", "gpt-4o-mini"),
    # ("openai",    "gpt-4o"),
    # ("anthropic", "claude-sonnet-4-6"),
]

_BOOK_MOVIE_STYLES_DD = [
    {"PROMPT_DOMAIN": "dependent",      "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "dependent",      "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "dependent_c1",   "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "dependent_c1",   "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "dependent_c2",   "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "dependent_c2",   "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "dependent_c1c2", "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "dependent_c1c2", "PROMPT_SHOT": 1},
]
_BOOK_MOVIE_STYLES_DI = [
    {"PROMPT_DOMAIN": "independent",      "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "independent",      "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "independent_c1",   "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "independent_c1",   "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "independent_c2",   "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "independent_c2",   "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "independent_c1c2", "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "independent_c1c2", "PROMPT_SHOT": 1},
]
_FLIGHT_STYLES_DD = [
    {"PROMPT_DOMAIN": "flight_dependent",      "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "flight_dependent",      "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "flight_dependent_c1",   "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "flight_dependent_c1",   "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "flight_dependent_c2",   "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "flight_dependent_c2",   "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "flight_dependent_c1c2", "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "flight_dependent_c1c2", "PROMPT_SHOT": 1},
]
_FLIGHT_STYLES_DI = [
    {"PROMPT_DOMAIN": "flight_independent",      "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "flight_independent",      "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "flight_independent_c1",   "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "flight_independent_c1",   "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "flight_independent_c2",   "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "flight_independent_c2",   "PROMPT_SHOT": 1},
    {"PROMPT_DOMAIN": "flight_independent_c1c2", "PROMPT_SHOT": 0},
    {"PROMPT_DOMAIN": "flight_independent_c1c2", "PROMPT_SHOT": 1},
]

_GATE_ATTRS = {"Departure gate", "Arrival gate"}


def _style_tag() -> str:
    return f"{config.PROMPT_DOMAIN}_{config.PROMPT_SHOT}shot"


# ── Book / Movie ──────────────────────────────────────────────────────────────

def _print_metrics_bm(m: dict):
    print(f"    Recall={m['recall']:.4f}  Precision={m['precision']:.4f}  F1={m['f1']:.4f}"
          f"  (find={m['find']}, missing={m['missing']}, error={m['error']})")


def _save_book_movie(dataset: str, pred_dict: dict, metrics: dict, truth_dict: dict):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    ts    = datetime.datetime.now().strftime("%d%m-%H%M")
    model = config.LLM_MODEL
    style = _style_tag()

    pred_path = os.path.join(config.OUTPUT_DIR, f"truth_result_{model}_{style}_{dataset}_{ts}.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("key\tpredicted\n")
        for key, values in sorted(pred_dict.items(), key=lambda x: str(x[0])):
            f.write(f"{key}\t{'; '.join(sorted(values))}\n")

    metric_path = os.path.join(config.OUTPUT_DIR, f"metrics_{model}_{style}_{dataset}_{ts}.txt")
    with open(metric_path, "w", encoding="utf-8") as f:
        f.write("Model\tStyle\tRecall\tPrecision\tF1\n")
        f.write(f"LLM ({config.LLM_PROVIDER}/{config.LLM_MODEL})\t{style}\t"
                f"{metrics['recall']}\t{metrics['precision']}\t{metrics['f1']}\n")

    per_key = compute_per_key_metrics(truth_dict, pred_dict)

    detail_csv = os.path.join(config.OUTPUT_DIR, f"detail_{model}_{style}_{dataset}_{ts}.csv")
    with open(detail_csv, "w", encoding="utf-8") as f:
        f.write("key,truth,predicted,TP,FN,FP,TP_values,FN_values,FP_values\n")
        for row in per_key:
            f.write(f'"{row["key"]}","{row["truth"]}","{row["predicted"]}",'
                    f'{row["find"]},{row["missing"]},{row["error"]},'
                    f'"{row["tp_values"]}","{row["fn_values"]}","{row["fp_values"]}"\n')

    detail_txt = os.path.join(config.OUTPUT_DIR, f"detail_{model}_{style}_{dataset}_{ts}.txt")
    with open(detail_txt, "w", encoding="utf-8") as f:
        f.write("key\ttruth\tpredicted\tTP\tFN\tFP\tTP_values\tFN_values\tFP_values\n")
        for row in per_key:
            f.write(f'{row["key"]}\t{row["truth"]}\t{row["predicted"]}\t'
                    f'{row["find"]}\t{row["missing"]}\t{row["error"]}\t'
                    f'{row["tp_values"]}\t{row["fn_values"]}\t{row["fp_values"]}\n')


def run_book(styles: list):
    claims  = preprocess(load_book_claims(config.BOOK_CLAIMS_PATH))
    truth   = load_book_truth(config.BOOK_TRUTH_PATH)
    targets = sorted(set(claims.keys()) & set(truth.keys()))
    for style in styles:
        config.PROMPT_DOMAIN = style["PROMPT_DOMAIN"]
        config.PROMPT_SHOT   = style["PROMPT_SHOT"]
        print(f"  [book] {_style_tag()}")
        pred_dict = {}
        for i, isbn in enumerate(targets):
            pred_dict[isbn] = parse_response(call_llm(build_book_prompt(isbn, claims[isbn])))
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(targets)} processed...")
        metrics = compute_metrics(truth, pred_dict)
        _print_metrics_bm(metrics)
        _save_book_movie("book", pred_dict, metrics, truth)


def run_movie(styles: list):
    claims  = preprocess(load_movie_claims(config.MOVIE_CLAIMS_PATH))
    truth   = load_movie_truth(config.MOVIE_TRUTH_PATH)
    targets = sorted(set(claims.keys()) & set(truth.keys()))
    for style in styles:
        config.PROMPT_DOMAIN = style["PROMPT_DOMAIN"]
        config.PROMPT_SHOT   = style["PROMPT_SHOT"]
        print(f"  [movie] {_style_tag()}")
        pred_dict = {}
        for i, (title, year) in enumerate(targets):
            pred_dict[(title, year)] = parse_response(
                call_llm(build_movie_prompt(title, year, claims[(title, year)]))
            )
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(targets)} processed...")
        metrics = compute_metrics(truth, pred_dict)
        _print_metrics_bm(metrics)
        _save_book_movie("movie", pred_dict, metrics, truth)


# ── Flight ────────────────────────────────────────────────────────────────────

def _normalize_truth_dates(truth_dict: dict) -> dict:
    normalized = {}
    for fid, attrs in truth_dict.items():
        norm_attrs = {}
        for attr, val in attrs.items():
            if attr in _GATE_ATTRS:
                norm_attrs[attr] = val.strip().upper()
            else:
                parsed = parse_time(val)
                norm_attrs[attr] = parsed if parsed else val
        normalized[fid] = norm_attrs
    return normalized



def _print_metrics_flight(metrics: dict):
    o = metrics["overall"]
    print(f"    [OVERALL] Recall={o['recall']:.4f}  Precision={o['precision']:.4f}"
          f"  F1={o['f1']:.4f}  (TP={o['tp']}, FN={o['fn']}, FP={o['fp']})")
    for attr, m in metrics["by_attr"].items():
        print(f"      {attr:<25} R={m['recall']:.4f}  P={m['precision']:.4f}  F1={m['f1']:.4f}")


def _save_flight(style_tag: str, claims_path: str, pred_dict: dict, metrics: dict, truth_dict: dict):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%d%m-%H%M")
    model    = config.LLM_MODEL
    date_tag = os.path.basename(claims_path).replace("-data.txt", "")

    metric_path = os.path.join(config.OUTPUT_DIR, f"metrics_{model}_{style_tag}_{date_tag}_{ts}.txt")
    with open(metric_path, "w", encoding="utf-8") as f:
        o = metrics["overall"]
        f.write("Model\tStyle\tDataset\tRecall\tPrecision\tF1\n")
        f.write(f"LLM ({config.LLM_PROVIDER}/{model})\t{style_tag}\tflight\t"
                f"{o['recall']}\t{o['precision']}\t{o['f1']}\n\n")
        f.write("Attribute\tRecall\tPrecision\tF1\tTP\tFN\tFP\n")
        for attr, m in metrics["by_attr"].items():
            f.write(f"{attr}\t{m['recall']}\t{m['precision']}\t{m['f1']}\t"
                    f"{m['tp']}\t{m['fn']}\t{m['fp']}\n")

    detail_path = os.path.join(config.OUTPUT_DIR, f"detail_{model}_{style_tag}_{date_tag}_{ts}.txt")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write("flight_id\tattribute\ttruth\tpredicted\tTP\tFN\tFP\tTP_value\tFN_value\tFP_value\n")
        for fid in sorted(pred_dict.keys()):
            truth_attrs = truth_dict.get(fid, {})
            pred_attrs  = pred_dict[fid]
            for attr in FLIGHT_ATTRS:
                t = truth_attrs.get(attr, "")
                p = pred_attrs.get(attr, "")
                if t and p and t == p:
                    tp, fn, fp = 1, 0, 0
                    tp_val, fn_val, fp_val = t, "", ""
                elif t and p:
                    tp, fn, fp = 0, 1, 1
                    tp_val, fn_val, fp_val = "", t, p
                elif t:
                    tp, fn, fp = 0, 1, 0
                    tp_val, fn_val, fp_val = "", t, ""
                elif p:
                    tp, fn, fp = 0, 0, 1
                    tp_val, fn_val, fp_val = "", "", p
                else:
                    tp, fn, fp = 0, 0, 0
                    tp_val, fn_val, fp_val = "", "", ""
                f.write(f"{fid}\t{attr}\t{t}\t{p}\t{tp}\t{fn}\t{fp}\t{tp_val}\t{fn_val}\t{fp_val}\n")


def run_flight(styles: list, claims_path: str, truth_path: str, blind: bool = False):
    claims  = load_flight_claims(claims_path)
    truth   = _normalize_truth_dates(load_flight_truth(truth_path))
    targets = sorted(set(claims.keys()) & set(truth.keys()))

    anon_ids: dict = {}
    if blind:
        anon_ids = {fid: f"FLIGHT-{i+1:03d}" for i, fid in enumerate(targets)}
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        date_tag     = os.path.basename(claims_path).replace("-data.txt", "")
        mapping_path = os.path.join(config.OUTPUT_DIR, f"flight_id_mapping_{date_tag}.txt")
        with open(mapping_path, "w", encoding="utf-8") as f:
            f.write("anon_id\treal_id\n")
            for fid, anon in sorted(anon_ids.items(), key=lambda x: x[1]):
                f.write(f"{anon}\t{fid}\n")
        print(f"  {len(targets)} flights (IDs anonymized, mapping → {mapping_path})")
    else:
        print(f"  {len(targets)} flights")

    for style in styles:
        config.PROMPT_DOMAIN = style["PROMPT_DOMAIN"]
        config.PROMPT_SHOT   = style["PROMPT_SHOT"]
        is_di     = config.PROMPT_DOMAIN.startswith("flight_independent")
        prefix    = "blind_" if blind else ""
        style_tag = f"{prefix}{_style_tag()}"
        print(f"\n  [{style_tag}]")

        pred_dict = {}
        for j, fid in enumerate(targets):
            flight_key = anon_ids[fid] if blind else fid
            prompt     = build_flight_prompt(flight_key, claims[fid])
            raw        = call_llm(prompt)
            pred_dict[fid] = parse_flight_di_response(raw) if is_di else parse_flight_response(raw)
            if (j + 1) % 20 == 0:
                print(f"    {j+1}/{len(targets)} processed...")

        metrics = compute_flight_metrics(truth, pred_dict)
        _print_metrics_flight(metrics)
        _save_flight(style_tag, claims_path, pred_dict, metrics, truth)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Data Fusion experiments")
    parser.add_argument(
        "--dataset",
        choices=["book", "movie", "both", "flight", "flight_blind"],
        default="both",
        help="Dataset to run (default: both = book + movie)",
    )
    parser.add_argument(
        "--mode",
        choices=["dd", "di", "both"],
        default="both",
        help="Prompt mode: domain-dependent (dd), domain-independent (di), or both (default: both)",
    )
    parser.add_argument("--claims", default=config.FLIGHT_CLAIMS_PATH, help="Flight claims file")
    parser.add_argument("--truth",  default=config.FLIGHT_TRUTH_PATH,  help="Flight truth file")
    args = parser.parse_args()

    for provider, model in MODELS:
        config.LLM_PROVIDER = provider
        config.LLM_MODEL    = model
        print(f"\n=== Model: {model} ===")

        if args.dataset in ("book", "both"):
            styles = ((_BOOK_MOVIE_STYLES_DD if args.mode in ("dd", "both") else []) +
                      (_BOOK_MOVIE_STYLES_DI if args.mode in ("di", "both") else []))
            print("\n→ BOOK")
            run_book(styles)

        if args.dataset in ("movie", "both"):
            styles = ((_BOOK_MOVIE_STYLES_DD if args.mode in ("dd", "both") else []) +
                      (_BOOK_MOVIE_STYLES_DI if args.mode in ("di", "both") else []))
            print("\n→ MOVIE")
            run_movie(styles)

        if args.dataset == "flight":
            styles = ((_FLIGHT_STYLES_DD if args.mode in ("dd", "both") else []) +
                      (_FLIGHT_STYLES_DI if args.mode in ("di", "both") else []))
            print("\n→ FLIGHT")
            run_flight(styles, args.claims, args.truth, blind=False)

        if args.dataset == "flight_blind":
            styles = ((_FLIGHT_STYLES_DD if args.mode in ("dd", "both") else []) +
                      (_FLIGHT_STYLES_DI if args.mode in ("di", "both") else []))
            print("\n→ FLIGHT (blind)")
            run_flight(styles, args.claims, args.truth, blind=True)

    print(f"\nDone. Results → {config.OUTPUT_DIR}/")
