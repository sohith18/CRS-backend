#!/usr/bin/env python3
"""
run_evaluation.py — Automated CRS evaluation with simulated users.

Usage:
    python run_evaluation.py                          # default: 5 personas x 5 runs
    python run_evaluation.py --runs 3                 # 3 runs per persona
    python run_evaluation.py --persona open_space_lover --runs 1  # single persona
    python run_evaluation.py --backend http://localhost:8000

Outputs (timestamped):
    eval_raw_<ts>.json       — full session logs
    eval_metrics_<ts>.json   — aggregate metrics
    eval_table_<ts>.csv      — paper-ready table
    eval_questions_<ts>.txt  — all questions generated (for qualitative review)
"""

import argparse
import csv
import json
import time
import re
from datetime import datetime
from pathlib import Path

import httpx
from simulated_user import SimulatedUser


# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_BACKEND   = "http://localhost:8000"
DEFAULT_RUNS_EACH = 5
POLL_INTERVAL_S   = 1.5
MAX_POLLS         = 400
IMAGE_PREFIX      = "/image/"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_one_session(
    persona_name: str,
    session_id:   str,
    backend_url:  str,
    verbose:      bool = True,
) -> dict:
    """
    Run one complete CRS session with a simulated user persona.
    Returns a dict of all session metrics and logs.
    """
    sim_user = SimulatedUser(persona_name)

    # ── Reset previous session, then start fresh ──────────────────────────
    httpx.post(f"{backend_url}/reset",  timeout=10)
    time.sleep(0.8)
    httpx.post(f"{backend_url}/start",json={} ,  timeout=10)

    metrics = {
        # Identity
        "session_id":           session_id,
        "persona":              persona_name,
        "timestamp":            datetime.now().isoformat(),
        # Efficiency
        "turns":                0,
        "invalid_questions":    0,
        "ambiguous_turns":      0,   # VLM returned "both" → counted from process log
        "unsure_turns":         0,   # detect_unsure fired → counted from user_context.unsure
        # Recommendation
        "final_count":          0,
        "final_layouts":        [],
        # Dialogue log
        "questions":            [],
        "answers":              [],
        "eliminated_per_turn":  [],  # remaining count after each answer
        # Status
        "completed":            False,
        "error":                None,
    }

    prev_remaining = None

    # ── Poll loop ──────────────────────────────────────────────────────────
    for poll_n in range(MAX_POLLS):
        time.sleep(POLL_INTERVAL_S)

        try:
            state = httpx.get(f"{backend_url}/question", timeout=10).json()
        except Exception as exc:
            metrics["error"] = f"Poll {poll_n}: HTTP error — {exc}"
            break

        status = state.get("status", "unknown")

        # ── Terminal states ────────────────────────────────────────────────
        if status == "done":
            metrics["completed"]     = True
            metrics["final_count"]   = state.get("remaining_count", 0)
            metrics["final_layouts"] = [
                r["id"] for r in (state.get("result") or [])
            ]
            if verbose:
                print(f"  ✅ Done — {metrics['final_count']} layouts recommended "
                      f"in {metrics['turns']} turns")
            break

        if status == "error":
            metrics["error"] = state.get("error", "unknown error")
            if verbose:
                print(f"  ❌ Error: {metrics['error']}")
            break

        # ── Waiting for answer ─────────────────────────────────────────────
        if status == "waiting":
            question  = state.get("question", "")
            images    = state.get("comparison_images") or {}
            turn      = state.get("turn", 0)
            remaining = state.get("remaining_count", 0)

            metrics["turns"] = turn

            # Track unsure count from user_context changes
            user_ctx = state.get("user_context") or {}
            metrics["unsure_turns"] = len(user_ctx.get("unsure", []))

            # Track eliminated this turn
            if prev_remaining is not None and remaining != prev_remaining:
                metrics["eliminated_per_turn"].append(prev_remaining - remaining)
            prev_remaining = remaining

            if not question:
                continue

            # Validate question
            is_invalid = (
                not question.strip()
                or "?" not in question
                or len(question) > 160
                or bool(re.search(r'\bimage\s*[12]\b', question, re.IGNORECASE))
            )
            if is_invalid:
                metrics["invalid_questions"] += 1
                if verbose:
                    print(f"  ⚠  Turn {turn} — INVALID question: '{question}'")

            # Strip /image/ prefix, then resolve relative to backend root (one level up)
            BACKEND_ROOT = Path(__file__).parent.parent   # eval/ → Qwen/

            def resolve_img(url: str) -> str:
                rel = url.lstrip("/").replace("image/", "", 1)   # "images/layout1.png"
                return str(BACKEND_ROOT / rel)                    # "/home/sri/Qwen/images/layout1.png"

            img_a = resolve_img(images.get("left",  ""))
            img_b = resolve_img(images.get("right", ""))

            # Simulated user answers
            answer = sim_user.answer(question, img_a, img_b)

            metrics["questions"].append(question)
            metrics["answers"].append(answer)

            if verbose:
                print(f"  [Turn {turn:02d}] Q: {question}")
                print(f"  [Turn {turn:02d}] A: {answer}")
                print()

            # Submit answer to backend
            try:
                httpx.post(
                    f"{backend_url}/answer",
                    json={"answer": answer},
                    timeout=10,
                )
            except Exception as exc:
                metrics["error"] = f"Answer POST failed: {exc}"
                break

        # status == "processing" / "initializing" → just keep polling

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def safe_mean(values: list) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def compute_metrics(results: list[dict], label: str = "all") -> dict:
    """Compute aggregate metrics from a list of session results."""
    if not results:
        return {"label": label, "n": 0, "error": "no results"}

    completed = [r for r in results if r["completed"]]
    n         = len(results)
    nc        = len(completed)

    if not completed:
        return {"label": label, "n": n, "completed": 0,
                "completion_rate": 0.0, "error": "no completed sessions"}

    turns_list        = [r["turns"]             for r in completed]
    unsure_list       = [r["unsure_turns"]       for r in completed]
    invalid_list      = [r["invalid_questions"]  for r in completed]
    final_count_list  = [r["final_count"]        for r in completed]
    total_q_list      = [len(r["questions"])      for r in completed]

    # Total layouts is inferred from first session's initial remaining — approximate
    # as max(final_count) + sum(eliminated). Use 10 as fallback.
    try:
        total_layouts = max(
            r["final_count"] + sum(r.get("eliminated_per_turn", []))
            for r in completed
        )
    except Exception:
        total_layouts = 10

    total_turns = sum(turns_list)
    total_qs    = sum(total_q_list)
    avg_final   = safe_mean(final_count_list)

    return {
        "label":                        label,
        "n_sessions":                   n,
        "n_completed":                  nc,

        # ── CRS Efficiency ────────────────────────────────────────────────
        "avg_turns_to_completion":      safe_mean(turns_list),
        "min_turns":                    min(turns_list),
        "max_turns":                    max(turns_list),
        "std_turns":                    round(
            (sum((t - safe_mean(turns_list))**2 for t in turns_list) / nc) ** 0.5, 3
        ),
        "elimination_rate_per_turn":    round(
            (total_layouts - avg_final) / max(safe_mean(turns_list), 1), 3
        ),

        # ── Question Quality ──────────────────────────────────────────────
        "total_questions_generated":    total_qs,
        "invalid_question_count":       sum(invalid_list),
        "question_validity_rate":       round(
            1 - (sum(invalid_list) / max(total_qs, 1)), 4
        ),
        "unsure_rate":                  round(
            safe_mean(unsure_list) / max(safe_mean(turns_list), 1), 4
        ),

        # ── Recommendation ────────────────────────────────────────────────
        "avg_final_recommendations":    avg_final,
        "min_final_recommendations":    min(final_count_list),
        "max_final_recommendations":    max(final_count_list),
        "completion_rate":              round(nc / n, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# REPORT WRITERS
# ══════════════════════════════════════════════════════════════════════════════
def write_csv(agg_all: dict, per_persona: dict[str, dict], path: str):
    """Write a paper-ready CSV table."""
    rows = []

    # Overall
    rows.append(["OVERALL", "", ""])
    rows.append(["Metric", "Value", "Notes"])
    skip = {"label", "n_sessions", "n_completed", "error"}
    for k, v in agg_all.items():
        if k not in skip and not isinstance(v, dict):
            rows.append([k.replace("_", " ").title(), v, ""])

    rows.append(["", "", ""])

    # Per-persona
    rows.append(["PER-PERSONA BREAKDOWN", "", ""])
    metric_keys = [k for k in agg_all if k not in skip
                   and not isinstance(agg_all[k], dict)]
    rows.append(["Metric"] + list(per_persona.keys()))
    for mk in metric_keys:
        row = [mk.replace("_", " ").title()]
        for persona_name in per_persona:
            row.append(per_persona[persona_name].get(mk, ""))
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_questions_log(all_results: list[dict], path: str):
    """Write all generated questions to a text file for qualitative review."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("ALL GENERATED QUESTIONS — CRS EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        for r in all_results:
            f.write(f"Session: {r['session_id']} | Persona: {r['persona']}\n")
            f.write("-" * 40 + "\n")
            for i, (q, a) in enumerate(
                zip(r.get("questions", []), r.get("answers", [])), 1
            ):
                f.write(f"  Q{i}: {q}\n")
                f.write(f"  A{i}: {a}\n")
            f.write(f"  → Final layouts: {r.get('final_count', '?')} | "
                    f"Completed: {r.get('completed', False)}\n\n")


def print_summary(agg_all: dict, per_persona: dict):
    """Print a clean summary table to stdout."""
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 65)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    # Column headers
    personas = list(per_persona.keys())
    col_w    = 14
    header   = f"{'Metric':<32}" + "".join(f"{p[:col_w]:>{col_w}}" for p in ["OVERALL"] + personas)
    print(header)
    print("-" * len(header))

    skip = {"label", "n_sessions", "n_completed", "error"}
    for k in agg_all:
        if k in skip or isinstance(agg_all[k], dict):
            continue
        label = k.replace("_", " ").title()[:31]
        row   = f"{label:<32}"
        row  += f"{fmt(agg_all[k]):>{col_w}}"
        for p in personas:
            row += f"{fmt(per_persona[p].get(k, '-')):>{col_w}}"
        print(row)

    print("=" * len(header))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="CRS Automated Evaluation")
    parser.add_argument("--backend",  default=DEFAULT_BACKEND,
                        help="FastAPI backend URL")
    parser.add_argument("--runs",     type=int, default=DEFAULT_RUNS_EACH,
                        help="Number of runs per persona")
    parser.add_argument("--persona",  default=None,
                        help="Run a single persona only")
    parser.add_argument("--quiet",    action="store_true",
                        help="Suppress per-turn output")
    args = parser.parse_args()

    # ── Health check ──────────────────────────────────────────────────────
    try:
        health = httpx.get(f"{args.backend}/health", timeout=5).json()
        print(f"[Eval] Backend healthy — status: {health.get('session_status')}")
    except Exception as exc:
        print(f"[Eval] ❌ Cannot reach backend at {args.backend}: {exc}")
        return

    personas = (
        [args.persona] if args.persona
        else SimulatedUser.list_personas()
    )

    print(f"[Eval] Running {len(personas)} persona(s) × {args.runs} run(s) "
          f"= {len(personas) * args.runs} sessions\n")

    all_results: list[dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for persona in personas:
        print(f"\n{'━' * 65}")
        print(f"  PERSONA: {persona.upper()}")
        print(f"  {SimulatedUser.PERSONAS[persona]['description']}")
        print(f"{'━' * 65}")

        for run in range(1, args.runs + 1):
            session_id = f"{persona}_run{run:02d}"
            print(f"\n[Eval] ── Session {session_id} ──")

            result = run_one_session(
                persona_name=persona,
                session_id=session_id,
                backend_url=args.backend,
                verbose=not args.quiet,
            )
            all_results.append(result)

            print(f"[Eval] Turns: {result['turns']} | "
                  f"Final layouts: {result['final_count']} | "
                  f"Unsure turns: {result['unsure_turns']} | "
                  f"Invalid Qs: {result['invalid_questions']} | "
                  f"✓" if result["completed"] else "✗")
            time.sleep(2.0)

    # ── Compute metrics ────────────────────────────────────────────────────
    agg_all    = compute_metrics(all_results, label="overall")
    per_persona = {
        p: compute_metrics(
            [r for r in all_results if r["persona"] == p], label=p
        )
        for p in personas
    }

    print_summary(agg_all, per_persona)

    # ── Save outputs ───────────────────────────────────────────────────────
    raw_path   = f"eval_raw_{timestamp}.json"
    met_path   = f"eval_metrics_{timestamp}.json"
    csv_path   = f"eval_table_{timestamp}.csv"
    qs_path    = f"eval_questions_{timestamp}.txt"

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open(met_path, "w", encoding="utf-8") as f:
        json.dump({"overall": agg_all, "per_persona": per_persona},
                  f, indent=2, ensure_ascii=False)

    write_csv(agg_all, per_persona, csv_path)
    write_questions_log(all_results, qs_path)

    print(f"\n[Eval] Output files saved:")
    print(f"  Raw sessions : {raw_path}")
    print(f"  Metrics JSON : {met_path}")
    print(f"  Paper CSV    : {csv_path}")
    print(f"  Questions log: {qs_path}")


if __name__ == "__main__":
    main()
