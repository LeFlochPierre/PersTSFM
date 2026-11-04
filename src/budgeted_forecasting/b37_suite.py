"""B3.7 zero-shot benchmark planner and verified resume (no training).

Only constructs commands that call existing B3 freeze/run-action/report APIs.
No GPU inference is performed by inspect/plan/verify.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from pathlib import Path

from . import capability_atlas as b3

MODEL_KEYS = (
    "chronos2", "tirex2", "moirai2_small", "toto2_22m", "toto2_313m", "timesfm3",
)
BASELINES = ("last_value", "seasonal_naive")


def _hash(p: Path) -> str:
    return b3.sha256(Path(p))


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _registry():
    from src.data import registry
    return registry


def _action_file(shards: Path, model: str, hours: float | None):
    if model in BASELINES:
        return shards / f"{model}.csv"
    return shards / f"{model}_{float(hours):g}h.csv"


def validate_action(cohort_file: Path, model: str, hours: float | None, shard: Path) -> str:
    """Reject stale or incomplete resumed shards; never silently trust file presence."""
    shard = Path(shard)
    cohort = Path(cohort_file)
    if not cohort.is_file() or not shard.is_file():
        raise FileNotFoundError(f"Missing cohort/shard: {cohort} / {shard}")
    mpath = shard.with_name(shard.stem + "_manifest.json")
    costpath = shard.with_name(shard.stem + "_costs.csv")
    if not mpath.is_file() or not costpath.is_file():
        raise FileNotFoundError(f"Missing action manifest or costs: {mpath} / {costpath}")
    cm = _read_json(cohort)
    m = _read_json(mpath)
    action = b3.action_name(model, hours)
    if m.get("schema_version") != "pertsfm_b3_shard_v1":
        raise ValueError("Unknown shard manifest schema")
    for field, expected in (
        ("cohort_sha256", _hash(cohort)),
        ("shard_sha256", _hash(shard)),
        ("action_id", action),
        ("model_key", model),
        ("dataset", cm["dataset_id"]),
        ("cases", len(cm["cases"])),
    ):
        if m.get(field) != expected:
            raise ValueError(f"Shard {shard}: manifest {field} mismatch: {m.get(field)} != {expected}")
    if model not in BASELINES and float(m.get("context_hours")) != float(hours):
        raise ValueError("Action context does not match manifest")
    cases = b3.load_cache(shard)
    if len(cases) != len(cm["cases"]):
        raise ValueError("Shard does not contain the exact number of frozen cases")
    expected = {x["case_id"]: x for x in cm["cases"]}
    if {x.case_id for x in cases} != set(expected):
        raise ValueError("Shard case identities mismatch")
    for x in cases:
        rec = expected[x.case_id]
        if x.series_id != rec["series_id"] or x.split != rec["split"]:
            raise ValueError(f"Shard case metadata mismatch: {x.case_id}")
        if set(x.actions) != {action}:
            raise ValueError(f"Shard actions mismatch: {x.case_id}")
        if b3.array_hash(x.truth) != rec["truth_sha256"]:
            raise ValueError(f"Shard truth mismatch: {x.case_id}")
    with costpath.open(newline="", encoding="utf-8") as f:
        c = list(csv.DictReader(f))
    if len(c) != len(expected) or {r["case_id"] for r in c} != set(expected):
        raise ValueError("Cost rows do not match frozen cases")
    return action


def validate_config(config: dict) -> list[dict]:
    if config.get("schema_version") != "pertsfm_b37_grid_v1":
        raise ValueError("Unrecognised B3.7 config")
    mapping = config.get("model_envs", {})
    if set(mapping) != set(MODEL_KEYS):
        raise ValueError("Specify an environment for every one of the 6 TSFMs")
    if any(not isinstance(x, str) or not x.strip() for x in mapping.values()):
        raise ValueError("Empty model environment name")
    exps = config.get("experiments")
    if not isinstance(exps, list) or len({x["dataset"] for x in exps}) != len(exps):
        raise ValueError("Expected uniquely named experiments")
    for x in exps:
        if not isinstance(x.get("active"), bool):
            raise ValueError("Each experiment needs active=true/false")
        if not x.get("contexts_hours") or len(set(x["contexts_hours"])) != len(x["contexts_hours"]):
            raise ValueError("Each experiment needs distinct contexts_hours")
    return exps


def _quoted_run(args: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in args)


def inspect(config: dict, *, print_details: bool = True) -> list[dict]:
    r = _registry()
    exps = validate_config(config)
    results = []
    for x in exps:
        spec = r.get_dataset_spec(x["dataset"])
        contexts = tuple(float(v) for v in x["contexts_hours"])
        if any(v not in spec.context_hours for v in contexts):
            raise ValueError(f"{spec.key}: unregistered context in {contexts}; registered={spec.context_hours}")
        if max(contexts) != float(x["max_context_hours"]):
            raise ValueError(f"{spec.key}: frozen max context must equal max(contexts)")
        exists = spec.protocol_path.is_file() and spec.raw_path.exists()
        rows = entities = max_rows = 0
        if spec.protocol_path.is_file():
            frame = r.load_protocol(spec, split=x.get("protocol_split", spec.default_split))
            rows, entities = len(frame), frame[spec.id_col].nunique()
            max_rows = int(frame.groupby(spec.id_col).size().max())
        cpath = Path(x["cohort"])
        state = "MISSING_COHORT"
        if cpath.is_file():
            c = _read_json(cpath)
            if c.get("schema_version") != "pertsfm_b3_cohort_v1" or c["dataset_id"] != x["dataset"]:
                raise ValueError(f"Cohort mismatch for {x['dataset']}: {cpath}")
            if float(c["max_context_hours"]) != max(contexts):
                raise ValueError(f"Cohort max context differs for {x['dataset']}: {cpath}")
            if spec.protocol_path.is_file() and c["protocol_sha256"] != _hash(spec.protocol_path):
                raise ValueError(f"Cohort protocol hash changed for {x['dataset']}")
            state = "FROZEN"
        row = dict(dataset=x["dataset"], active=x["active"], state=state,
                   protocol_rows=rows, entities=entities, max_origins_per_entity=max_rows,
                   raw_and_protocol_found=exists, registered_contexts=list(spec.context_hours),
                   benchmark_contexts=list(contexts), resolution_minutes=spec.resolution_minutes,
                   horizon_steps=spec.forecast_steps)
        results.append(row)
        if print_details:
            print(f"{x['dataset']:15} | {'ACTIVE' if x['active'] else 'HOLD':6} | {state:14} | "
                  f"protocol={rows} origins/{entities} entities; max/entity={max_rows} | "
                  f"horizon={spec.forecast_steps} | contexts={list(contexts)}")
    return results


def build_plan(config: dict, output_dir: Path) -> list[dict]:
    audit = inspect(config)
    exps = config["experiments"]
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty plan directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", "# Run from PersTSFM repository root. Existing valid shards are validated and skipped.", ""]
    reports = ["#!/usr/bin/env bash", "set -euo pipefail", "# Run only after all required run-action commands succeed.", ""]
    rows=[]
    model_envs=config["model_envs"]
    for x, status in zip(exps, audit):
        if not x["active"]:
            continue
        if status["state"] != "FROZEN":
            raise ValueError(f"Active dataset {x['dataset']} has no frozen cohort at {x['cohort']}. Freeze/validate before planning.")
        cohort = x["cohort"]
        shards = Path(x["shards"])
        acts=[("last_value", None)]
        # Daily seasonal-naive cannot run on histories shorter than one day.
        if b3.context_steps(x["max_context_hours"], status["resolution_minutes"]) >= (1440 // status["resolution_minutes"] if 1440 % status["resolution_minutes"] == 0 else 10**12):
            acts.append(("seasonal_naive", None))
        for model in MODEL_KEYS:
            for hours in x["contexts_hours"]:
                if model == "tirex2" and b3.context_steps(hours, status["resolution_minutes"]) > 2048:
                    rows.append(dict(dataset=x["dataset"], model=model, context_hours=hours,
                                     status="UNSUPPORTED_TIREX_MAX2048", shard="", environment=model_envs[model]))
                    continue
                acts.append((model, float(hours)))
        paths=[]
        commands.append(f"echo '===== {x['dataset']} ====='")
        for model, hours in acts:
            shard = _action_file(shards, model, hours)
            action = b3.action_name(model, hours)
            environment = config.get("baseline_env", "PERTSFM_ENV") if model in BASELINES else model_envs[model]
            if shard.is_file():
                validate_action(Path(cohort), model, hours, shard)
                state = "VERIFIED_EXISTING"
            elif any(p.exists() for p in [shard.with_name(shard.stem + "_manifest.json"),
                                           shard.with_name(shard.stem + "_costs.csv")]):
                raise FileExistsError(f"Partial shard outputs for {shard}; inspect before proceeding")
            else:
                state = "PENDING"
            args = ["python", "-m", "src.budgeted_forecasting.b37_suite", "verify-action",
                    "--cohort", cohort, "--model", model, "--shard", str(shard)]
            if hours is not None: args += ["--context-hours", str(hours)]
            verify_cmd = _quoted_run(args)
            run_args = ["conda", "run", "--no-capture-output", "-n", environment, "python", "-m",
                        "src.budgeted_forecasting.capability_atlas", "run-action", "--cohort", cohort,
                        "--model", model, "--output", str(shard)]
            if hours is not None: run_args += ["--context-hours", str(hours)]
            commands.extend([f"if [[ -f {shlex.quote(str(shard))} ]]; then",
                             f"  {verify_cmd}", "else", f"  {_quoted_run(run_args)}", "fi"])
            rows.append(dict(dataset=x["dataset"], model=model, context_hours="" if hours is None else hours,
                             status=state, shard=str(shard), environment=environment))
            paths.append(shard)
        target = Path(x["report_dir"])
        report_args = ["python", "-m", "src.budgeted_forecasting.capability_atlas", "report", "--cohort", cohort]
        for path in paths:
            report_args += ["--shard", str(path)]
        report_args += ["--output-dir", str(target)]
        reports.append(f"# {x['dataset']}: validate all required shards before joining")
        for model, hours in acts:
            shard = _action_file(shards, model, hours)
            verify_args = ["python", "-m", "src.budgeted_forecasting.b37_suite", "verify-action", "--cohort", cohort,
                           "--model", model, "--shard", str(shard)]
            if hours is not None: verify_args += ["--context-hours", str(hours)]
            reports.append(_quoted_run(verify_args))
        reports.append(_quoted_run(report_args))
    with (output_dir / "actions.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["dataset","model","context_hours","status","shard","environment"])
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "run_missing.sh").write_text("\n".join(commands)+"\n",encoding="utf-8")
    (output_dir / "build_reports.sh").write_text("\n".join(reports)+"\n",encoding="utf-8")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    (output_dir / "config_snapshot.json").write_text(json.dumps(config,indent=2)+"\n")
    print(f"PASS B3.7 plan: {len(rows)} model/context actions, "
          f"{sum(r['status']=='VERIFIED_EXISTING' for r in rows)} verified existing; "
          f"{sum(r['status']=='PENDING' for r in rows)} pending -> {output_dir}")
    return rows


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    for cmd in ("inspect","plan"):
        c=sub.add_parser(cmd); c.add_argument("--config",type=Path,required=True)
        if cmd=="plan": c.add_argument("--output-dir",type=Path,required=True)
    v=sub.add_parser("verify-action")
    v.add_argument("--cohort",type=Path,required=True)
    v.add_argument("--model",required=True,choices=MODEL_KEYS+BASELINES)
    v.add_argument("--context-hours",type=float)
    v.add_argument("--shard",type=Path,required=True)
    args=p.parse_args(argv)
    if args.cmd=="verify-action":
        print("VERIFIED:",validate_action(args.cohort,args.model,args.context_hours,args.shard)); return 0
    config=_read_json(args.config)
    if args.cmd=="inspect": inspect(config)
    elif args.cmd=="plan": build_plan(config,args.output_dir)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
