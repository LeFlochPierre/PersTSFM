#!/usr/bin/env python3
"""Non-destructive patch: reconstruct B1.2 cases from exact B1.1 cache origin keys.

Dry-run by default. With --apply, writes a backup under archive/maintenance/
and updates only src/budgeted_forecasting/extend_atlas.py.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
from pathlib import Path

TARGET = Path('src/budgeted_forecasting/extend_atlas.py')
OLD_IMPORT = '''from .real_atlas import (
    OriginCase, _timed_predict, auto_budgets, git_sha,
    load_electricity_pilot, write_csv_atomic,
)'''
NEW_IMPORT = '''from .real_atlas import (
    OriginCase, _timed_predict, auto_budgets, git_sha,
    prepare_cases, write_csv_atomic,
)'''
OLD_FUNCTION = '''def load_matched_dataset(base_cache: Path, base_manifest: Path):
    base = load_cache(base_cache)
    info = json.loads(base_manifest.read_text(encoding="utf-8"))
    if info.get("dataset") != "electricity" or info.get("horizon_steps") != 96:
        raise ValueError("Expected B1.1 electricity baseline manifest (horizon=96)")
    entities = sorted({case.series_id for case in base})
    if entities != sorted(str(x) for x in info["entities"]):
        raise ValueError("Manifest entity list does not match the base cache")
    cal = {case.origin_time for case in base if case.split == "calibration"}
    eva = {case.origin_time for case in base if case.split == "evaluation"}
    if not cal or not eva:
        raise ValueError("Both calibration and evaluation origins are required")
    real, details = load_electricity_pilot(
        entities=len(entities), calibration_origins=len(cal), evaluation_origins=len(eva),
        seed=int(info["seed"]), protocol_split=info["protocol_split"],
    )
    _assert_matched_cases(base, real)
    return base, real, info, details
'''
NEW_FUNCTION = '''def load_matched_dataset(base_cache: Path, base_manifest: Path):
    """Rebuild EXACT baseline cache origins from the frozen validation protocol.

    A B1.1 global-origin selection can give different entities different numbers
    of cases. Resampling by counts or seed is not an exact inverse operation.
    Cache IDs/metadata identify the required cases; raw protocol and observed
    targets independently verify them. All reconstruction remains read-only.
    """
    import pandas as pd
    from src.data.registry import LoadedDataset, _load_electricity, get_dataset_spec, load_protocol

    base = load_cache(base_cache)
    info = json.loads(base_manifest.read_text(encoding="utf-8"))
    if info.get("dataset") != "electricity" or info.get("horizon_steps") != 96:
        raise ValueError("Expected B1.1 electricity baseline manifest (horizon=96)")
    entities = sorted({case.series_id for case in base})
    if entities != sorted(str(x) for x in info["entities"]):
        raise ValueError("Manifest entity list does not match the base cache")
    spec = get_dataset_spec("electricity")
    if spec.key != "electricity" or spec.resolution_minutes != 15 or spec.forecast_steps != 96:
        raise ValueError("Unexpected electricity registry configuration")
    protocol = load_protocol(spec, split=info["protocol_split"]).copy()
    protocol[spec.id_col] = protocol[spec.id_col].astype(str)
    protocol["eval_origin"] = pd.to_datetime(protocol["eval_origin"], errors="raise")
    if protocol["eval_origin"].isna().any():
        raise ValueError("Protocol contains missing evaluation origins")
    if protocol["eval_origin"].dt.tz is not None:
        raise ValueError("Expected timezone-naive electricity protocol origins")
    if protocol.duplicated([spec.id_col, "eval_origin"]).any():
        raise ValueError("Duplicate entity/origin in source protocol")

    # Electricity uses a naive chronological index; B0 cache timestamps are
    # UTC-tagged solely for stable chronology, not as a geographic assertion.
    required = {}
    for case in base:
        if case.dataset_id != "electricity":
            raise ValueError(f"Unexpected dataset in baseline cache: {case.dataset_id}")
        origin = pd.Timestamp(case.origin_time).tz_convert("UTC").tz_localize(None)
        key = (case.series_id, origin)
        if key in required:
            raise ValueError(f"Duplicate baseline origin: {key}")
        required[key] = case

    selected_rows = []
    seen = set()
    for _, row in protocol.iterrows():
        key = (str(row[spec.id_col]), pd.Timestamp(row["eval_origin"]))
        if key in required:
            case = required[key]
            part = row.copy()
            part["split"] = case.split
            selected_rows.append(part)
            seen.add(key)
    missing = sorted(set(required) - seen)
    if missing:
        raise ValueError(f"Baseline origins missing from source protocol: {missing[:5]}")
    selected = pd.DataFrame(selected_rows)
    if len(selected) != len(base):
        raise ValueError("Reconstructed origin count does not match baseline cache")
    selected = selected.sort_values(["eval_origin", spec.id_col, "eval_index"], kind="mergesort")
    series = _load_electricity(spec, entities)
    loaded = LoadedDataset(spec=spec, protocol=selected, series=series)
    real = prepare_cases(loaded, selected)
    _assert_matched_cases(base, real)
    details = {
        "selection": "exact_origin_keys_from_base_cache_checked_against_protocol_and_truth",
        "dataset": "electricity", "protocol_split": info["protocol_split"],
        "entities": entities, "seed_original": info.get("seed"),
        "cases": len(real),
        "calibration_cases": sum(c.split == "calibration" for c in real),
        "evaluation_cases": sum(c.split == "evaluation" for c in real),
    }
    return base, real, info, details
'''


def patched(source: str) -> str:
    if NEW_IMPORT in source and NEW_FUNCTION in source:
        raise SystemExit('ALREADY PATCHED: no changes made')
    for needle in (OLD_IMPORT, OLD_FUNCTION):
        if source.count(needle) != 1:
            raise RuntimeError('Expected exactly one unmodified B1.2 code block; refusing patch')
    new = source.replace(OLD_IMPORT, NEW_IMPORT, 1).replace(OLD_FUNCTION, NEW_FUNCTION, 1)
    compile(new, str(TARGET), 'exec')
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not TARGET.is_file():
        raise SystemExit(f'Missing {TARGET}; run this in the PersTSFM repository root')
    old = TARGET.read_text(encoding='utf-8')
    new = patched(old)
    print('PATCH VALIDATED: exactly matches expected B1.2 implementation')
    print('OLD SHA256:', hashlib.sha256(old.encode()).hexdigest())
    print('NEW SHA256:', hashlib.sha256(new.encode()).hexdigest())
    if not args.apply:
        print('DRY RUN ONLY: pass --apply to write patch with a backup')
        return
    backup_dir = Path('archive/maintenance')
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = backup_dir / f'extend_atlas_before_exact_origin_patch_{stamp}.py'
    if backup.exists():
        raise FileExistsError(f'Backup already exists: {backup}')
    backup.write_text(old, encoding='utf-8')
    TARGET.write_text(new, encoding='utf-8')
    print('PATCH APPLIED:', TARGET)
    print('BACKUP:', backup)


if __name__ == '__main__':
    main()
