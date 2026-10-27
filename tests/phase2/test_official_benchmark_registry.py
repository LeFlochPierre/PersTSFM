"""Contract tests for Phase 2.6 metadata-only official benchmark audit."""

import hashlib
import json

import yaml

from scripts.phase2.audit_official_benchmarks import audit, build_tables, clean_alias


def sample_configs():
    return [
        {"dataset_path": "autogluon/chronos_datasets", "dataset_config": "m4_hourly", "horizon": 24, "num_windows": 2},
        {"dataset_path": "autogluon/chronos_datasets", "dataset_config": "m4_hourly", "horizon": 48, "num_windows": 1},
        {"dataset_path": "autogluon/fev_datasets", "dataset_config": "shop", "horizon": 7, "target": "sales", "known_dynamic_columns": ["promo"]},
        {"dataset_path": "autogluon/fev_datasets", "dataset_config": "ETT_1H", "horizon": 24, "target": ["OT", "LUFL"]},
    ]


def test_family_grouping_over_horizons_and_overlap():
    catalog, families, feasibility, overlaps = build_tables(sample_configs(), (96, 384))
    assert len(catalog) == 4
    assert len(families) == 3
    assert len(feasibility) == 8
    m4 = next(f for f in families if f["family_id_exact"].endswith("/m4_hourly"))
    assert m4["n_tasks"] == 2
    assert json.loads(m4["horizons_json"]) == [24, 48]
    assert len([x for x in overlaps if x["existing_pilot"] == "m4_hourly"]) == 2


def test_features_are_not_silently_dropped():
    rows, *_ = build_tables(sample_configs())
    assert rows[0]["target_only_candidate"] is True
    assert rows[2]["target_only_candidate"] is False
    assert rows[2]["has_covariates"] is True
    assert rows[3]["target_only_candidate"] is False
    assert rows[3]["is_multivariate"] is True


def test_unknown_history_is_not_marked_feasible():
    *_, feasibility, _ = build_tables(sample_configs(), (96, 3072))
    assert all(x["history_feasibility"].startswith("UNKNOWN") for x in feasibility)
    assert all(x["model_context_feasibility"].startswith("UNKNOWN") for x in feasibility)


def test_alias_is_only_review_hint():
    assert clean_alias("monash_M4-Hourly") == "m4_hourly"


def test_end_to_end_offline_fixture(tmp_path):
    data = yaml.safe_dump({"tasks": sample_configs()}).encode()
    source = tmp_path / "fixture.yaml"
    source.write_bytes(data)
    out = tmp_path / "registry"
    manifest = audit(str(source), out, (96, 192), 4)
    assert manifest["task_count"] == 4
    assert manifest["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert (out / "official_tasks.yaml").read_bytes() == data
    assert (out / "task_catalog.csv").is_file()
    assert (out / "family_groups.csv").is_file()
    assert (out / "feasibility.csv").is_file()
    assert (out / "overlap_audit.csv").is_file()
