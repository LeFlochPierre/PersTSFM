from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile

MODULE = Path(__file__).resolve().parents[2] / 'scripts' / 'maintenance' / 'PersTSFM_root_tidy_stage1.py'
spec = importlib.util.spec_from_file_location('tidy', MODULE)
assert spec and spec.loader
import sys
mod = importlib.util.module_from_spec(spec)
sys.modules['tidy'] = mod
spec.loader.exec_module(mod)


def repo(root: Path):
    (root/'README.md').write_text('# repo\n')
    for d in ['scripts/phase3', 'src/models', 'tests/phase3', 'results', 'data', 'archive/manifests']:
        (root/d).mkdir(parents=True, exist_ok=True)
    (root/'scripts/phase3/PersTSFM_phase3_1_adaptive_racing.py').write_text('print("alive")\n')
    (root/'PersTSFM_phase3_3c_timestamp_fix.zip').write_bytes(b'zip-data')
    (root/'README_TIMESTAMP_FIX.md').write_text('docs')
    (root/'PersTSFM_phase3_3a_fix_rolling_history.py').write_text('# superseded patch')
    (root/'results'/'experimental.csv').write_text('results-safe')
    (root/'tests'/'phase3'/'test_something.py').write_text('from PersTSFM_phase3_1_adaptive_racing import run\n')
    (root/'scripts'/'phase3'/'live.py').write_text('print(123)\n')


def test_preview_apply_restore_preserves_code():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        repo(root)
        mod.repository_check(root)
        refs=mod.scan_references(root, [p.name for p in root.iterdir() if p.is_file()])
        moves, warns = mod.collect_moves(root, refs)
        assert len(moves)==3, (moves, warns)
        assert all('scripts/phase3/PersTSFM_phase3_1_adaptive_racing.py' != m.source for m in moves)
        manifest=mod.apply_moves(root, moves)
        assert (root/'archive/packages/PersTSFM_phase3_3c_timestamp_fix.zip').exists()
        assert (root/'docs/phase3/README_TIMESTAMP_FIX.md').exists()
        assert (root/'archive/installers/PersTSFM_phase3_3a_fix_rolling_history.py').exists()
        assert (root/'results/experimental.csv').read_text()=='results-safe'
        assert (root/'scripts/phase3/PersTSFM_phase3_1_adaptive_racing.py').exists()
        assert (root/'scripts/phase3/live.py').exists()
        assert len(json.loads((root/manifest).read_text())['moves'])==3
        mod.restore(root, manifest)
        assert (root/'PersTSFM_phase3_3c_timestamp_fix.zip').read_bytes()==b'zip-data'
        assert (root/'README_TIMESTAMP_FIX.md').exists()


def test_patch_referenced_by_code_is_not_moved():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        repo(root)
        (root/'tests/phase3/test_something.py').write_text('run("PersTSFM_phase3_3a_fix_rolling_history.py")')
        refs=mod.scan_references(root, ['PersTSFM_phase3_3a_fix_rolling_history.py'])
        moves, warns=mod.collect_moves(root, refs)
        assert 'PersTSFM_phase3_3a_fix_rolling_history.py' not in [m.source for m in moves]
        assert any('KEEP PersTSFM_phase3_3a_fix_rolling_history.py' in w for w in warns)


def test_restore_refuses_changed_archive():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        repo(root)
        refs=mod.scan_references(root, [])
        moves,_=mod.collect_moves(root, refs)
        manifest=mod.apply_moves(root, moves)
        (root/'archive/packages/PersTSFM_phase3_3c_timestamp_fix.zip').write_text('different')
        try:
            mod.restore(root, manifest)
        except RuntimeError as exc:
            assert 'Cannot safely restore' in str(exc)
        else:
            raise AssertionError('Expected restore to refuse modified archive')
        assert not (root/'README_TIMESTAMP_FIX.md').exists(), 'Restore must preflight before moving anything'


def test_traversal_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp).resolve()
        for rel in ['../outside', '/tmp/absolute']:
            try:
                mod.in_root(root, rel)
            except ValueError:
                pass
            else:
                raise AssertionError('Unsafe path accepted')
