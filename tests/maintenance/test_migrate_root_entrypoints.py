from pathlib import Path
import importlib.util
import subprocess
import sys
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'maintenance' / 'migrate_root_entrypoints.py'
spec = importlib.util.spec_from_file_location('root_migration', SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def repo(tmp_path):
    for directory in ('src', 'scripts', 'tests'):
        (tmp_path / directory).mkdir()
    (tmp_path / 'README.md').write_text('repo\n')
    p = tmp_path / 'scripts/phase3/PersTSFM_phase3_0_probe_value.py'
    p.write_text('def value(): return 42\n')
    q = tmp_path / 'scripts/phase3/PersTSFM_phase3_1_adaptive_racing.py'
    q.write_text('from PersTSFM_phase3_0_probe_value import value\nassert value() == 42\n')
    t = tmp_path / 'tests' / 'test_script.py'
    t.write_text('''from pathlib import Path
import importlib.util
from scripts.phase3.PersTSFM_phase3_0_probe_value import value
assert value() == 42
p = Path(__file__).resolve().parents[1] / 'scripts/phase3/PersTSFM_phase3_1_adaptive_racing.py'
assert p.is_file()
def test_actual():
    assert value() == 42
''')
    return tmp_path


def test_dry_run_does_not_mutate(tmp_path):
    root = repo(tmp_path)
    before = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert m.main(['--root', str(root)]) == 0
    after = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert before == after


def test_move_rewrite_import_and_path_then_restore(tmp_path):
    root = repo(tmp_path)
    assert m.main(['--root', str(root), '--apply']) == 0
    assert not (root / 'scripts/phase3/PersTSFM_phase3_0_probe_value.py').exists()
    assert (root / 'scripts/phase3/PersTSFM_phase3_0_probe_value.py').is_file()
    text = (root / 'tests/test_script.py').read_text()
    assert 'from scripts.phase3.PersTSFM_phase3_0_probe_value import value' in text
    assert "'scripts/phase3/PersTSFM_phase3_1_adaptive_racing.py'" in text
    code = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_script.py'],
                          cwd=root, text=True, capture_output=True)
    assert code.returncode == 0, code.stdout + code.stderr
    manifest = next((root / 'archive/manifests').rglob('manifest.json'))
    assert m.main(['--root', str(root), '--restore', str(manifest)]) == 0
    assert (root / 'scripts/phase3/PersTSFM_phase3_0_probe_value.py').is_file()
    assert not (root / 'scripts/phase3/PersTSFM_phase3_0_probe_value.py').exists()
    assert 'from PersTSFM_phase3_0_probe_value import value' in (root / 'tests/test_script.py').read_text()


def test_collisions_refuse_without_mutation(tmp_path):
    root = repo(tmp_path)
    p = root / 'scripts/phase3/PersTSFM_phase3_0_probe_value.py'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('WRONG\n')
    with pytest.raises(RuntimeError, match='collision'):
        m.plan(root)
    assert (root / 'scripts/phase3/PersTSFM_phase3_0_probe_value.py').exists()


def test_restore_rejects_post_migration_edit(tmp_path):
    root=repo(tmp_path)
    m.main(['--root',str(root),'--apply'])
    p=root/'scripts/phase3/PersTSFM_phase3_0_probe_value.py'
    p.write_text('# edited after migration\n')
    manifest=next((root/'archive/manifests').rglob('manifest.json'))
    with pytest.raises(RuntimeError, match='modified since migration'):
        m.restore(root,manifest)
    assert p.exists()


def test_installer_archived(tmp_path):
    root=repo(tmp_path)
    (root/'PersTSFM_phase3_3a_fix_rolling_history.py').write_text('print("legacy one-off patch")\n')
    moves, edits=m.plan(root)
    assert any(dest.relative_to(root).as_posix() == 'archive/installers/PersTSFM_phase3_3a_fix_rolling_history.py' for _,dest in moves)
    m.apply(root,moves,edits)
    assert (root/'archive/installers/PersTSFM_phase3_3a_fix_rolling_history.py').exists()
