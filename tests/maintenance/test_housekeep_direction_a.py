import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/maintenance/housekeep_direction_a.py'


def setup_repo(tmp_path):
    root = tmp_path / 'project'
    root.mkdir()
    subprocess.run(['git','init','-q'],cwd=root,check=True)
    (root/'PersTSFM_phase3_6_eval_mode_fix.zip').write_bytes(b'test archive')
    (root/'tatus --short').write_text('accidental typo\n')
    (root/'src').mkdir()
    (root/'src'/'code.py').write_text('stay untouched\n')
    (root/'results').mkdir()
    (root/'results'/'important.json').write_text('{}')
    (root/'pertsfm_phase2_benchmark').mkdir()
    (root/'pertsfm_phase2_benchmark'/'data.csv').write_text('a,b\n')
    return root


def run(root,*args):
    return subprocess.run([sys.executable,str(SCRIPT),'--repo',str(root),*args], text=True,capture_output=True)


def test_dry_run_does_not_move(tmp_path):
    root=setup_repo(tmp_path)
    r=run(root)
    assert r.returncode==0, r.stderr
    assert (root/'PersTSFM_phase3_6_eval_mode_fix.zip').exists()
    assert not (root/'archive/patches/phase3').exists()


def test_apply_and_restore_preserve_protected(tmp_path):
    root=setup_repo(tmp_path)
    r=run(root,'--apply')
    assert r.returncode==0, r.stderr
    manifest=next((root/'archive/maintenance').glob('housekeeping_*.json'))
    dat=json.loads(manifest.read_text())
    assert len(dat['moves'])==2
    assert (root/'archive/patches/phase3/PersTSFM_phase3_6_eval_mode_fix.zip').exists()
    assert (root/'archive/triage/tatus --short').exists()
    assert (root/'src/code.py').read_text()=='stay untouched\n'
    assert (root/'results/important.json').exists()
    assert (root/'pertsfm_phase2_benchmark/data.csv').exists()
    r=run(root,'--restore',str(manifest))
    assert r.returncode==0, r.stderr
    assert (root/'PersTSFM_phase3_6_eval_mode_fix.zip').read_bytes()==b'test archive'
    assert (root/'tatus --short').exists()


def test_conflict_is_skipped(tmp_path):
    root=setup_repo(tmp_path)
    (root/'archive/patches/phase3').mkdir(parents=True)
    (root/'archive/patches/phase3/PersTSFM_phase3_6_eval_mode_fix.zip').write_bytes(b'occupied')
    r=run(root,'--apply')
    assert r.returncode==0
    assert (root/'PersTSFM_phase3_6_eval_mode_fix.zip').exists()
    assert (root/'archive/patches/phase3/PersTSFM_phase3_6_eval_mode_fix.zip').read_bytes()==b'occupied'
