import json
from pathlib import Path
import subprocess
import sys

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/maintenance/audit_direction_a_foundation.py'


def test_audit_read_only(tmp_path):
    root=tmp_path/'repo'; root.mkdir()
    subprocess.run(['git','init','-q'],cwd=root,check=True)
    (root/'src').mkdir()
    p=subprocess.run([sys.executable,str(SCRIPT),'--repo',str(root)],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    data=json.loads(p.stdout)
    assert data['report_type']=='foundation_read_only'
    assert data['paths']['data']['exists'] is False
    assert data['paths']['src/models']['exists'] is False
    assert not (root/'archive').exists()


def test_audit_no_overwrite(tmp_path):
    root=tmp_path/'repo'; root.mkdir()
    subprocess.run(['git','init','-q'],cwd=root,check=True)
    path=tmp_path/'report.json'; path.write_text('original')
    p=subprocess.run([sys.executable,str(SCRIPT),'--repo',str(root),'--output',str(path)],capture_output=True,text=True)
    assert p.returncode==2
    assert path.read_text()=='original'
