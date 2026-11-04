#!/usr/bin/env python3
"""Guarded Phase 3.6 patch: audit unmerged PEFT reload before merged diagnostics.

Only installs over exactly the expected previous eval-mode fix. Nothing deleted.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'archive/installers/phase36_unmerged_reload'
DEST = ROOT / 'src/experiments/adaptation_benchmark'
EXPECTED_PREVIOUS = {
    'capability_models.py': '1bce5cc192bb098f7b153e4b1a214c61194cc983451c1531896d3d5a2c0758ae',
    'inference_safety.py': 'da5f6e73147a7da5627c9f08b0306b0a5b919d12a55681987a36ff1be1e01b29',
}


def hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    todo = []
    for name, expected in EXPECTED_PREVIOUS.items():
        src, dst = SOURCE/name, DEST/name
        if not src.is_file() or not dst.is_file():
            raise SystemExit(f'Missing source or destination for {name}; did you unpack the ZIP into repo root?')
        actual = hash_file(dst)
        new_hash = hash_file(src)
        if actual == new_hash:
            print(f'ALREADY UPDATED: {dst.relative_to(ROOT)}')
            continue
        if actual != expected:
            raise SystemExit(f'REFUSING to overwrite locally modified {dst.relative_to(ROOT)}\n'
                             f'Expected SHA256={expected}\nActual SHA256={actual}\n'
                             'Inspect manually; the patch is designed for the previous Phase 3.6 eval-mode release.')
        todo.append((src,dst,actual,new_hash))
        print(f'PATCH: {dst.relative_to(ROOT)}\n  {actual[:16]} -> {new_hash[:16]}')
    if not todo:
        print('PASS: all files already match the new patch')
        return
    if not args.apply:
        print('PREVIEW ONLY. Run again with --apply to back up and install.')
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = ROOT/'archive/patch_backups'/f'phase36_unmerged_reload_{stamp}'
    backup.mkdir(parents=True, exist_ok=False)
    entries=[]
    for src,dst,old_hash,new_hash in todo:
        shutil.copy2(dst, backup/dst.name)
        entries.append({'path':str(dst.relative_to(ROOT)), 'before_sha256':old_hash,
                        'after_sha256':new_hash,'backup':str((backup/dst.name).relative_to(ROOT))})
    (backup/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
    completed=[]
    try:
        for src,dst,_,new_hash in todo:
            shutil.copy2(src,dst)
            if hash_file(dst)!=new_hash:
                raise RuntimeError(f'Installed hash mismatch for {dst}')
            completed.append(dst)
    except Exception:
        for dst in completed:
            shutil.copy2(backup/dst.name,dst)
        raise
    print(f'PASS: {len(completed)} files installed; original code backed up in {backup}')
    print('NEXT: python -m pytest -q tests/phase2 tests/phase3')


if __name__=='__main__': main()
