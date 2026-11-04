"""Guarded fix for a retained Hugging Face Accelerate BF16 forward wrapper.

Dry-run by default; --apply backs up and patches exactly three known modules.
"""
from __future__ import annotations
import argparse
import datetime as dt
import difflib
from pathlib import Path
import shutil

CORE = Path('src/experiments/adaptation_benchmark')

HELPER = '''

def normalize_post_training_pipeline(pipeline) -> bool:
    """Remove Accelerate's training autocast forward wrapper *after* fit().

    The in-memory PEFT model may still use BF16 autocast from Hugging Face
    Trainer, unlike the fresh checkpoint reload. Never remove this wrapper
    during training; do so before comparing or using model predictions.

    Returns whether a retained wrapper was observed. Strict parity checks
    remain mandatory and are NOT bypassed by this helper.
    """
    model = getattr(pipeline, 'model', None)
    if model is None:
        raise RuntimeError('Chronos pipeline has no model to normalize')
    had_wrapper = '_original_forward' in vars(model)
    if had_wrapper:
        from accelerate.utils import extract_model_from_parallel
        result = extract_model_from_parallel(model, keep_fp32_wrapper=False)
        pipeline.model = result
        if '_original_forward' in vars(result):
            raise RuntimeError('Accelerate forward wrapper was not removed')
    ensure_eval_mode(pipeline)
    return had_wrapper
'''


def replace_once(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected one insertion point, got {n}; no files changed')
    return src.replace(old, new, 1)


def transform(kind: str, src: str) -> str:
    if kind == 'inference_safety.py':
        if 'def normalize_post_training_pipeline(' in src:
            return src
        if 'def ensure_eval_mode(pipeline)' not in src:
            raise RuntimeError('Unexpected inference_safety.py: eval guard missing')
        return src + HELPER
    if kind == 'models.py':
        if 'normalize_post_training_pipeline(fitted)' in src:
            return src
        s = replace_once(src, 'from .inference_safety import ensure_eval_mode',
                         'from .inference_safety import ensure_eval_mode, normalize_post_training_pipeline', kind)
        return replace_once(s, '        self.adapter.pipeline = fitted\n        return elapsed,',
                            '        normalize_post_training_pipeline(fitted)\n        self.adapter.pipeline = fitted\n        return elapsed,', kind)
    if kind == 'capability_models.py':
        if 'normalize_post_training_pipeline(trained)' in src:
            return src
        s = replace_once(src,
            'from .inference_safety import (ensure_eval_mode, compare_checkpoint_parameters,',
            'from .inference_safety import (ensure_eval_mode, normalize_post_training_pipeline, compare_checkpoint_parameters,',kind)
        s = replace_once(s,
            '        torch.cuda.synchronize()\n        duration=float(time.perf_counter()-begin)',
            '        forward_wrapper_removed=normalize_post_training_pipeline(trained)\n        torch.cuda.synchronize()\n        duration=float(time.perf_counter()-begin)',kind)
        return replace_once(s,
            "        audit={**weight,**train,**schedule,'mode':mode,'seed':seed,",
            "        audit={**weight,**train,**schedule,'mode':mode,'seed':seed,\n               'accelerate_forward_wrapper_removed':forward_wrapper_removed,",kind)
    raise KeyError(kind)


def build_changes(repo: Path):
    changes=[]
    for name in ('inference_safety.py', 'models.py', 'capability_models.py'):
        path = repo / CORE / name
        if not path.is_file():
            raise RuntimeError(f'Missing required module: {path}')
        before = path.read_text()
        changes.append((path, before, transform(name, before)))
    return changes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--root', type=Path, default=Path('.'))
    args = ap.parse_args()
    repo = args.root.resolve()
    changes = build_changes(repo)
    print('Phase 3.6 Accelerate BF16 forward cleanup')
    for path, old, new in changes:
        print(f'  {path.relative_to(repo)}: {"ALREADY PATCHED" if old == new else "UPDATE"}')
        if old != new and not args.apply:
            diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                            fromfile='before', tofile='after', lineterm=''))
            print('\n'.join(diff[:38]))
    if not args.apply:
        print('DRY RUN. Apply with --apply after reviewing.'); return
    changes = [(p, old, new) for p, old, new in changes if old != new]
    if not changes:
        print('No changes needed.'); return
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = repo / 'archive' / 'patch_backups' / ('phase36_accelerate_forward_' + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    for path, old, new in changes:
        shutil.copy2(path, backup / path.name)
    for path, old, new in changes:
        path.write_text(new)
    print('PASS: applied targeted patch; original files backed up:', backup)
    print('NEXT: python -m pytest -q tests/phase2 tests/phase3')

if __name__ == '__main__':
    main()
