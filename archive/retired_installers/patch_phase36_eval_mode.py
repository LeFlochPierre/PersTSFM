"""Safe, idempotent patch for Phase 3.4 & 3.6 train/eval-mode leakage.

Usage:
  python scripts/maintenance/patch_phase36_eval_mode.py
  python scripts/maintenance/patch_phase36_eval_mode.py --apply

Only changes two precisely matched fragments in the existing source. Saves
backups under archive/patch_backups/ and will fail if the source is unexpected.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil

BASE = Path('src/experiments/adaptation_benchmark')
PATCHES = {
    BASE / 'models.py': [
        (
            'from .metrics import organise_forecast\n',
            'from .metrics import organise_forecast\nfrom .inference_safety import ensure_eval_mode\n',
        ),
        (
            '        result = np.empty((len(records),horizon,4),dtype=np.float32)\n',
            '        ensure_eval_mode(self.adapter.pipeline)\n'
            '        result = np.empty((len(records),horizon,4),dtype=np.float32)\n',
        ),
    ],
    BASE / 'capability_models.py': [
        (
            'from .artifacts import write_json\n',
            'from .artifacts import write_json\nfrom .inference_safety import ensure_eval_mode, compare_checkpoint_parameters\n',
        ),
        (
            '            self.adapter.pipeline=trained\n'
            '            ref,_,_=self.forecast(records[:1],horizon=horizon,context=context,lag=2,\n',
            '            self.adapter.pipeline=trained\n'
            '            ensure_eval_mode(trained)\n'
            '            ref,_,_=self.forecast(records[:1],horizon=horizon,context=context,lag=2,\n',
        ),
        (
            "            loaded=Chronos2Pipeline.from_pretrained(str(checkpoint),device_map='cuda')\n"
            '            self.adapter.pipeline=loaded\n',
            "            loaded=Chronos2Pipeline.from_pretrained(str(checkpoint),device_map='cuda')\n"
            '            ensure_eval_mode(loaded)\n'
            '            self.adapter.pipeline=loaded\n',
        ),
        (
            "            if not audit['verified_checkpoint_reload']:\n"
            "                raise RuntimeError(f'Saved checkpoint reload changed forecasts (scaled max diff {delta/scale:.5g})')\n",
            "            if not audit['verified_checkpoint_reload']:\n"
            "                audit['trained_model_class']=str(type(trained.model))\n"
            "                audit['loaded_model_class']=str(type(loaded.model))\n"
            "                audit['trained_model_eval']=not trained.model.training\n"
            "                audit['loaded_model_eval']=not loaded.model.training\n"
            "                audit['checkpoint_parameter_comparison']=compare_checkpoint_parameters(trained.model,loaded.model)\n"
            "                write_json(output_dir/'capability_audit_failed.json',audit)\n"
            "                raise RuntimeError(\n"
            "                    f'Saved checkpoint reload still differs in eval mode '\n"
            "                    f'(scaled max diff {delta/scale:.5g}); inspect '\n"
            "                    f'{output_dir}/capability_audit_failed.json'\n"
            "                )\n",
        ),
    ],
}


def transform(text: str, fragments: list[tuple[str, str]]) -> tuple[str, bool]:
    changed = False
    for before, after in fragments:
        if after in text:
            continue
        n = text.count(before)
        if n != 1:
            raise RuntimeError(f'Unexpected source (expected exactly one match; got {n}): {before[:90]!r}')
        text = text.replace(before, after)
        changed = True
    return text, changed


def execute(root: Path, apply: bool) -> list[tuple[str, str]]:
    root = Path(root)
    outputs = []
    staged = []
    # Prepare/validate EVERY target before changing anything.
    for rel, changes in PATCHES.items():
        src = root / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        raw = src.read_text()
        updated, changed = transform(raw, changes)
        staged.append((src, updated, changed, rel))
    helper = root / BASE / 'inference_safety.py'
    if not helper.is_file():
        raise FileNotFoundError(f'Install helper from the ZIP first: {helper}')
    if apply:
        backups = root / 'archive/patch_backups/phase36_eval_mode'
        backups.mkdir(parents=True, exist_ok=True)
        for src, updated, changed, rel in staged:
            if changed:
                dest = backups / rel.name
                if dest.exists() and dest.read_bytes() != src.read_bytes():
                    # A previous backup should not silently be overwritten.
                    raise RuntimeError(f'Backup already exists and differs: {dest}')
                if not dest.exists():
                    shutil.copy2(src, dest)
                src.write_text(updated)
            outputs.append((str(rel), 'PATCHED' if changed else 'ALREADY PATCHED'))
    else:
        outputs = [(str(rel), 'WOULD PATCH' if changed else 'ALREADY PATCHED')
                   for src, updated, changed, rel in staged]
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--root', type=Path, default=Path('.'))
    a = p.parse_args()
    for name, status in execute(a.root, a.apply):
        print(f'{status}: {name}')
    if not a.apply:
        print('Dry-run only. Re-run with --apply to modify source files.')
    else:
        print('PASS: inference-mode patch installed (backups saved).')


if __name__ == '__main__':
    main()
