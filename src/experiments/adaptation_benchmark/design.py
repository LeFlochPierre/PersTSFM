"""Declarative experiment grid, feasibility checks, and selection policy.

No GPU or FEV imports. No use of lag1 target values in any selection function.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .history import Snapshot, Series, case_window, cadence_minutes

@dataclass(frozen=True)
class Trial:
    strategy: str
    context: int
    history: int
    steps: int
    seed: int                 # -1 for zero-shot (seed-independent)

    @property
    def key(self):
        return self.strategy if self.seed == -1 else f'{self.strategy}_seed{self.seed}'

@dataclass(frozen=True)
class Plan:
    task: str
    snapshot_path: str
    snapshot: Snapshot
    records: tuple[Series, ...]
    excluded: tuple[dict, ...]
    config: dict
    cadence: int
    trials: tuple[Trial, ...]


def load_batch(path: Path) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    if raw.get('schema') != 'pertsfm_phase34_batch_v1' or not isinstance(raw.get('experiments'), list):
        raise ValueError('Expected schema pertsfm_phase34_batch_v1 and experiments list')
    items = raw['experiments']
    if not items or any(not isinstance(x, dict) for x in items):
        raise ValueError('At least one experiment object required')
    return items


def integer_grid(items, label, *, min_value):
    if not isinstance(items, list) or not items or any(type(x) is not int or x < min_value for x in items):
        raise ValueError(f'{label}: expected nonempty list of integers >= {min_value}')
    if len(set(items)) != len(items):
        raise ValueError(f'{label}: duplicate values')
    return sorted(items)


def normalise(config: dict) -> dict:
    if config.get('model', 'chronos2') != 'chronos2':
        raise ValueError('Initial runner supports chronos2; model interface allows future backends')
    required = ('task', 'snapshot', 'contexts', 'lora_contexts', 'histories', 'budgets', 'seeds')
    for k in required:
        if k not in config:
            raise ValueError(f'Missing required field {k}')
    cfg = dict(config)
    cfg['model'] = 'chronos2'
    cfg['contexts'] = integer_grid(cfg['contexts'], 'contexts', min_value=2)
    cfg['lora_contexts'] = integer_grid(cfg['lora_contexts'], 'lora_contexts', min_value=2)
    cfg['budgets'] = integer_grid(cfg['budgets'], 'budgets', min_value=1)
    cfg['seeds'] = integer_grid(cfg['seeds'], 'seeds', min_value=0)
    hist = cfg['histories']
    if (not isinstance(hist, list) or not hist or
        any(h != 'full' and (type(h) is not int or h < 1) for h in hist) or
        len(set(map(str, hist))) != len(hist)):
        raise ValueError('histories: distinct positive integers or "full"')
    cfg['histories'] = hist
    cfg['max_entities'] = int(cfg.get('max_entities', 0))
    cfg['batch_size'] = int(cfg.get('batch_size', 8))
    cfg['train_batch_size'] = int(cfg.get('train_batch_size', 8))
    cfg['learning_rate'] = float(cfg.get('learning_rate', 1e-5))
    cfg['invalid_policy'] = cfg.get('invalid_policy', 'error')
    cfg['selection_metric'] = cfg.get('selection_metric', 'validation_mae')
    for k in ('max_entities', 'batch_size', 'train_batch_size'):
        if cfg[k] < (0 if k == 'max_entities' else 1):
            raise ValueError(f'Invalid {k}')
    if cfg['invalid_policy'] not in ('error', 'drop'):
        raise ValueError('invalid_policy must be error or drop')
    if cfg['selection_metric'] not in ('validation_mae', 'validation_pinball3'):
        raise ValueError('Unsupported selection metric')
    if cfg['learning_rate'] <= 0 or not np.isfinite(cfg['learning_rate']):
        raise ValueError('Bad learning rate')
    if not set(cfg['lora_contexts']).issubset(set(cfg['contexts'])):
        raise ValueError('LoRA contexts must be in zero-shot context grid for fair comparisons')
    if max(cfg['contexts']) > 8192:
        raise ValueError('Contexts over 8192 unsupported by current Chronos-2 runner')
    return cfg


def build_plan(cfg: dict, snapshot: Snapshot, snapshot_path: Path) -> Plan:
    cfg = normalise(cfg)
    if snapshot.task != cfg['task']:
        raise ValueError(f"Task mismatch: {snapshot.task} != {cfg['task']}")
    cohort = snapshot.records[:cfg['max_entities']] if cfg['max_entities'] else snapshot.records
    # Resolve full history from the cohort's smallest pre-lag2 prefix.
    available = min(len(r.y)-2*snapshot.horizon for r in cohort)
    histories = sorted(set(available if h == 'full' else h for h in cfg['histories']))
    if min(histories) < max(cfg['lora_contexts']) + 2*snapshot.horizon:
        raise ValueError('LoRA history must be >= max LoRA context + 2*H')
    if max(histories) > available:
        raise ValueError(f'Max history {max(histories)} exceeds smallest pre-lag2 pool {available}')
    selected, excluded = [], []
    for r in cohort:
        try:
            for context in cfg['contexts']:
                case_window(r, snapshot.horizon, context, 2)
                case_window(r, snapshot.horizon, context, 1)  # no lag1 truth
            # The planned training pools must be finite for EVERY candidate.
            if not np.isfinite(r.y[-2*snapshot.horizon-max(histories):-2*snapshot.horizon]).all():
                raise ValueError(f'{r.id}: nonfinite pre-lag2 training history')
            cadence_minutes((r,))
        except (ValueError, RuntimeError) as exc:
            if cfg['invalid_policy'] == 'error':
                raise ValueError(f'{r.id}: incompatible with full matched grid ({exc})') from exc
            excluded.append({'id':r.id, 'reason':str(exc)})
        else:
            selected.append(r)
    if not selected:
        raise ValueError('No compatible entities after deterministic filtering')
    cadence = cadence_minutes(selected)
    # No unequal entity cohorts across configurations; every candidate uses same selected records.
    trials = [Trial(f'zero_c{c}', c, 0, 0, -1) for c in cfg['contexts']]
    for c in cfg['lora_contexts']:
        for h in histories:
            for b in cfg['budgets']:
                for s in cfg['seeds']:
                    trials.append(Trial(f'lora_c{c}_h{h}_s{b}', c, h, b, s))
    if len({t.key for t in trials}) != len(trials):
        raise RuntimeError('Trial key collision')
    return Plan(cfg['task'], str(Path(snapshot_path).resolve()), snapshot,
                tuple(selected), tuple(excluded), cfg, cadence, tuple(trials))


def choose_strategy(validation_rows: list[dict], metric: str = 'validation_mae'):
    """One decision per strategy; LoRA seeds averaged, not cherry-picked."""
    grouped = {}
    for row in validation_rows:
        grouped.setdefault(row['strategy'], []).append(row)
    if not grouped or metric not in ('validation_mae', 'validation_pinball3'):
        raise ValueError('No rows or invalid selection metric')
    out = []
    for strategy, rows in grouped.items():
        scores = np.asarray([r[metric] for r in rows], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError('Invalid validation metrics')
        out.append({'strategy':strategy, 'mean':float(scores.mean()),
                    'std':float(scores.std(ddof=1)) if len(scores)>1 else 0.,
                    'n':len(rows), 'steps':int(rows[0]['steps']),
                    'context':int(rows[0]['context']), 'history':int(rows[0]['history'])})
    # Exact tie favours: no adaptation, fewer steps, less history, shorter context.
    winner = min(out, key=lambda r:(r['mean'], r['steps'], r['history'], r['context'],r['strategy']))
    return winner['strategy'], sorted(out,key=lambda r:(r['mean'],r['steps'])),
