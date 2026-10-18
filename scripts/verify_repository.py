"""Check notebook hygiene, syntax, configuration and local Markdown links."""
import ast
import json
import re
from pathlib import Path
from notebook_tools import (REPO_ROOT, BASELINE_NOTEBOOK, WEIGHTED_NOTEBOOK,
                            literal_settings)


def main():
    errors = []
    notebooks = sorted((REPO_ROOT / 'notebooks').glob('*.ipynb'))
    if len(notebooks) != 3:
        errors.append('Expected three research notebooks')
    cell_count = 0
    for path in notebooks:
        nb = json.loads(path.read_text(encoding='utf-8'))
        if nb.get('nbformat') != 4:
            errors.append(f'{path.name}: not notebook format 4')
        ids = []
        for i, cell in enumerate(nb['cells']):
            ids.append(cell.get('id'))
            if cell['cell_type'] != 'code':
                continue
            cell_count += 1
            if cell.get('outputs') or cell.get('execution_count') is not None:
                errors.append(f'{path.name}:{i}: outputs/counters need review before committing')
            source = '\n'.join(
                'pass' if line.lstrip().startswith('%pip ') else line
                for line in ''.join(cell['source']).splitlines())
            try:
                ast.parse(source)
            except SyntaxError as exc:
                errors.append(f'{path.name}:{i}: {exc}')
        if None in ids or len(ids) != len(set(ids)):
            errors.append(f'{path.name}: missing or duplicate cell IDs')
    baseline, weighted = literal_settings(BASELINE_NOTEBOOK), literal_settings(WEIGHTED_NOTEBOOK)
    controlled = ['RANDOM_SEED', 'WINDOW_TOKENS', 'TRAIN_HOP_TOKENS', 'EVAL_HOP_TOKENS',
                  'MIN_WINDOW_TOKENS', 'EMBEDDING_DIM', 'HIDDEN_SIZE', 'NUM_LAYERS',
                  'DROPOUT', 'BATCH_SIZE', 'MAX_EPOCHS', 'EARLY_STOPPING_PATIENCE',
                  'LEARNING_RATE', 'WEIGHT_DECAY', 'GRAD_CLIP']
    for key in controlled:
        if key not in baseline or key not in weighted or baseline[key] != weighted[key]:
            errors.append(f'Controlled setting differs or is missing: {key}')
    if baseline.get('USE_CLASS_WEIGHTS') is not False or weighted.get('USE_CLASS_WEIGHTS') is not True:
        errors.append('Baseline and weighted objective flags are incorrect')
    for path in REPO_ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if target.startswith(('https://', 'http://', '#', 'mailto:')):
                continue
            local = target.split('#', 1)[0]
            if local and not (path.parent / local).exists():
                errors.append(f'{path.relative_to(REPO_ROOT)}: missing link {local}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {len(notebooks)} notebooks, {cell_count} code cells, controlled settings and local links')


if __name__ == '__main__':
    main()
