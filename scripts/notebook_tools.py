"""Read selected notebook definitions without executing downloads or training."""
import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_NOTEBOOK = REPO_ROOT / 'notebooks' / '01_Wikifonia_Preprocessing_and_TopN_Encoding.ipynb'
BASELINE_NOTEBOOK = REPO_ROOT / 'notebooks' / '02_Causal_LSTM_Top200_Top250_Baseline.ipynb'
WEIGHTED_NOTEBOOK = REPO_ROOT / 'notebooks' / '03_Causal_LSTM_Top200_ClassWeighted.ipynb'


def source_tree(path):
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    source = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        source.extend(
            'pass' if line.lstrip().startswith('%pip ') else line
            for line in ''.join(cell['source']).splitlines()
        )
        source.append('')
    return ast.parse('\n'.join(source), filename=str(path))


def load_definitions(path, names, namespace):
    """Execute only named top-level class/function definitions supplied by this repo."""
    wanted = set(names)
    selected = [node for node in source_tree(path).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    found = {node.name for node in selected}
    if found != wanted:
        raise ValueError(f'Missing definitions: {sorted(wanted - found)}')
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def literal_settings(path):
    result = {}
    for node in source_tree(path).body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = value
    return result
