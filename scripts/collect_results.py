"""Export a compact, comparable public summary from completed experiment metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'top200_unweighted': ('causal_lstm_tsd24_top200_unk_v3_audited', 200, False),
    'top250_unweighted': ('causal_lstm_tsd24_top250_unk_v3_audited', 250, False),
    'top200_weighted': ('causal_lstm_tsd24_top200_unk_v3_classweighted_audited', 200, True),
}
METRICS = ['slot_accuracy', 'retained_only_slot_accuracy', 'top3_accuracy',
           'macro_f1_including_unk', 'macro_f1_excluding_unk',
           'weighted_f1_excluding_unk', 'chord_onset_accuracy',
           'chord_onset_accuracy_excluding_unk', 'cross_entropy_loss', 'perplexity',
           'normalised_cross_entropy_vs_uniform', 'unk_target_rate', 'unk_prediction_rate',
           'invalid_pad_prediction_rate', 'majority_baseline_slot_accuracy',
           'encoding_train_slot_coverage_pct', 'supervised_test_slots', 'best_epoch']
CONTROLLED = ['random_seed', 'test_split_sha256', 'split_assignment_sha256', 'input_interface_sha256',
              'window_tokens', 'train_hop_tokens', 'eval_hop_tokens',
              'embedding_dim', 'hidden_size', 'num_layers',
              'dropout', 'batch_size', 'max_epochs', 'early_stopping_patience',
              'learning_rate', 'weight_decay', 'grad_clip',
              'max_train_batches', 'max_validation_batches']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True, type=Path)
    args = parser.parse_args()
    output, configs = {}, {}
    for name, (folder, top_n, weighted) in RUNS.items():
        path = args.artifact_root / folder / 'test_metrics.json'
        raw = path.read_bytes()
        payload = json.loads(raw)
        config = payload['experiment_config']
        if config.get('active_top_n') != top_n or config.get('use_class_weights') is not weighted:
            raise ValueError(f'{name}: wrong experimental condition')
        missing = [key for key in CONTROLLED if key not in config]
        if missing:
            raise ValueError(f'{name}: missing configuration fields: {missing}')
        if config['max_train_batches'] is not None or config['max_validation_batches'] is not None:
            raise ValueError(f'{name}: refuses batch-limited runs')
        numbers = {}
        for key in METRICS:
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name}: invalid metric {key}')
            numbers[key] = value
        safe_config = {key: config[key] for key in CONTROLLED}
        safe_config.update(active_top_n=top_n, use_class_weights=weighted)
        configs[name] = safe_config
        output[name] = {'source_metrics_sha256': hashlib.sha256(raw).hexdigest(),
                        'configuration': safe_config, 'metrics': numbers}
    reference = configs['top200_unweighted']
    for name, config in configs.items():
        mismatches = [key for key in CONTROLLED if config[key] != reference[key]]
        if mismatches:
            raise ValueError(f'{name}: controlled settings/splits differ: {mismatches}')
    result_root = REPO_ROOT / 'results'
    result_root.mkdir(exist_ok=True)
    (result_root / 'metrics_summary.json').write_text(
        json.dumps({'scope': 'Exported supplied metrics; not independently rerun', 'runs': output},
                   indent=2, allow_nan=False) + '\n', encoding='utf-8')
    rows = ['# Completed-run comparison', '',
            'Exported from completed metric files. No independent evaluation was run by this exporter.', '',
            '| Metric | Top-200 unweighted | Top-250 unweighted | Top-200 weighted |',
            '| --- | ---: | ---: | ---: |']
    for metric in METRICS:
        cells = [f'{output[name]["metrics"][metric]:.6g}' for name in RUNS]
        rows.append('| ' + metric + ' | ' + ' | '.join(cells) + ' |')
    rows += ['', 'Top-N conditions use different mapped label spaces. Interpret likelihood metrics with coverage and retained-only scores.', '']
    (result_root / 'comparison.md').write_text('\n'.join(rows), encoding='utf-8')
    print('Wrote results/metrics_summary.json and results/comparison.md')


if __name__ == '__main__':
    main()
