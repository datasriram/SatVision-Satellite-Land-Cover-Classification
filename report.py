"""Build a comparison table only from completed experiments and saved metrics."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    rows = []
    for name in ['cnn', 'mobilenet', 'cnn_rotation']:
        folder = ROOT / 'results' / name
        record = json.loads((folder / 'run.json').read_text())
        if record['status'] != 'completed':
            raise ValueError(f'{name} has not completed training and validation')
        for split in ['validation', 'test']:
            metrics = json.loads((folder / split / 'metrics.json').read_text())
            rows.append({'model': name, 'split': split, **metrics,
                         'training_seconds': record['seconds'],
                         'epochs': record['epochs_completed'],
                         'best_epoch': record['best_epoch_by_val_loss']})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / 'results' / 'comparison.csv', index=False)
    print(table.to_string(index=False))
    baseline = table[(table.model == 'cnn') & (table.split == 'validation')].iloc[0]
    rotation = table[(table.model == 'cnn_rotation') & (table.split == 'validation')].iloc[0]
    experiment = {'change': 'Add training-only random rotations up to 90 degrees in either direction',
                  'baseline': 'cnn', 'candidate': 'cnn_rotation', 'selection_data': 'validation only',
                  'accuracy_difference_percentage_points': float(100*(rotation.accuracy-baseline.accuracy)),
                  'macro_f1_difference': float(rotation.macro_f1-baseline.macro_f1),
                  'test_used_for_development': False}
    for name in ['cnn', 'cnn_rotation']:
        folder = ROOT / 'results' / name / 'validation'
        matrix = pd.read_csv(folder / 'confusion_matrix.csv', index_col=0)
        per_class = pd.read_csv(folder / 'classification_report.csv', index_col=0)
        experiment[name] = {
            'highway_recall': float(per_class.loc['Highway', 'recall']),
            'highway_predicted_as_river': int(matrix.loc['Highway', 'River']),
            'river_predicted_as_highway': int(matrix.loc['River', 'Highway'])}
    (ROOT / 'results' / 'experiment_comparison.json').write_text(json.dumps(experiment, indent=2))

if __name__ == '__main__':
    main()
