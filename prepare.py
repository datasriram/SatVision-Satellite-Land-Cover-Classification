"""Inspect local images and save one reproducible split; never downloads data."""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'models' / '.mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    classes = sorted(p.name for p in data.iterdir() if p.is_dir())
    if len(classes) != 10:
        raise ValueError(f'Expected 10 class folders, found {classes}')
    rows, bad, dimensions, hashes = [], [], Counter(), {}
    for label, name in enumerate(classes):
        print(f'Checking {name}', flush=True)
        for path in sorted((data / name).glob('*.jpg')):
            relative = path.relative_to(data).as_posix()
            try:
                with Image.open(path) as img:
                    img.load()  # Decode every file, holding only one image at a time.
                    dimensions[str((img.width, img.height, img.mode))] += 1
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes.setdefault(digest, []).append(relative)
                rows.append({'path': relative, 'class': name, 'label': label})
            except Exception as exc:
                bad.append({'path': relative, 'error': str(exc)})
    (results / 'invalid_files.json').write_text(json.dumps(bad, indent=2))
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    if bad or duplicates:
        (results / 'duplicates.json').write_text(json.dumps(duplicates, indent=2))
        raise ValueError('Invalid or byte-identical files found; inspect results before splitting.')
    frame = pd.DataFrame(rows)
    train, holdout = train_test_split(frame, test_size=.30, stratify=frame.label, random_state=42)
    validation, test = train_test_split(holdout, test_size=.50, stratify=holdout.label, random_state=42)
    split = pd.concat([part.assign(split=name) for part, name in
                       [(train, 'train'), (validation, 'validation'), (test, 'test')]])
    assert split.path.is_unique and len(split) == len(frame)
    split.to_csv(results / 'splits.csv', index=False)
    counts = frame.groupby('class').size()
    counts.to_csv(results / 'class_counts.csv', header=['images'])
    split.groupby(['class', 'split']).size().unstack().to_csv(results / 'split_counts.csv')
    fig, ax = plt.subplots(figsize=(11, 5))
    counts.plot.bar(ax=ax)
    ax.set(ylabel='Images', title='EuroSAT RGB class distribution', xlabel='')
    fig.tight_layout(); fig.savefig(results / 'class_distribution.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(13, 6))
    for ax, name in zip(axes.flat, classes):
        path = frame[frame['class'] == name].iloc[0].path
        with Image.open(data / path) as img:
            ax.imshow(img)
        ax.set_title(name, fontsize=9); ax.axis('off')
    fig.tight_layout(); fig.savefig(results / 'samples.png', dpi=150); plt.close(fig)
    summary = {'images': len(frame), 'classes': classes, 'dimensions': dict(dimensions),
               'invalid_files': len(bad), 'duplicate_groups': len(duplicates),
               'largest_to_smallest_class_ratio': float(counts.max() / counts.min()),
               'splits': split.split.value_counts().to_dict(), 'seed': 42}
    (results / 'data_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
