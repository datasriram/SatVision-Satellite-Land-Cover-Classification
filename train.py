"""Train on train/validation only; evaluate the held-out test set explicitly."""
import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Keras writes a temporary HDF5 file when saving. Keep it in a writable local folder.
temporary_directory = ROOT / 'models' / '.tmp'
temporary_directory.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(temporary_directory)
os.environ.setdefault('KERAS_HOME', str(ROOT / 'models' / '.keras'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'models' / '.mpl'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)
keras.utils.set_random_seed(42)

def dataset(frame, data, training=False, batch_size=64):
    paths = [str(data / path) for path in frame.path]
    ds = tf.data.Dataset.from_tensor_slices((paths, frame.label.to_numpy(dtype='int32')))
    if training:
        ds = ds.shuffle(len(frame), seed=42, reshuffle_each_iteration=True)
    def load(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, (64, 64))
        return image, label
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    return ds.map(load, num_parallel_calls=4).batch(batch_size).prefetch(1).with_options(options)

def build_model(name):
    inputs = keras.Input((64, 64, 3))
    x = keras.layers.RandomFlip('horizontal_and_vertical', seed=42)(inputs)
    if name == 'cnn_rotation':
        # A controlled orientation-invariance experiment, enabled only in training.
        x = keras.layers.RandomRotation(.25, fill_mode='reflect', seed=43)(x)
    if name == 'mobilenet':
        x = keras.layers.Resizing(96, 96)(x)
        x = keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
        # ImageNet features reduce the number of weights learned from this dataset.
        backbone = keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                                  input_shape=(96, 96, 3))
        # Freeze initially to preserve pretrained features; batch-normalization uses inference statistics.
        backbone.trainable = False
        x = backbone(x, training=False)
    else:
        x = keras.layers.Rescaling(1 / 255)(x)
        for filters in (16, 32, 64):
            x = keras.layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
            x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    # Modest dropout regularizes the classification head without deepening the CNN.
    x = keras.layers.Dropout(.2, seed=44)(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_error_grid(rows, data, path):
    if rows.empty:
        return
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.axis('off')
    for ax, (_, row) in zip(axes.flat, rows.head(12).iterrows()):
        with Image.open(data / row['path']) as image:
            ax.imshow(image)
        ax.set_title(f"True: {row['class']}\nPred: {row['predicted']} ({row['confidence']:.2f})", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)

def evaluate(model, frame, data, folder, classes):
    folder.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict(dataset(frame, data), verbose=0)
    predicted = probabilities.argmax(axis=1)
    truth = frame.label.to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=range(10), average='macro', zero_division=0)
    metrics = {'accuracy': float(accuracy_score(truth, predicted)),
               'macro_precision': float(precision), 'macro_recall': float(recall), 'macro_f1': float(f1),
               'images': len(frame)}
    (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    report = classification_report(truth, predicted, labels=range(10), target_names=classes,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(folder / 'classification_report.csv')
    predictions = frame.copy()
    predictions['predicted'] = [classes[i] for i in predicted]
    predictions['confidence'] = probabilities.max(axis=1)
    predictions.to_csv(folder / 'predictions.csv', index=False)
    matrix = confusion_matrix(truth, predicted, labels=range(10))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(folder / 'confusion_matrix.csv')
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(10), yticks=range(10), xticklabels=classes, yticklabels=classes,
           xlabel='Predicted class', ylabel='True class', title=folder.name)
    plt.setp(ax.get_xticklabels(), rotation=65, ha='right')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, matrix[i, j], ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, j] > matrix.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(folder / 'confusion_matrix.png', dpi=140); plt.close(fig)
    pairs = [{'true': classes[i], 'predicted': classes[j], 'count': int(matrix[i,j])}
             for i in range(10) for j in range(10) if i != j and matrix[i,j]]
    pairs.sort(key=lambda row: row['count'], reverse=True)
    (folder / 'confused_pairs.json').write_text(json.dumps(pairs[:10], indent=2))
    wrong = predictions[truth != predicted].sort_values('confidence', ascending=False).head(12)
    save_error_grid(wrong, data, folder / 'misclassified.png')
    pair_errors = predictions[((predictions['class'] == 'Highway') & (predictions.predicted == 'River')) |
                              ((predictions['class'] == 'River') & (predictions.predicted == 'Highway'))]
    save_error_grid(pair_errors, data, folder / 'highway_river_errors.png')
    print(folder, json.dumps(metrics), flush=True)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', choices=['cnn', 'mobilenet', 'cnn_rotation'], default='cnn')
    parser.add_argument('--stage', choices=['smoke', 'train', 'validation', 'test'], default='train')
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    data = args.data.resolve()
    split = pd.read_csv(ROOT / 'results' / 'splits.csv')
    assert split.path.is_unique, 'Overlapping split paths'
    classes = json.loads((ROOT / 'results' / 'data_summary.json').read_text())['classes']
    for path, name, label in split[['path', 'class', 'label']].itertuples(index=False, name=None):
        assert classes[label] == name and (data / path).is_file()
    models = ROOT / 'models'; models.mkdir(exist_ok=True)
    folder = ROOT / 'results' / args.model
    folder.mkdir(exist_ok=True)
    environment = {'python': platform.python_version(), 'tensorflow': tf.__version__,
                   'keras': keras.__version__, 'logical_cpus': os.cpu_count(),
                   'devices': [str(device) for device in tf.config.list_physical_devices()],
                   'intra_op_threads': 8, 'inter_op_threads': 2}
    (ROOT / 'results' / 'environment.json').write_text(json.dumps(environment, indent=2))
    print(environment, flush=True)
    model_path = models / f'{args.model}.keras'
    if args.stage in ['validation', 'test']:
        model = keras.models.load_model(model_path)
        evaluate(model, split[split.split == args.stage], data, folder / args.stage, classes)
        return
    model = build_model(args.model)
    model.summary()
    if args.stage == 'smoke':
        tiny = split[split.split == 'train'].groupby('label').head(2)
        print('Smoke: loading 20 images', flush=True)
        batch = next(iter(dataset(tiny, data)))
        assert tuple(batch[0].shape) == (20, 64, 64, 3)
        assert tuple(model(batch[0], training=False).shape) == (20, 10)
        print('Smoke: forward pass passed; training one batch', flush=True)
        metrics = model.train_on_batch(*batch, return_dict=True)
        assert all(np.isfinite(value) for value in metrics.values())
        path = models / f'{args.model}_smoke.keras'
        print('Smoke: finite batch metrics; saving and reloading', flush=True)
        model.save(path)
        loaded = keras.models.load_model(path)
        np.testing.assert_allclose(model(batch[0], training=False), loaded(batch[0], training=False), atol=1e-6)
        path.unlink()
        (folder / 'smoke.json').write_text(json.dumps({'status': 'passed', 'batch_images': 20,
                                                      'metrics': {k: float(v) for k,v in metrics.items()}}, indent=2))
        print('SMOKE PASSED (20 images; not an experiment)', flush=True)
        return
    if model_path.exists():
        raise FileExistsError(f'{model_path} exists. Preserve previous results before rerunning.')
    start = time.perf_counter()
    history = model.fit(dataset(split[split.split == 'train'], data, training=True),
                        validation_data=dataset(split[split.split == 'validation'], data),
                        epochs=args.epochs, verbose=2, shuffle=False,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                                                 restore_best_weights=True),
                                   keras.callbacks.ModelCheckpoint(str(model_path), monitor='val_loss',
                                                                   save_best_only=True),
                                   keras.callbacks.CSVLogger(str(folder / 'history.csv'))])
    elapsed = time.perf_counter() - start
    # Reload the checkpoint so reported metrics always describe the saved model.
    model = keras.models.load_model(model_path)
    values = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        ax.plot(range(1, len(values)+1), values[metric], label='train')
        ax.plot(range(1, len(values)+1), values['val_'+metric], label='validation')
        ax.set(xlabel='Epoch', ylabel=metric); ax.legend()
    fig.tight_layout(); fig.savefig(folder / 'learning_curves.png', dpi=150); plt.close(fig)
    metrics = evaluate(model, split[split.split == 'validation'], data, folder / 'validation', classes)
    record = {'status': 'completed', 'model': args.model, 'seconds': elapsed,
              'epochs_completed': len(values), 'epoch_limit': args.epochs,
              'best_epoch_by_val_loss': int(values.val_loss.argmin()+1),
              'selection_criterion': 'minimum validation loss', 'seed': 42,
              'validation': metrics}
    (folder / 'run.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)

if __name__ == '__main__':
    main()
