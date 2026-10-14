"""Inference and Grad-CAM for the original, unchanged MobileNetV2 checkpoint."""
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

ROOT = Path(__file__).resolve().parent

def class_names():
    # This is the saved label mapping, not a new directory listing or guessed order.
    names = json.loads((ROOT / 'results/data_summary.json').read_text())['classes']
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError('Expected ten unique saved class names.')
    return names

def checkpoint_path():
    manifest = json.loads((ROOT / 'model_asset.json').read_text())
    path = ROOT / 'models/mobilenet.keras'
    if not path.is_file():
        raise FileNotFoundError('Missing models/mobilenet.keras. Include the original trained '
                                'checkpoint in the repository before deploying. See DEPLOYMENT.md.')
    verify_checkpoint(path, manifest)
    return path

def verify_checkpoint(path, manifest):
    if path.stat().st_size != manifest['bytes']:
        raise ValueError('Model checkpoint size does not match model_asset.json.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('Model checkpoint SHA-256 mismatch; refusing to load it.')

def load_classifier():
    model = tf.keras.models.load_model(checkpoint_path(), compile=False, safe_mode=True)
    if model.input_shape != (None, 64, 64, 3) or model.output_shape != (None, 10):
        raise ValueError('Unexpected checkpoint input/output shape.')
    return model

def prepare_image(content):
    if len(content) > 10 * 1024 * 1024:
        raise ValueError('Choose an image smaller than 10 MB.')
    try:
        with Image.open(io.BytesIO(content)) as image:
            file_format = image.format
            if file_format not in ('JPEG', 'PNG'):
                raise ValueError('Only JPEG and PNG images are supported.')
            if image.width * image.height > 4_000_000:
                raise ValueError('Choose an image with at most four million pixels.')
            image.verify()
        # TensorFlow JPEG decoding and bilinear resizing match train.py exactly.
        decoder = tf.io.decode_jpeg if file_format == 'JPEG' else tf.io.decode_png
        pixels = decoder(content, channels=3)
        batch = tf.image.resize(pixels, (64, 64))[None, ...]
        preview = Image.fromarray(pixels.numpy())
        return preview, batch
    except (UnidentifiedImageError, OSError, tf.errors.InvalidArgumentError, Image.DecompressionBombError) as exc:
        raise ValueError('The image is corrupt or cannot be decoded.') from exc

def predict(model, batch):
    probabilities = model(batch, training=False).numpy()
    if probabilities.shape != (1, 10) or not np.isfinite(probabilities).all():
        raise ValueError('Model returned invalid probabilities.')
    if not np.isclose(probabilities.sum(), 1, atol=1e-5):
        raise ValueError('Model probabilities do not sum to one.')
    return probabilities[0]

def grad_cam(model, batch, class_index):
    if not 0 <= class_index < 10:
        raise ValueError('Invalid class index.')
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    convolution = backbone.get_layer('Conv_1')
    feature_model = tf.keras.Model(backbone.inputs, [convolution.output, backbone.output])
    position = model.layers.index(backbone)
    x = batch
    for layer in model.layers[1:position]:
        x = layer(x, training=False)
    with tf.GradientTape() as tape:
        # Watching the input records the frozen backbone's operations as well.
        tape.watch(x)
        features, hidden = feature_model(x, training=False)
        for layer in model.layers[position + 1:-1]:
            hidden = layer(hidden, training=False)
        output_layer = model.layers[-1]
        # Use pre-softmax scores without mutating the saved model's activation.
        logits = tf.matmul(hidden, output_layer.kernel) + output_layer.bias
        score = logits[:, class_index]
    gradients = tape.gradient(score, features)
    if gradients is None or not bool(tf.reduce_all(tf.math.is_finite(gradients))):
        raise ValueError('Grad-CAM gradients are missing or non-finite.')
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))[0]
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    reconstructed = tf.nn.softmax(logits).numpy()[0]
    return heatmap.numpy(), {'layer': f'{backbone.name}/Conv_1',
                             'gradient_l1': float(tf.reduce_sum(tf.abs(gradients))),
                             'reconstructed_probabilities': reconstructed}

def overlay_heatmap(preview, heatmap, alpha):
    # The low-resolution feature map is interpolated over the same displayed pixels.
    resized = tf.image.resize(heatmap[..., None], (preview.height, preview.width)).numpy()[..., 0]
    colors = np.stack([resized, .25 * (1 - np.abs(2 * resized - 1)), 1 - resized], axis=-1)
    blended = (1 - alpha) * np.asarray(preview, dtype=np.float32) + alpha * colors * 255
    return Image.fromarray(np.clip(blended, 0, 255).astype('uint8'))
