"""Local regression checks; no fitting, dataset changes or experiment-metric writes."""
import hashlib
import io
import json
import platform
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import inference

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'app_checks'
REPORT.mkdir(exist_ok=True)
tempfile.tempdir = str(REPORT)

class AppChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.started = time.perf_counter()
        cls.model = inference.load_classifier()
        cls.names = inference.class_names()
        cls.examples = json.loads((ROOT / 'examples/manifest.json').read_text())
        cls.saved = pd.read_csv(ROOT / 'results/mobilenet/test/predictions.csv').set_index('path')
        cls.records = []

    def test_real_samples_and_gradients(self):
        for example in self.examples:
            with self.subTest(example=example['file']):
                content = (ROOT / 'examples' / example['file']).read_bytes()
                preview, batch = inference.prepare_image(content)
                reference = tf.image.resize(tf.io.decode_jpeg(content, channels=3), (64, 64))[None]
                np.testing.assert_array_equal(batch.numpy(), reference.numpy())
                probabilities = inference.predict(self.model, batch)
                self.assertEqual(probabilities.shape, (10,))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertAlmostEqual(float(probabilities.sum()), 1, places=5)
                predicted = int(probabilities.argmax())
                saved = self.saved.loc[example['source_path']]
                self.assertEqual(self.names[predicted], saved.predicted)
                self.assertAlmostEqual(float(probabilities.max()), saved.confidence, delta=2e-5)
                heatmap, details = inference.grad_cam(self.model, batch, predicted)
                np.testing.assert_allclose(details['reconstructed_probabilities'], probabilities, atol=2e-5)
                self.assertEqual(heatmap.shape, (3, 3))
                self.assertTrue(np.isfinite(heatmap).all())
                self.assertGreater(details['gradient_l1'], 0)
                self.assertGreater(float(heatmap.max()), 0)
                self.assertGreater(float(np.ptp(heatmap)), 0)
                self.assertTrue((heatmap >= 0).all() and (heatmap <= 1).all())
                overlay = inference.overlay_heatmap(preview, heatmap, .4)
                self.assertEqual(overlay.size, preview.size)
                overlay.save(REPORT / (Path(example['file']).stem + '_gradcam.png'))
                alternative, _ = inference.grad_cam(self.model, batch, (predicted + 1) % 10)
                self.assertFalse(np.allclose(heatmap, alternative))
                self.records.append({'file': example['file'], 'original_label': example['label'],
                                     'prediction': self.names[predicted], 'softmax_probability': float(probabilities.max()),
                                     'feature_layer': details['layer'], 'gradient_l1': details['gradient_l1'],
                                     'matches_saved_prediction': True})

    def test_formats_and_invalid_inputs(self):
        for mode in ['RGB', 'L', 'RGBA', 'P']:
            buffer = io.BytesIO()
            Image.new(mode, (80, 45)).save(buffer, format='PNG')
            preview, batch = inference.prepare_image(buffer.getvalue())
            self.assertEqual(preview.mode, 'RGB')
            self.assertEqual(tuple(batch.shape), (1, 64, 64, 3))
            inference.predict(self.model, batch)
        for content in [b'not an image', b'\xff\xd8corrupt', b'x' * (10*1024*1024 + 1)]:
            with self.assertRaises(ValueError):
                inference.prepare_image(content)
        buffer = io.BytesIO()
        Image.new('RGB', (5, 5)).save(buffer, format='GIF')
        with self.assertRaises(ValueError):
            inference.prepare_image(buffer.getvalue())

    def test_class_order_and_checksum(self):
        saved_names = json.loads((ROOT / 'results/data_summary.json').read_text())['classes']
        self.assertEqual(self.names, saved_names)
        self.assertEqual(len(self.names), 10)
        manifest = json.loads((ROOT / 'model_asset.json').read_text())
        with tempfile.TemporaryDirectory(dir=REPORT) as temporary:
            bad = Path(temporary) / 'wrong.keras'
            bad.write_bytes(b'x' * manifest['bytes'])
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                inference.verify_checkpoint(bad, manifest)
            self.assertEqual(inference.checkpoint_path(), ROOT / 'models/mobilenet.keras')
            # Missing repository assets fail clearly instead of downloading a replacement.
            root = Path(temporary)
            (root / 'model_asset.json').write_text(json.dumps(manifest))
            with patch.object(inference, 'ROOT', root):
                with self.assertRaisesRegex(FileNotFoundError, 'Include the original trained'):
                    inference.checkpoint_path()
            (root / 'models').mkdir()
            content = (ROOT / 'models/mobilenet.keras').read_bytes()
            local = root / 'models/mobilenet.keras'
            local.write_bytes(content)
            with patch.object(inference, 'ROOT', root):
                self.assertEqual(inference.checkpoint_path(), local)
                local.write_bytes(b'x' * len(content))
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    inference.checkpoint_path()

    def test_streamlit_interaction(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any('Prediction:' in heading.value for heading in app.subheader))
        app.slider[0].set_value(.7).run()
        app.selectbox[1].set_value(1).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        app.radio[0].set_value('Upload image').run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)

    @classmethod
    def tearDownClass(cls):
        if cls.records:
            (REPORT / 'sample_checks.json').write_text(json.dumps(cls.records, indent=2))

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AppChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'python': platform.python_version(), 'tensorflow': tf.__version__,
              'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'successful': result.wasSuccessful(), 'new_model_accuracy_measured': False,
              'seconds': time.perf_counter() - AppChecks.started}
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                                   'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                                   'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
        memory = Memory(); memory.cb = ctypes.sizeof(memory)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
            report['test_process_peak_rss_mib'] = memory.PeakWorkingSetSize / 2**20
    (REPORT / 'test_summary.json').write_text(json.dumps(report, indent=2))
    raise SystemExit(not result.wasSuccessful())
