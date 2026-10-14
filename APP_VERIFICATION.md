# SatVision application verification

## Direct repository checkpoint update

The model is now loaded only from `Path(__file__).resolve().parent / 'models/mobilenet.keras'` in `inference.py`, with the original size and SHA-256 checks. Release downloads, model URLs and download-cache fallback code have been removed. Git ignore checks confirm that `models/mobilenet.keras` can be added normally while other checkpoints, model temporary directories and caches remain ignored. No staging, commit or push was performed.

Modified project files: `.gitignore`, `inference.py`, `model_asset.json`, `test_app.py`, `README.md`, `DEPLOYMENT.md` and this verification record. The tests regenerated `app_checks/test_summary.json`; current direct-loading evidence is in `app_checks/direct_checkpoint_verification.json`.

The current full test suite passed all four groups with zero failures/errors in 11.94 seconds after imports, peaking at 507.07 MiB RSS. All six example predictions and their recorded probabilities match the previous application run exactly (maximum probability difference 0.0). Missing and tampered local checkpoints fail correctly. A fresh Streamlit server on port 8502 returned HTTP 200 from its health endpoint and was stopped after verification. Tests initially stalled in the sandbox's local socket/event-loop setup; granting network permission for loopback testing resolved the stall, and the complete suite was rerun successfully.

Source comparisons confirm that class ordering, RGB preprocessing, prediction, Grad-CAM, overlay and classifier-loading functions are unchanged; only checkpoint path resolution changed. A before/after checksum check covered all 93 existing files under `results/` and `models/`, including the trained checkpoints and caches. None changed. The checkpoint was neither retrained nor reserialized. All seven unchanged pinned app dependencies match the saved Linux CPython 3.13 wheel-resolution report. Community Cloud preparation is checked, but an actual Linux/Cloud runtime deployment has not been performed.

## Files and preservation

The earlier application upgrade added `app.py`, `inference.py`, `test_app.py`, `model_asset.json`, `.streamlit/config.toml`, six example JPEGs with manifest/license/attribution, deployment instructions, and generated application checks/captures. The former README is preserved verbatim in `EXPERIMENTS.md`; the former requirements are retained in `requirements-training.txt`.

A before/after SHA-256 check covered 86 original result, model, experiment-source and execution-record files. None changed. No model was fitted, retrained, reserialized or overwritten. Original test metrics remain unchanged. No new classification accuracy estimate was produced from the examples.

## Executed local checks

The final full suite ran four test groups successfully, with zero failures/errors, under Python 3.13.9, TensorFlow 2.21.0, Keras 3.15.1 and Streamlit 1.65.0:

- Repository-local checkpoint size/SHA-256 verification, load, saved class ordering, and rejection of missing or tampered local checkpoints.
- RGB, grayscale, palette and RGBA PNG inputs; unsupported GIF, corrupt data, and files exceeding the byte limit. Valid inputs reached inference with shape `(1, 64, 64, 3)` and raw pixel values.
- Six real EuroSAT examples: exact equality with the training loader's JPEG decode/resize; `(1, 10)` model output; finite probabilities summing to one; original saved top-class/probability agreement within `2e-5`; Grad-CAM finite nonzero gradients, nonconstant normalized 3×3 heatmaps, class-dependent maps and aligned overlays. Reconstructed backbone/head probabilities matched the checkpoint output within `2e-5`.
- Streamlit AppTest: initial render, classification, prediction display, Grad-CAM class selection, opacity-slider changes and switching to upload mode without application errors. The final chart uses a fixed 0–100% probability axis.

The earlier upgrade suite took 21.00 seconds after imports and peaked at 507.27 MiB Windows process RSS; current direct-loading measurements are listed above. These include test overhead and more than one model instance, not a Cloud memory measurement. Full machine-readable records are in `app_checks/test_summary.json` and `app_checks/sample_checks.json`. The supplied SeaLake example is misclassified as AnnualCrop, matching the original saved prediction. Examples were not filtered for success.

An actual local Streamlit server started on `127.0.0.1:8501`. `/_stcore/health` returned HTTP 200 with `ok`. Browser verification showed input preview, predicted class, top-three probabilities, all-class chart, selectable Grad-CAM overlay and explanatory text. Genuine narrow-browser screenshots are saved in `app_checks/app-screenshot.jpg` and `app_checks/gradcam-screenshot.jpg`; the six saved overlay PNGs are computed results, not application screenshots.

Source syntax compilation and `pip check` passed. Pip successfully resolved 59 application packages for Linux CPython 3.13 using compatible manylinux wheels; the detailed report is `app_checks/linux-dependency-resolution.json`. This was a dry run that downloaded wheels, not a Linux runtime execution. No Community Cloud deployment was attempted.

## Commands

From the original workspace in PowerShell:

```powershell
$python = 'C:\Users\tumma\Documents\Codex\sat-env\Scripts\python.exe'
& $python -m pip install --no-cache-dir streamlit==1.65.0
& $python -m pip check
& $python outputs\satellite-land-use\test_app.py
python -m py_compile outputs\satellite-land-use\app.py outputs\satellite-land-use\inference.py outputs\satellite-land-use\test_app.py
```

The compatibility dry-run command was:

```powershell
& $python -m pip install --dry-run --ignore-installed --only-binary=:all: --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --python-version 3.13 --implementation cp --abi cp313 -r outputs\satellite-land-use\requirements.txt --report outputs\satellite-land-use\app_checks\linux-dependency-resolution.json
```

Start the app from the original workspace:

```powershell
& $python -m streamlit run outputs\satellite-land-use\app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

The actual server launch used the project directory as working directory and `app.py` as the entrypoint, with `--server.headless true`. Both forms resolve model/sample paths from the source directory. Use the project directory as working directory to apply its `.streamlit/config.toml` theme.

## Fixes and remaining deployment work

During the earlier upgrade, the Windows memory-reporting call needed explicit ctypes argument types, and inspection of pip's Unicode metadata needed UTF-8 decoding. Those fixes remain in place. Download-related code has since been removed in favor of repository-local loading.

Before public deployment, manually include the original `models/mobilenet.keras` in the repository alongside the application, select Python 3.13 in Community Cloud, and perform a hosted runtime/memory check. No model URL, release asset or download step is required. No GitHub push or external deployment occurred.

Real aligned multi-date satellite change detection is future work, not an implemented feature. Grad-CAM is qualitative and coarse; probabilities are not calibrated confidence.
