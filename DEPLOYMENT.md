# Streamlit Community Cloud preparation

The app is prepared locally; no public repository, model release or Cloud deployment was created. Public publication requires the user's approval.

## Model asset

The original `models/mobilenet.keras` is 10,630,358 bytes (about 10.14 MiB). It remains excluded from Git. The app verifies this exact SHA-256 before loading:

```text
31bc6c59ad4a5bed12b0b73b552bce61ff36ee08efd4f80c0e9a7c453333c7ed
```

After approval, upload the existing checkpoint as `mobilenet-v1.keras` to a public GitHub Release tagged `eurosat-mobilenet-v1`. Do not retrain, reserialize or replace it. Set `model_asset.json`'s `url` to its actual HTTPS asset URL, for example `https://github.com/OWNER/REPOSITORY/releases/download/eurosat-mobilenet-v1/mobilenet-v1.keras`. This example is not an existing release.

The filename, byte count and checksum are already recorded in the manifest; its URL is deliberately `null`. Alternatively, set the `SATVISION_MODEL_URL` environment variable. The loader first checks the original local checkpoint, then a verified `.model-cache` file, then downloads the configured asset. Downloads use a temporary file, enforce the expected byte count, and require the expected checksum before caching. A mismatched existing file fails closed. Delete an invalid cached download before retrying; do not change the expected checksum to accommodate an unknown file.

The release URL is public and does not need a token or secret. Keep release assets versioned and immutable by convention. The committed checksum remains the integrity anchor even if a release URL's contents are replaced. The download/cache path was tested with a mocked HTTPS response; an actual public release download remains untested because publication has not been approved.

## Repository and Cloud configuration

Use this project directory as the repository root. Include application source, `requirements.txt`, `model_asset.json`, `.streamlit/config.toml`, `results/data_summary.json` for the class mapping, the examples and their attribution, and the existing experiment evidence/documentation. Exclude the full dataset, environments, model binaries, caches and `.streamlit/secrets.toml` as specified by `.gitignore`.

After approval and repository publication:

1. Confirm the release asset downloads and matches the checksum above.
2. In Community Cloud, select the repository and `app.py` entrypoint.
3. In Advanced settings, explicitly select Python **3.13**. Do not rely on the platform default or a `runtime.txt` file.
4. Deploy, inspect installation/runtime logs, classify an example and test Grad-CAM. Check actual memory under the Cloud allocation before calling deployment successful.

Python selection is documented in [Streamlit's deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). Match the tested local version rather than assuming defaults. Community Cloud runs Linux; [Streamlit's platform limitations](https://docs.streamlit.io/deploy/streamlit-community-cloud/status) describe path and repository-root behavior. Paths in the app resolve relative to `__file__` and use portable `Path` objects.

## Compatibility and resource limits

Local runtime tests used Python 3.13.9, TensorFlow 2.21.0, Keras 3.15.1 and Streamlit 1.65.0. Direct app dependencies are pinned. Linux CPython 3.13 wheel resolution is tested separately with pip; the resulting report is in `app_checks/linux-dependency-resolution.json` when successful. Resolution does not prove that Linux native libraries load or that a hosted app has started. No Linux/Cloud runtime result is claimed.

The Windows test process peaked at approximately 507 MiB RSS. Cloud memory can differ and concurrent sessions can add overhead. The app caches one classifier per process, performs batch-one inference, uses two TensorFlow computation threads, and limits cached analysis/Grad-CAM entries. Uploads are limited to 10 MB and four million pixels. The model file's small disk size is not its total runtime memory requirement.

For training-script reproduction only, install `requirements-training.txt`; those scripts are not run by the app and were not rerun during this upgrade.
