# Streamlit Community Cloud deployment

The app loads the existing trained checkpoint directly from **`models/mobilenet.keras` in the repository**. No GitHub Release, download URL, token or model-download step is required. No commit, push or external deployment has been performed automatically.

## Checkpoint and integrity

The original file is 10,630,358 bytes (about 10.14 MiB). `.gitignore` allows only this checkpoint under `models/`; other model files, temporary directories and caches remain ignored.

`inference.py` resolves the file using `Path(__file__).resolve().parent / 'models/mobilenet.keras'`. Loading is independent of the current working directory. Before `tf.keras.models.load_model(..., compile=False, safe_mode=True)`, its byte count and SHA-256 are checked against `model_asset.json`:

```text
31bc6c59ad4a5bed12b0b73b552bce61ff36ee08efd4f80c0e9a7c453333c7ed
```

Use the original binary without retraining, editing or reserializing it. Missing files and checksum mismatches produce explicit errors. There is no network fallback. Streamlit caches the loaded classifier once per process; preprocessing and Grad-CAM are unchanged.

## Files to include

Use this project directory as the GitHub repository root. Include `app.py`, `inference.py`, `requirements.txt`, `model_asset.json`, **`models/mobilenet.keras`**, `.streamlit/config.toml`, `results/data_summary.json` for class ordering, and the supplied examples with their manifest and attribution. Keep the existing experiment evidence and documentation as appropriate for the portfolio. Do not include the entire dataset, other checkpoints, virtual environments, caches or `.streamlit/secrets.toml`.

Before publishing, verify that the checkpoint appears in Git's file list. `.gitignore` now permits it, but changing exclusions does not automatically stage, commit or upload a file. Ordinary Git storage is sufficient for this approximately 10 MB file; Git LFS is not required for this setup.

## Community Cloud setup

When you choose to publish:

1. Manually commit and push the application files and original `models/mobilenet.keras` checkpoint to your GitHub repository.
2. In Community Cloud, select the repository, branch and `app.py` entrypoint.
3. In Advanced settings, explicitly select **Python 3.13** to match the tested environment. Do not depend on the platform default or a `runtime.txt` file.
4. Deploy and inspect installation/runtime logs. Classify an example, test Grad-CAM and check memory usage before calling the hosted deployment successful.

The [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) documents repository/entrypoint selection and Python version configuration. [Dependency instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) describe installation from `requirements.txt`.

## Compatibility and resources

Application dependencies remain pinned and unchanged: Python 3.13 was tested locally with TensorFlow 2.21.0, Keras 3.15.1 and Streamlit 1.65.0. The saved Linux CPython 3.13 dependency-resolution report confirms compatible wheels for all direct dependencies. This is dependency compatibility evidence, not a Linux runtime or Cloud deployment result.

The application tests include checkpoint loading, preprocessing/prediction regression checks, Grad-CAM, input validation and Streamlit interaction. Test evidence is in `app_checks/` and the current summary in `APP_VERIFICATION.md`.

The original Windows test process used about 507 MiB peak RSS; Cloud memory can differ. Batch-one inference, two TensorFlow computation threads, bounded result caches and upload limits remain unchanged. Model file size alone does not measure runtime memory. No model retraining is needed for deployment.
