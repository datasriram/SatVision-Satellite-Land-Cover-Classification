# Execution record

The project was executed on the local EuroSAT copy at `C:\Users\tumma\Downloads\Satellite_Land_Use_Project\EuroSAT_RGB`. The dataset was not downloaded or copied into the repository. Another extracted copy was found under `Downloads\archive (5)` and was not used.

## Successful commands

From the original workspace in PowerShell:

```powershell
python --version
python -m pip list --format=json
python outputs\satellite-land-use\prepare.py --data C:\Users\tumma\Downloads\Satellite_Land_Use_Project\EuroSAT_RGB
$python = 'C:\Users\tumma\Documents\Codex\sat-env\Scripts\python.exe'
$data = 'C:\Users\tumma\Downloads\Satellite_Land_Use_Project\EuroSAT_RGB'
python -m pip --python $python install --no-cache-dir -r outputs\satellite-land-use\requirements.txt
& $python -m pip check
& $python -m pip freeze
& $python outputs\satellite-land-use\train.py --data $data --model cnn --stage smoke
& $python outputs\satellite-land-use\train.py --data $data --model mobilenet --stage smoke
& $python outputs\satellite-land-use\train.py --data $data --model cnn_rotation --stage smoke
& $python outputs\satellite-land-use\train.py --data $data --model cnn --epochs 15
& $python outputs\satellite-land-use\train.py --data $data --model mobilenet --epochs 15
& $python outputs\satellite-land-use\train.py --data $data --model cnn_rotation --epochs 15
& $python outputs\satellite-land-use\train.py --data $data --model cnn --stage test
& $python outputs\satellite-land-use\train.py --data $data --model mobilenet --stage test
& $python outputs\satellite-land-use\train.py --data $data --model cnn_rotation --stage test
& $python outputs\satellite-land-use\report.py
python -m py_compile outputs\satellite-land-use\prepare.py outputs\satellite-land-use\train.py outputs\satellite-land-use\report.py
```

Training and test output was captured using `Tee-Object` in each model's results directory. The successful preparation run used a writable Matplotlib cache through `$env:MPLCONFIGDIR`; the scripts now configure local cache folders themselves.

Independent checks verified that split paths are unique and nonoverlapping, that the initial trainable weights of both CNN variants match exactly after resetting seed 42, and that all three scripts compile. `results/verification.json` records verification status. Each smoke test used 20 training images, executed a forward pass and one training batch, and checked prediction equality after saving/reloading. These are tests, not completed accuracy experiments.

Final verification independently recomputed all six validation/test metric rows from saved prediction CSVs, checked manifest/prediction order and labels, confirmed confusion-matrix totals of 4,050 per evaluation, checked checkpoint-selection epochs against histories, and verified all 22 generated PNG files. These checks passed. Trained-model SHA-256 hashes are recorded in `results/verification.json`. The temporary verification script was executed with `& $python work\verify_results.py` from the original workspace.

All three training runs, all three validation evaluations, all three held-out test evaluations, and `report.py` completed. There are no unfinished required experiments and no next command needed to finish this pipeline. Full measured results and the mixed outcome of the rotation experiment are in `README.md` and `results/comparison.csv`.

## Environment and errors resolved

- The default Python is 3.14.5 and lacked TensorFlow. Python 3.13.9 was available and used for the training environment. Exact installed versions are saved in `results/environment-packages.txt`; `pip check` reported no broken requirements.
- Initial virtual-environment bootstrapping failed because the sandbox temporary directory denied writes. Setting `TEMP` and `TMP` to workspace `work\tmp` allowed bootstrapping.
- The first dependency download failed with PyPI DNS resolution errors. Network access was requested and granted; installation and the MobileNetV2 ImageNet weight download then succeeded.
- TensorFlow installation under the long workspace `.venv` path failed on a nested header filename. A shorter environment at `C:\Users\tumma\Documents\Codex\sat-env` installed successfully. No system registry setting was changed.
- The first image audit was interrupted during slow startup. A writable local Matplotlib cache resolved that delay, and the replacement audit completed all 27,000 images.
- Initial CNN and MobileNetV2 smoke attempts were interrupted while Keras saving stalled. A diagnostic traceback showed repeated temporary-file creation in `tempfile.NamedTemporaryFile`. Setting a writable temporary directory in `train.py` resolved saving; both tests were rerun successfully. No interrupted smoke output is treated as an experiment result.
- CIM hardware inspection was denied and `nvidia-smi` was unavailable. Windows memory APIs reported 15.68 GiB RAM, with about 3.42 GiB available at the initial check. Python reported 16 logical CPUs. TensorFlow exposed one CPU device and no GPU device. Training completed on CPU with eight computation threads, four image-loading workers, and one prefetched batch.
- Keras warned that its default `shuffle=True` argument is ignored for a `tf.data.Dataset`. The dataset already shuffles explicitly. The script now passes `shuffle=False` to `fit` to remove that redundant warning; training data remains shuffled by `tf.data`.

## Files

The source consists of `prepare.py`, `train.py`, `report.py`, `README.md`, `requirements.txt`, and `.gitignore`, plus this execution record. `results/` holds generated evidence, including split manifests, histories, metrics, predictions, classification reports, curves, confusion matrices, and actual error images. `models/` holds three trained Keras checkpoints. The dataset, model binaries, environments, and temporary caches are excluded from Git. No commit or push was performed.

The deliberate future change-detection extension is not implemented. It requires aligned multi-date imagery rather than unrelated EuroSAT image pairs.
