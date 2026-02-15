# Installation Details

Reference for reinstalling dependencies if the environment needs to be rebuilt.

## Python Virtual Environment

```bash
python3 -m venv /workspace/venv
/workspace/venv/bin/pip install --upgrade pip wheel setuptools
```

## PyTorch (CUDA 12.8)

```bash
/workspace/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check available CUDA versions at: https://download.pytorch.org/whl/

## Python Packages

```bash
/workspace/venv/bin/pip install \
    fastai \
    monai \
    scipy \
    scikit-image \
    tifffile \
    pybind11 \
    connected-components-3d \
    dijkstra3d \
    ipywidgets \
    fastprogress==1.0.5 \
    kaggle
```

**Note:** `fastprogress` must be pinned to 1.0.5 — version 1.1.3 has a bug where
`NBMasterBar` is missing the `out` attribute, which crashes fastai's `ProgressCallback`.

## Topometrics Library (Competition Metric)

Source: Kaggle dataset `dheyeong/vesuvius-metric-resources` (~133 MB)

```bash
# Download source
export KAGGLE_API_TOKEN=<your-token>
kaggle datasets download dheyeong/vesuvius-metric-resources \
    -p /workspace/vesuvius-kaggle-competition/libs/ --unzip

# Move to expected location
mv libs/vesuvius-metric-resources/topological-metrics-kaggle libs/topological-metrics-kaggle
```

### Building the C++ Betti-Matching Extension

Requires: `cmake`, `python3-dev`, `pybind11` (installed via pip above)

```bash
cd libs/topological-metrics-kaggle/external/Betti-Matching-3D
rm -rf build && mkdir -p build && cd build

PYBIN=/workspace/venv/bin/python
PY_INC=$($PYBIN -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")
PY_LIB=$($PYBIN -c "import sysconfig, os; libdir=sysconfig.get_config_var('LIBDIR'); ldver=sysconfig.get_config_var('LDVERSION'); print(next((p for p in [os.path.join(libdir, f'libpython{ldver}.so'), os.path.join(libdir, f'libpython{ldver}.a')] if os.path.exists(p)), ''))")
PYBIND11_DIR=$($PYBIN -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake -S .. -B . \
    -DPython_EXECUTABLE="$PYBIN" \
    -DPython_INCLUDE_DIR="$PY_INC" \
    -DPython_LIBRARY="$PY_LIB" \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DPYBIND11_FINDPYTHON=ON

cmake --build . --parallel
```

Then install the Python package:

```bash
pip install -e /workspace/vesuvius-kaggle-competition/libs/topological-metrics-kaggle/
```

**Note:** The C++ `.so` file is machine-specific and needs to be rebuilt when
moving to a different machine (different CPU, OS, or Python version). The
`start.sh` script handles this automatically.

**Note:** The build script bundled with topometrics (`scripts/build_betti.sh`)
may fail because: (1) it can't find `pybind11` via cmake — you need to pass
`-Dpybind11_DIR` explicitly, and (2) the script permissions may need `chmod +x`.

## SuPreM Pretrained Weights

```bash
mkdir -p pretrained_weights
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_segresnet_2100.pth \
    -O pretrained_weights/supervised_suprem_segresnet_2100.pth
```

Source: HuggingFace `MrGiovanni/SuPreM` (~56.5 MB)
Paper: https://arxiv.org/abs/2310.16754 (ICLR 2024 Oral)

## Competition Data

```bash
kaggle competitions download -c vesuvius-challenge-surface-detection -p data/
cd data/ && unzip -q vesuvius-challenge-surface-detection.zip
# Move extracted folders up to project root level
mv train_images train_labels test_images train.csv test.csv ../
```

~25 GB download. Contains 786 training volumes (320x320x320 uint8 TIFFs).

## Kaggle Traced Models (from our uploads)

```bash
kaggle datasets download mgoldfield/vesuvius-unet3d-weights \
    -p kaggle/kaggle_weights_download/ --unzip
```

Contains: v9 traced model (best), v10 traced, v11 fold0/1/2 traced, plus
cc3d and dijkstra3d wheel files for offline Kaggle install.
