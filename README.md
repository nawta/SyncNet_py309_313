# syncnet-python

[![PyPI version](https://badge.fury.io/py/syncnet-python.svg)](https://badge.fury.io/py/syncnet-python)
[![Python](https://img.shields.io/pypi/pyversions/syncnet-python.svg)](https://pypi.org/project/syncnet-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pip-installable SyncNet for Python 3.9 to 3.13 and PyTorch 2. It computes the confidence and minimum distance scores that talking-head and lip-sync papers report as LSE-C and LSE-D.

SyncNet (Chung and Zisserman, 2016) is a network that measures how well mouth movements in a video match the speech audio. This package wraps the original SyncNet model and its S3FD face detector in one Python class and one command. Given a video, it finds and tracks faces, crops each face track, and returns three numbers per track: the audio-video offset in frames, the SyncNet confidence (LSE-C, higher means better sync), and the minimum audio-video feature distance (LSE-D, lower means better sync).

## Quickstart

### 1. Install

```bash
pip install syncnet-python "scenedetect<0.7"
```

You also need `ffmpeg` on your `PATH` (`brew install ffmpeg` or `sudo apt-get install ffmpeg`).

Install the `scenedetect<0.7` pin together with the package. Version 0.2.2 on PyPI declares `scenedetect>=0.6.0`, so pip installs scenedetect 0.7.x, which removed the `scenedetect.video_manager` module the pipeline imports. Without the pin, `from syncnet_python import SyncNetPipeline` gives `None`.

This command installs two OpenCV packages, `opencv-python` (required by `scenedetect[opencv]`) and `opencv-contrib-python` (required by this package). Both provide the `cv2` module. In a fresh environment the installer picks the same version for both (5.0.0.93 on 2026-10-06) and `cv2` works. If you later uninstall either package, `cv2` breaks; fix it with `pip install --force-reinstall opencv-contrib-python` at the version you have.

### 2. Download the weights

The package does not include the model weights. Both files come from the Oxford VGG SyncNet page, the same URLs that the original repository's `download_model.sh` uses:

```bash
mkdir -p weights
wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -O weights/syncnet_v2.model
wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -O weights/sfd_face.pth
```

The same two files are also in the [`weights/`](weights/) folder of this repository. They are byte-identical to the Oxford files (SHA-256 `961e8696…5442` for `syncnet_v2.model` and `d54a87c2…c491` for `sfd_face.pth`).

### 3. Python API

```python
from syncnet_python import SyncNetPipeline

pipe = SyncNetPipeline(
    {"s3fd_weights": "weights/sfd_face.pth", "syncnet_weights": "weights/syncnet_v2.model"},
    device="cpu",  # or "cuda"
)
offsets, confs, dists, best_conf, min_dist, s3fd_json, has_face = pipe.inference(
    video_path="clip.mp4",
    audio_path=None,  # None uses the video's own audio track; or pass a .wav path
)
print("AV offset (frames):", offsets[0])
print("LSE-C (confidence):", round(best_conf, 3))
print("LSE-D (min distance):", round(min_dist, 3))
```

`offsets`, `confs` and `dists` are lists with one entry per face track. `best_conf` is the largest confidence over all tracks and `min_dist` is the smallest distance over all tracks. For a video with one face these equal `confs[0]` and `dists[0]`. With several faces the two values can come from different tracks, so use the per-track lists if you need scores for one speaker.

`syncnet_python.calculate_lse_metrics(pipe, video_path)` returns `(lse_c, lse_d, quality_label)` from the same values.

### 4. Command line

```bash
syncnet-python clip.mp4 --device cpu -o results.json
```

The command reads the weights from `weights/` in the current directory by default; change this with `--s3fd-weights` and `--syncnet-weights`. It prints the offset and confidence for each video and writes offset, confidence and minimum distance to the JSON file. The default device is `cuda`, so pass `--device cpu` on a machine without an NVIDIA GPU.

### Tested setup

The examples above were run on 2026-10-06 on an Apple Silicon Mac (CPU) with `syncnet-python==0.2.2` from PyPI, using [`example/video.avi`](example/video.avi) converted to MP4. The OpenCV column gives the version of `opencv-contrib-python`; `opencv-python` was at the same version in each run.

| Python | PyTorch | NumPy | OpenCV | scenedetect | Offset | LSE-C | LSE-D |
|---|---|---|---|---|---|---|---|
| 3.13 (arm64) | 2.14.1 | 2.5.3 | 5.0.0 | 0.6.7.1 | 1 | 4.529 | 9.237 |
| 3.9 (x86_64 under Rosetta 2) | 2.2.2 | 1.26.4 | 4.11.0 | 0.6.7.1 | 1 | 4.524 | 9.291 |

The Python 3.9 interpreter was an x86_64 build running under Rosetta 2. PyTorch 2.2.2 is the newest release with macOS x86_64 wheels, and it does not work with NumPy 2, so this run needed `numpy<2`, `opencv-python<4.12` and `opencv-contrib-python<4.12` installed by hand. The full run on the 5.3-second clip took about 10 seconds on CPU.

The `main` branch of this repository is at version 0.2.1, one release behind PyPI. In 0.2.1, `audio_path=None` fails, so pass an audio file if you install from source.

## Comparison with joonson/syncnet_python

The original repository, [joonson/syncnet_python](https://github.com/joonson/syncnet_python), was updated by its author on 2026-04-17 (PR #78). This table compares that version with this package.

| | joonson/syncnet_python (2026-04) | syncnet-python 0.2.2 |
|---|---|---|
| Install | clone, then `conda env create -f environment.yml`; no `setup.py` or `pyproject.toml` | `pip install syncnet-python` |
| Python | 3.10 (pinned in `environment.yml`) | 3.9 to 3.13 (3.9 and 3.13 tested above) |
| PyTorch | 2.5.1 (pinned) | `torch>=2.0.0` |
| scenedetect | 0.6.7.1 (pinned) | 0.6.x (`>=0.6.0` declared; 0.7 breaks the import, see above) |
| Python API | `SyncNetInstance.evaluate()` and `extract_feature()` score a pre-cropped face clip; face detection, tracking and cropping run only through `run_pipeline.py` | `SyncNetPipeline(...).inference(video_path, audio_path)` runs face detection, tracking, cropping and scoring in one call |
| Command line | `run_pipeline.py`, `run_syncnet.py`, `run_visualise.py` run in sequence, plus `demo_syncnet.py` for pre-cropped clips | one `syncnet-python` command that runs detection, tracking, cropping and scoring |
| Output | offset, minimum distance and confidence written to the log; per-frame distances saved as `activesd.pckl` under `--data_dir` | values returned to Python, or written to JSON by the CLI |
| Weights | downloaded by `download_model.sh` | downloaded separately (see above) |
| Visualisation of the result | `run_visualise.py` | not included |

## Errors this fixes

Before the April 2026 update, the original repository pinned `scenedetect==0.5.1` and used `np.int`. Users hit these two errors in `run_pipeline.py`:

```
TypeError: 'tuple' object does not support item assignment
```

This comes from scenedetect 0.5.1's `ContentDetector` running with newer OpenCV (upstream issues [#55](https://github.com/joonson/syncnet_python/issues/55) and [#69](https://github.com/joonson/syncnet_python/issues/69)). This package uses scenedetect 0.6.

```
AttributeError: module 'numpy' has no attribute 'int'.
```

This comes from `.astype(np.int)` in `detectors/s3fd/box_utils.py`. NumPy 1.24 removed `np.int`. This package uses `.astype(int)`.

## Repository layout

The PyPI package contains only the `syncnet_python/` folder. Most of its files (`syncnet_pipeline.py`, `SyncNetInstance.py`, `SyncNetModel.py`, `detectors/` and the scripts named `run_syncnet_pipeline_on_*.py`) come from the SyncNet evaluation code in [MoChaBench](https://github.com/congwei1230/MoChaBench) (`eval-lipsync/script/`), which builds on the original repository. This package adds error handling around the `ffmpeg` calls in `syncnet_pipeline.py`, plus `cli.py`.

The `syncnet/` folder holds a separate refactor with configuration files, logging and batch helpers. It is in this repository only and pip does not install it. `scripts/` has example scripts, and `example/` has a short test video with its audio.

## Credits

The SyncNet model, its pretrained weights and the original code are by Joon Son Chung and Andrew Zisserman ([joonson/syncnet_python](https://github.com/joonson/syncnet_python), [project page](https://www.robots.ox.ac.uk/~vgg/software/lipsync/)). The S3FD face detector weights (`sfd_face.pth`) are downloaded from the same page. The model, detector and pipeline files in `syncnet_python/` come from [MoChaBench](https://github.com/congwei1230/MoChaBench); [NOTICE](NOTICE) lists them.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@InProceedings{Chung16a,
  author       = "Chung, J.~S. and Zisserman, A.",
  title        = "Out of time: automated lip sync in the wild",
  booktitle    = "Workshop on Multi-view Lip-reading, ACCV",
  year         = "2016",
}
```

## License

This repository uses two licenses. [NOTICE](NOTICE) lists which file falls under which.

- The original SyncNet code by Joon Son Chung and the code written for this repository are under the MIT License ([LICENSE](LICENSE)).
- The files taken from MoChaBench are under the Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE)).

The model weights have their own terms. The Oxford VGG SyncNet page says: "The model can be used for research purposes under Creative Commons Attribution License." The MIT and Apache licenses above do not cover the weights.

## Links

- Source and issues: https://github.com/nawta/SyncNet_py309_313
- PyPI: https://pypi.org/project/syncnet-python/
- Original SyncNet: https://github.com/joonson/syncnet_python
