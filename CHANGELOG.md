# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-10-06

### Fixed
- Pin `scenedetect` to `>=0.6.0,<0.7`. scenedetect 0.7 removed
  `scenedetect.video_manager`, which `syncnet_pipeline.py` imports.
- Depend on `scenedetect` without the `[opencv]` extra, so
  `opencv-contrib-python` is the only package that installs `cv2`.
  The extra also installed `opencv-python`, a second copy of `cv2`.
- `syncnet_python/__init__.py` raises `ImportError` with the original
  message when an import fails. It used to set the exports to `None`.

### Changed
- License metadata is `MIT AND Apache-2.0` (PEP 639). The wheel and sdist
  include `LICENSE`, `LICENSE-APACHE` and `NOTICE`. Files taken from
  MoChaBench are Apache-2.0; NOTICE lists them.
- Project URLs point at https://github.com/nawta/SyncNet_py309_313.
- README rewritten with a tested quickstart.

## [0.2.2] - 2025-07-07

Published on PyPI; the code was committed to the repository in 0.2.3.

### Added
- `safe_syncnet_utils.py` with `safe_syncnet_inference`,
  `extract_audio_from_video` and `calculate_lse_metrics`.
- `SyncNetPipeline.inference(audio_path=None)` extracts the audio track
  from the video.

### Fixed
- In `inference`, the ffmpeg-python calls for video conversion, frame
  extraction and audio resampling catch `ffmpeg.Error` and raise
  `RuntimeError("FFmpeg <step> failed: ...")`. The message contains
  `str(ffmpeg.Error)`, which is "ffmpeg error (see stderr output for
  detail)"; ffmpeg's own output goes to the console. The two ffmpeg calls
  per face track (frame and audio extraction from each crop) are not
  wrapped.
- `inference` raises `RuntimeError("No frames were extracted from the
  video")` when frame extraction yields no images.
- The new `_extract_audio_from_video` runs ffmpeg with `subprocess.run`
  and raises `RuntimeError("FFmpeg audio extraction failed: <ffmpeg
  stderr>")`, or a "FFmpeg not found" `RuntimeError` when the binary is
  missing.

## [0.2.1] - 2025-01-06

### Fixed
- FFmpeg NoneType error in `syncnet_pipeline.py` when ffmpeg_bin is None
- Improved error handling for FFmpeg subprocess calls
- Fixed import paths for direct script execution (relative imports with fallback)

### Changed
- Replaced `subprocess.call` with `subprocess.run` for better error handling
- Added explicit FFmpeg binary path validation
- Improved error messages for FFmpeg-related failures

## [0.2.0] - 2025-01-24

### Added
- Abstract base classes for all major components (BaseModel, AudioProcessor, VideoProcessor, etc.)
- Comprehensive exception hierarchy with specific error types
- Configuration management system with validation
- Structured logging with color support and JSON output
- Memory management utilities and monitoring
- Parallel video processing support
- Streaming audio processing capabilities
- Optimized synchronization analyzer with caching
- Factory pattern for model creation
- Batch processing utilities
- YAML/JSON configuration support
- Detailed analysis testing with per-crop offsets and min distances
- Comprehensive functionality verification tests

### Changed
- Complete refactoring of core modules with clean architecture
- Separated concerns into distinct modules (audio, video, models, sync_analyzer)
- Improved error handling throughout the codebase
- Added input validation and sanitization
- Optimized tensor operations for better performance
- Modernized code with Python 3.13 features while maintaining 3.9+ compatibility
- Fixed import paths in syncnet_python module for proper functionality
- Updated type definitions with Python 3.9 compatibility fallbacks

### Removed
- Duplicate code in script/ directory
- Build artifacts and temporary files
- Redundant documentation files
- Hard-coded configuration values
- Intermediate test files and temporary outputs

### Fixed
- Memory leaks in video processing
- Thread safety issues in parallel processing
- Improper resource cleanup
- Import path issues in syncnet_pipeline.py
- Legacy model compatibility layer
- PyYAML and psutil optional dependencies

### Tested
- Successfully verified all APIs with example video and audio files
- Confirmed face detection working at 100% success rate (134 frames)
- Validated audio-visual sync analysis with 1 frame offset detection
- Tested command line interface functionality
- Verified error handling and edge cases
- Performance tested: 191.4 fps processing, 0.646s compute time

### Performance Results
- Example video: example/video.avi (134 frames)
- Example audio: example/speech.wav
- AV Offset detected: 1 frame (excellent sync)
- Confidence: 4.500 (very high)
- Min distance: 9.316
- Face detection confidence: 0.959-1.000 range
- Processing speed: 191.4 fps

## [0.1.1] - 2025-01-24

### Added
- Proper attribution to the original [SyncNet implementation](https://github.com/joonson/syncnet_python) by Joon Son Chung
- Python 3.9 compatibility fixes for type hints with fallback imports
- PYPI_RELEASE_NOTES.md for tracking PyPI releases

### Changed
- Updated project description to clarify this is an updated version for modern Python versions
- Modified type definitions in `syncnet/core/types.py` to support Python 3.9+

### Fixed
- Type hint compatibility issues for Python versions prior to 3.10

## [0.1.0] - 2025-01-24

### Added
- Complete refactoring for Python 3.13 compatibility
- Modern Python features:
  - Type hints throughout the codebase
  - Async/await support for concurrent processing
  - Exception groups for better error handling
  - Dataclasses for configuration management
- Modular package structure:
  - `syncnet/core/` - Refactored core components
  - `syncnet_python/` - Legacy compatibility wrapper
- PyPI package configuration with `pyproject.toml`
- Comprehensive documentation:
  - README.md with installation and usage instructions
  - CLAUDE.md for AI-assisted development
- Development tools configuration:
  - Black formatter settings
  - Ruff linter configuration
  - MyPy type checking setup

### Changed
- Migrated from `setup.py` to modern `pyproject.toml` configuration
- Replaced deprecated `torch.autograd.Variable` with direct tensor usage
- Updated S3FD face detector model loading with proper state dict mapping
- Reorganized directory structure for better maintainability
- Cleaned up intermediate files and temporary outputs

### Fixed
- PyTorch deprecation warnings
- S3FD model state dict loading issues with parameter name mapping
- Legacy model compatibility through dedicated compatibility layer

### Tested
- Successfully tested on example video:
  - AV offset: 1 frame
  - Confidence: 4.568
- Verified functionality matches original implementation

### Technical Details
- Python 3.13 primary development target
- Backward compatibility to Python 3.9
- Dependencies:
  - PyTorch >= 2.0.0
  - OpenCV >= 4.8.0
  - NumPy >= 1.24.0
  - SciPy >= 1.10.0
  - Other supporting libraries

### Notes
- Model weights (`sfd_face.pth`, `syncnet_v2.model`) must be downloaded separately due to PyPI size limitations
- Primary testing done on Python 3.13; other versions may have minor compatibility issues