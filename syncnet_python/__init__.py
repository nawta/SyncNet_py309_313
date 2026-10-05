"""SyncNet Python: Audio-visual synchronization detection.

This package provides a PyTorch implementation of SyncNet for detecting
synchronization between audio and video in multimedia content.
"""

__version__ = "0.2.3"

# Import main components. An ImportError here (for example a missing
# dependency or an incompatible scenedetect version) is raised with its
# original message instead of being replaced by None.
try:
    from .syncnet_pipeline import SyncNetPipeline
    from .SyncNetModel import S as SyncNetModel
    from .SyncNetInstance import SyncNetInstance
    from .safe_syncnet_utils import (
        safe_syncnet_inference,
        extract_audio_from_video,
        calculate_lse_metrics
    )
except ImportError as e:
    raise ImportError(
        f"syncnet_python could not import its components: {e}. "
        "Check that the dependencies are installed with the versions listed "
        "in pyproject.toml (scenedetect must be >=0.6,<0.7)."
    ) from e

__all__ = [
    "SyncNetPipeline",
    "SyncNetModel", 
    "SyncNetInstance",
    "safe_syncnet_inference",
    "extract_audio_from_video", 
    "calculate_lse_metrics",
    "__version__"
]

def get_version():
    """Get package version."""
    return __version__