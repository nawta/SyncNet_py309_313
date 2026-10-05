"""SyncNet Python: Audio-visual synchronization detection.

This package provides a PyTorch implementation of SyncNet for detecting
synchronization between audio and video in multimedia content.
"""

__version__ = "0.2.2"

# Import main components
try:
    from .syncnet_pipeline import SyncNetPipeline
    from .SyncNetModel import S as SyncNetModel
    from .SyncNetInstance import SyncNetInstance
    from .safe_syncnet_utils import (
        safe_syncnet_inference,
        extract_audio_from_video,
        calculate_lse_metrics
    )
except ImportError:
    # Fallback for development
    SyncNetPipeline = None
    SyncNetModel = None
    SyncNetInstance = None
    safe_syncnet_inference = None
    extract_audio_from_video = None
    calculate_lse_metrics = None

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