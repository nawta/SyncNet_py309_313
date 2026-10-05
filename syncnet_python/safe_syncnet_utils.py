"""Safe SyncNet utilities with comprehensive error handling."""

import logging
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, List
try:
    from .syncnet_pipeline import SyncNetPipeline
except ImportError:
    from syncnet_pipeline import SyncNetPipeline


def safe_syncnet_inference(
    pipeline: SyncNetPipeline,
    video_path: str,
    audio_path: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Tuple[List[int], List[float], List[float], float, float, str, bool]:
    """
    Safe SyncNet inference with automatic audio extraction if needed.
    
    This function provides a wrapper around SyncNetPipeline.inference() that:
    1. Handles audio_path=None by automatically extracting audio from video
    2. Provides comprehensive error handling and logging
    3. Ensures proper cleanup of temporary files
    
    Args:
        pipeline: Initialized SyncNetPipeline instance
        video_path: Path to input video file
        audio_path: Path to audio file (None for auto-extraction from video)
        cache_dir: Directory for temporary files (None for auto-cleanup)
        
    Returns:
        Tuple of (offsets, confidences, distances, max_conf, min_dist, s3fd_json, has_face)
        
    Raises:
        RuntimeError: If processing fails at any stage
        FileNotFoundError: If video file doesn't exist
        ValueError: If video format is not supported
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if audio_path and not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    logging.info(f"Starting safe SyncNet inference for video: {video_path}")
    
    try:
        # Use the updated inference method that handles audio_path=None
        results = pipeline.inference(
            video_path=video_path,
            audio_path=audio_path,
            cache_dir=cache_dir
        )
        
        offsets, confs, dists, max_conf, min_dist, s3fd_json, has_face = results
        
        # Validate results
        if not has_face:
            logging.warning("No faces detected in the video")
        elif not offsets:
            logging.warning("No valid face tracks found")
        else:
            logging.info(f"Successfully processed {len(offsets)} face tracks")
            logging.info(f"LSE-C (max confidence): {max_conf:.3f}")
            logging.info(f"LSE-D (min distance): {min_dist:.3f}")
        
        return results
        
    except Exception as e:
        logging.error(f"SyncNet inference failed: {str(e)}")
        raise RuntimeError(f"SyncNet processing failed: {str(e)}") from e


def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to input video file
        audio_path: Path for output audio file
        
    Raises:
        RuntimeError: If audio extraction fails
        FileNotFoundError: If video file doesn't exist
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create a temporary pipeline instance just for audio extraction
    pipeline = SyncNetPipeline()
    
    try:
        pipeline._extract_audio_from_video(video_path, audio_path)
        logging.info(f"Successfully extracted audio from {video_path} to {audio_path}")
    except Exception as e:
        logging.error(f"Audio extraction failed: {str(e)}")
        raise RuntimeError(f"Audio extraction failed: {str(e)}") from e


def calculate_lse_metrics(
    pipeline: SyncNetPipeline,
    video_path: str,
    audio_path: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Tuple[float, float, str]:
    """
    Calculate LSE-C and LSE-D metrics for a video.
    
    Args:
        pipeline: Initialized SyncNetPipeline instance
        video_path: Path to input video file
        audio_path: Path to audio file (None for auto-extraction)
        cache_dir: Directory for temporary files
        
    Returns:
        Tuple of (lse_c, lse_d, quality_assessment)
        
    Raises:
        RuntimeError: If metric calculation fails
    """
    try:
        results = safe_syncnet_inference(pipeline, video_path, audio_path, cache_dir)
        offsets, confs, dists, max_conf, min_dist, s3fd_json, has_face = results
        
        if not has_face:
            return 0.0, float('inf'), "NO_FACE"
        
        if not confs or not dists:
            return 0.0, float('inf'), "NO_TRACKS"
        
        # LSE-C is the maximum confidence across all tracks
        lse_c = max_conf
        
        # LSE-D is the minimum distance across all tracks  
        lse_d = min_dist
        
        # Quality assessment based on typical thresholds
        if lse_c > 3.5 and lse_d < 7.0:
            quality = "GOOD"
        elif lse_c > 2.0 and lse_d < 10.0:
            quality = "FAIR"
        else:
            quality = "POOR"
        
        logging.info(f"LSE Metrics - C: {lse_c:.3f}, D: {lse_d:.3f}, Quality: {quality}")
        
        return lse_c, lse_d, quality
        
    except Exception as e:
        logging.error(f"LSE metric calculation failed: {str(e)}")
        raise RuntimeError(f"LSE metric calculation failed: {str(e)}") from e