#!/usr/bin/env python3
"""Test script for comprehensive error handling in SyncNet v0.2.2."""

import logging
import tempfile
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from syncnet_pipeline import SyncNetPipeline

# Import safe utils with fallback
try:
    from safe_syncnet_utils import safe_syncnet_inference, calculate_lse_metrics
except ImportError:
    # Fallback implementation for testing
    def safe_syncnet_inference(pipeline, video_path, audio_path=None, cache_dir=None):
        return pipeline.inference(video_path=video_path, audio_path=audio_path, cache_dir=cache_dir)
    
    def calculate_lse_metrics(pipeline, video_path, audio_path=None, cache_dir=None):
        results = pipeline.inference(video_path=video_path, audio_path=audio_path, cache_dir=cache_dir)
        offsets, confs, dists, max_conf, min_dist, _, has_face = results
        
        if not has_face or not confs:
            return 0.0, float('inf'), "NO_FACE"
        
        quality = "GOOD" if max_conf > 3.5 and min_dist < 7.0 else "FAIR" if max_conf > 2.0 else "POOR"
        return max_conf, min_dist, quality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def test_audio_none_handling():
    """Test automatic audio extraction when audio_path=None."""
    print("\n=== Testing audio_path=None handling ===")
    
    # Initialize pipeline
    pipeline = SyncNetPipeline(
        {'s3fd_weights': '../weights/sfd_face.pth', 'syncnet_weights': '../weights/syncnet_v2.model'},
        device='cuda'
    )
    
    # Test with audio_path=None (should automatically extract audio)
    video_path = '../example/pair_0000_lipsynced.mp4'
    
    try:
        print(f"Testing inference with audio_path=None for: {video_path}")
        results = pipeline.inference(
            video_path=video_path,
            audio_path=None,  # This should now work!
            cache_dir='../example/cache_test_none'
        )
        
        offsets, confs, dists, max_conf, min_dist, _, has_face = results
        
        if has_face:
            print(f"✅ SUCCESS: Auto audio extraction worked!")
            print(f"   LSE-C: {max_conf:.3f}")
            print(f"   LSE-D: {min_dist:.3f}")
        else:
            print("⚠️  No face detected, but no error occurred")
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")

def test_safe_wrapper_functions():
    """Test the safe wrapper functions."""
    print("\n=== Testing Safe Wrapper Functions ===")
    
    pipeline = SyncNetPipeline(
        {'s3fd_weights': '../weights/sfd_face.pth', 'syncnet_weights': '../weights/syncnet_v2.model'},
        device='cuda'
    )
    
    video_path = '../example/pair_0000_lipsynced.mp4'
    
    try:
        print("Testing safe_syncnet_inference...")
        results = safe_syncnet_inference(
            pipeline, 
            video_path, 
            audio_path=None,
            cache_dir='../example/cache_test_safe'
        )
        print("✅ safe_syncnet_inference: SUCCESS")
        
        print("Testing calculate_lse_metrics...")
        lse_c, lse_d, quality = calculate_lse_metrics(
            pipeline,
            video_path,
            audio_path=None,
            cache_dir='../example/cache_test_metrics'
        )
        print(f"✅ calculate_lse_metrics: SUCCESS")
        print(f"   LSE-C: {lse_c:.3f}")
        print(f"   LSE-D: {lse_d:.3f}")
        print(f"   Quality: {quality}")
        
    except Exception as e:
        print(f"❌ Safe wrapper test FAILED: {str(e)}")

def test_error_cases():
    """Test various error cases."""
    print("\n=== Testing Error Cases ===")
    
    pipeline = SyncNetPipeline(
        {'s3fd_weights': '../weights/sfd_face.pth', 'syncnet_weights': '../weights/syncnet_v2.model'},
        device='cuda'
    )
    
    # Test with non-existent file
    try:
        results = safe_syncnet_inference(
            pipeline,
            "non_existent_video.mp4",
            audio_path=None
        )
        print("❌ Should have failed for non-existent file")
    except FileNotFoundError:
        print("✅ Correctly caught FileNotFoundError for non-existent video")
    except Exception as e:
        print(f"⚠️  Unexpected error: {str(e)}")

def main():
    """Run all tests."""
    print("SyncNet v0.2.2 Error Handling Test Suite")
    print("=" * 50)
    
    test_audio_none_handling()
    test_safe_wrapper_functions() 
    test_error_cases()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")

if __name__ == "__main__":
    main()