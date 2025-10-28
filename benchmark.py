#!/usr/bin/env python3
"""
Performance benchmark script for Expression Tracker Web
Tests frame processing performance and memory usage
"""

import time
import cv2
import numpy as np
import psutil
import sys
from detection_core import ExpressionDetector
from image_manager import ImageManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    def __init__(self):
        """Initialize benchmark with detection components"""
        self.detector = None
        self.image_manager = None
        self.setup_detection_components()
        
    def setup_detection_components(self):
        """Initialize detection components (same as production)"""
        try:
            # Initialize detector (no arguments needed)
            self.detector = ExpressionDetector()
            
            logger.info("Detection components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize detection components: {e}")
            raise
    
    def generate_test_frame(self, width=640, height=480):
        """Generate a synthetic test frame with face-like features"""
        # Create a simple test frame with some features
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add some face-like features for detection
        # Draw a simple face rectangle
        cv2.rectangle(frame, (200, 150), (440, 380), (200, 180, 160), -1)
        
        # Draw eyes
        cv2.circle(frame, (280, 220), 20, (0, 0, 0), -1)
        cv2.circle(frame, (360, 220), 20, (0, 0, 0), -1)
        
        # Draw mouth (smile)
        cv2.ellipse(frame, (320, 300), (30, 15), 0, 0, 180, (0, 0, 0), 2)
        
        return frame
    
    def benchmark_frame_processing(self, num_frames=50, frame_width=640, frame_height=480):
        """Benchmark frame processing performance"""
        logger.info(f"Starting frame processing benchmark...")
        logger.info(f"Processing {num_frames} frames of {frame_width}x{frame_height}")
        
        processing_times = []
        memory_usage = []
        
        # Warmup - process a few frames to initialize models
        logger.info("Warming up detection models...")
        for i in range(5):
            test_frame = self.generate_test_frame(frame_width, frame_height)
            try:
                _ = self.detector.process_frame(test_frame)
            except Exception as e:
                logger.warning(f"Warmup frame {i} failed: {e}")
        
        logger.info("Starting benchmark...")
        
        for i in range(num_frames):
            # Generate test frame
            test_frame = self.generate_test_frame(frame_width, frame_height)
            
            # Measure processing time
            start_time = time.time()
            try:
                result = self.detector.process_frame(test_frame)
                end_time = time.time()
                
                processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
                processing_times.append(processing_time)
                
                # Get memory usage
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                memory_usage.append(memory_mb)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{num_frames} frames")
                    
            except Exception as e:
                logger.error(f"Frame {i} processing failed: {e}")
                processing_times.append(0)  # Record failure as 0ms
        
        return processing_times, memory_usage
    
    def analyze_results(self, processing_times, memory_usage):
        """Analyze and display benchmark results"""
        # Filter out failed frames (0ms)
        valid_times = [t for t in processing_times if t > 0]
        failed_frames = len(processing_times) - len(valid_times)
        
        if not valid_times:
            logger.error("No frames processed successfully!")
            return
        
        # Calculate statistics
        avg_time = np.mean(valid_times)
        min_time = np.min(valid_times)
        max_time = np.max(valid_times)
        std_time = np.std(valid_times)
        
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        
        # Performance thresholds
        target_fps = 15  # Target 15 FPS for smooth experience
        target_frame_time = 1000 / target_fps  # ~66.7ms per frame
        
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE BENCHMARK RESULTS")
        logger.info("="*60)
        
        logger.info(f"Total frames processed: {len(processing_times)}")
        logger.info(f"Successful frames: {len(valid_times)}")
        logger.info(f"Failed frames: {failed_frames}")
        
        logger.info(f"\nFrame Processing Times:")
        logger.info(f"  Average: {avg_time:.2f} ms")
        logger.info(f"  Minimum: {min_time:.2f} ms")
        logger.info(f"  Maximum: {max_time:.2f} ms")
        logger.info(f"  Std Dev: {std_time:.2f} ms")
        
        logger.info(f"\nMemory Usage:")
        logger.info(f"  Average: {avg_memory:.1f} MB")
        logger.info(f"  Peak: {max_memory:.1f} MB")
        
        logger.info(f"\nPerformance Analysis:")
        logger.info(f"  Target FPS: {target_fps}")
        logger.info(f"  Target frame time: {target_frame_time:.1f} ms")
        logger.info(f"  Actual average frame time: {avg_time:.2f} ms")
        
        if avg_time <= target_frame_time:
            logger.info(f"  ✅ PERFORMANCE: ACCEPTABLE ({avg_time:.2f}ms <= {target_frame_time:.1f}ms)")
        else:
            logger.info(f"  ❌ PERFORMANCE: NEEDS IMPROVEMENT ({avg_time:.2f}ms > {target_frame_time:.1f}ms)")
        
        # Calculate theoretical FPS
        theoretical_fps = 1000 / avg_time if avg_time > 0 else 0
        logger.info(f"  Theoretical FPS: {theoretical_fps:.1f}")
        
        logger.info(f"\nSystem Resources:")
        logger.info(f"  CPU cores: {psutil.cpu_count()}")
        logger.info(f"  Total RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
        logger.info(f"  Available RAM: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
        
        logger.info("="*60)
        
        return {
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'avg_memory_mb': avg_memory,
            'max_memory_mb': max_memory,
            'successful_frames': len(valid_times),
            'failed_frames': failed_frames,
            'theoretical_fps': theoretical_fps,
            'performance_acceptable': avg_time <= target_frame_time
        }

def main():
    """Main benchmark function"""
    logger.info("Expression Tracker Web - Performance Benchmark")
    logger.info("=" * 50)
    
    try:
        # Initialize benchmark
        benchmark = PerformanceBenchmark()
        
        # Run benchmark with different frame sizes
        test_configs = [
            {"frames": 30, "width": 480, "height": 360, "name": "Small (480x360)"},
            {"frames": 30, "width": 640, "height": 480, "name": "Medium (640x480)"},
            {"frames": 20, "width": 800, "height": 600, "name": "Large (800x600)"}
        ]
        
        all_results = []
        
        for config in test_configs:
            logger.info(f"\nTesting {config['name']}...")
            
            processing_times, memory_usage = benchmark.benchmark_frame_processing(
                num_frames=config['frames'],
                frame_width=config['width'],
                frame_height=config['height']
            )
            
            results = benchmark.analyze_results(processing_times, memory_usage)
            if results:
                results['config'] = config['name']
                all_results.append(results)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK SUMMARY")
        logger.info(f"{'='*60}")
        
        for result in all_results:
            status = "✅ ACCEPTABLE" if result['performance_acceptable'] else "❌ NEEDS IMPROVEMENT"
            logger.info(f"{result['config']}: {result['avg_time_ms']:.2f}ms avg, {result['theoretical_fps']:.1f} FPS - {status}")
        
        # Overall recommendation
        worst_time = max([r['avg_time_ms'] for r in all_results])
        if worst_time <= 66.7:  # 15 FPS threshold
            logger.info(f"\n🎉 OVERALL: Performance is acceptable for real-time use!")
        else:
            logger.info(f"\n⚠️  OVERALL: Performance needs improvement. Consider upgrading instance.")
            logger.info(f"   Worst case: {worst_time:.2f}ms per frame")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


















