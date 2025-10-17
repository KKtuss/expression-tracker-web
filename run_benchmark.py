#!/usr/bin/env python3
"""
Simple script to run the performance benchmark
Usage: python run_benchmark.py
"""

import subprocess
import sys
import os

def main():
    """Run the benchmark script"""
    print("Expression Tracker Web - Performance Benchmark Runner")
    print("=" * 50)
    
    # Check if benchmark.py exists
    if not os.path.exists("benchmark.py"):
        print("❌ Error: benchmark.py not found in current directory")
        sys.exit(1)
    
    try:
        # Run the benchmark
        print("🚀 Starting performance benchmark...")
        print("This will test frame processing performance and memory usage.")
        print("It may take a few minutes to complete.\n")
        
        result = subprocess.run([sys.executable, "benchmark.py"], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("\n✅ Benchmark completed successfully!")
        else:
            print(f"\n❌ Benchmark failed with return code: {result.returncode}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
