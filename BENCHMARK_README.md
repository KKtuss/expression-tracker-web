# Performance Benchmark Guide

This guide explains how to benchmark the Expression Tracker Web performance on different Render instances.

## Files Added

- `benchmark.py` - Main performance benchmark script
- `run_benchmark.py` - Simple runner script
- `/benchmark` endpoint - HTTP endpoint to run benchmark via API

## Current Performance Issue

The app is running on Render's **free tier** with:
- **512MB RAM** - Insufficient for real-time computer vision
- **Shared CPU** - Variable performance
- **No guaranteed resources**

## Benchmark Script Features

### What It Tests
- **Frame processing time** (milliseconds per frame)
- **Memory usage** during processing
- **Different frame sizes** (480x360, 640x480, 800x600)
- **Success/failure rates**

### Performance Targets
- **Target FPS**: 15 FPS for smooth experience
- **Target frame time**: ~66.7ms per frame
- **Memory**: Should stay under available RAM

### Output Example
```
PERFORMANCE BENCHMARK RESULTS
============================================================
Total frames processed: 30
Successful frames: 30
Failed frames: 0

Frame Processing Times:
  Average: 45.23 ms
  Minimum: 38.12 ms
  Maximum: 52.34 ms
  Std Dev: 3.45 ms

Memory Usage:
  Average: 245.6 MB
  Peak: 267.8 MB

Performance Analysis:
  Target FPS: 15
  Target frame time: 66.7 ms
  Actual average frame time: 45.23 ms
  ✅ PERFORMANCE: ACCEPTABLE (45.23ms <= 66.7ms)
  Theoretical FPS: 22.1

System Resources:
  CPU cores: 1
  Total RAM: 0.5 GB
  Available RAM: 0.4 GB
```

## How to Run Benchmark

### Method 1: HTTP Endpoint (Easiest)
```bash
curl -X POST https://expression-tracker-web.onrender.com/benchmark
```

### Method 2: Local Testing
```bash
cd deployment/
python benchmark.py
```

### Method 3: Using Runner Script
```bash
cd deployment/
python run_benchmark.py
```

## Render Upgrade Recommendations

### Current Free Tier Issues
- **512MB RAM**: Too little for MediaPipe + OpenCV
- **Shared CPU**: Inconsistent performance
- **No CPU guarantee**: Can be throttled

### Recommended Upgrades

#### Option 1: Starter Plan ($7/month)
- **512MB RAM** (same as free)
- **0.5 vCPU** (dedicated)
- **Still might be tight on RAM**

#### Option 2: Standard Plan ($25/month)
- **2GB RAM** ✅ **Recommended**
- **1 vCPU** ✅ **Good for real-time processing**
- **Much better for computer vision workloads**

#### Option 3: Pro Plan ($85/month)
- **8GB RAM** (overkill for this app)
- **2 vCPU** (overkill)
- **Only if you expect high traffic**

## Expected Performance After Upgrade

### On Standard Plan (2GB RAM, 1 vCPU):
- **Frame processing**: 30-50ms per frame
- **Theoretical FPS**: 20-33 FPS
- **Memory usage**: 200-400MB
- **Should be smooth** for real-time use

### On Starter Plan (512MB RAM, 0.5 vCPU):
- **Frame processing**: 60-100ms per frame
- **Theoretical FPS**: 10-16 FPS
- **Memory usage**: 400-500MB (near limit)
- **Might still be laggy**

## Testing After Upgrade

1. **Upgrade your Render instance** to Standard Plan ($25/month)
2. **Wait for redeployment** (2-3 minutes)
3. **Run benchmark**:
   ```bash
   curl -X POST https://expression-tracker-web.onrender.com/benchmark
   ```
4. **Check results** for:
   - Average frame time < 66.7ms
   - Memory usage < 1.5GB
   - Success rate > 95%

## Troubleshooting

### If Benchmark Fails
- Check Render logs for memory issues
- Ensure all dependencies are installed
- Verify MediaPipe is working

### If Performance Still Poor
- Consider Pro Plan for 2 vCPU
- Check for memory leaks in detection code
- Optimize frame processing pipeline

## Cost Analysis

| Plan | Monthly Cost | RAM | vCPU | Expected Performance |
|------|-------------|-----|------|---------------------|
| Free | $0 | 512MB | Shared | ❌ Laggy |
| Starter | $7 | 512MB | 0.5 | ⚠️ Still tight |
| Standard | $25 | 2GB | 1 | ✅ Good |
| Pro | $85 | 8GB | 2 | ✅ Excellent |

**Recommendation**: Start with **Standard Plan ($25/month)** for best performance/cost ratio.
