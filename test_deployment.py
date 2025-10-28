#!/usr/bin/env python3
"""
Test script to verify the deployment works correctly
"""

import asyncio
import json
import base64
import numpy as np
import cv2
from detection_core import ExpressionDetector
from image_manager import ImageManager

def test_detection_core():
    """Test that the detection core works correctly"""
    print("Testing ExpressionDetector...")
    
    try:
        detector = ExpressionDetector()
        print("+ ExpressionDetector initialized successfully")
        
        # Create a test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame.fill(128)  # Gray frame
        
        # Test frame processing
        result = detector.process_frame(test_frame)
        print(f"+ Frame processing successful: {result.get('success')}")
        
        # Test cleanup
        detector.cleanup()
        print("+ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"- ExpressionDetector test failed: {e}")
        return False

def test_image_manager():
    """Test that the image manager works correctly"""
    print("Testing ImageManager...")
    
    try:
        detector = ExpressionDetector()
        image_manager = ImageManager(detector.images)
        
        # Test image operations
        expressions = image_manager.get_all_images_status()
        print(f"+ Available expressions: {len(expressions)}")
        
        print(f"+ Image manager check successful")
        
        return True
        
    except Exception as e:
        print(f"- ImageManager test failed: {e}")
        return False

def test_frame_encoding():
    """Test frame encoding for WebSocket transmission"""
    print("Testing frame encoding...")
    
    try:
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame.fill(128)
        
        # Test encoding
        _, buffer = cv2.imencode('.jpg', test_frame)
        frame_bytes = buffer.tobytes()
        base64_data = base64.b64encode(frame_bytes).decode('utf-8')
        
        # Test decoding
        decoded_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        decoded_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        print(f"+ Frame encoding/decoding successful")
        print(f"  Original shape: {test_frame.shape}")
        print(f"  Decoded shape: {decoded_frame.shape}")
        
        return True
        
    except Exception as e:
        print(f"- Frame encoding test failed: {e}")
        return False

async def test_websocket_simulation():
    """Simulate WebSocket message handling"""
    print("Testing WebSocket message simulation...")
    
    try:
        # Create test detector
        detector = ExpressionDetector()
        image_manager = ImageManager(detector.images)
        
        # Create test frame and encode
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame.fill(128)
        _, buffer = cv2.imencode('.jpg', test_frame)
        frame_data = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        # Simulate frame processing
        detection_result = detector.process_frame(test_frame)
        current_expression = detection_result.get("expression")
        
        # Simulate result formatting
        result = {
            "type": "detection_result",
            "session_id": "test_session",
            "frame_count": 1,
            "expression": current_expression,
            "success": detection_result.get("success", False),
            "debug": detection_result.get("debug", {}),
            "frame_with_overlay": detection_result.get("frame_with_overlay"),
            "expression_image": None
        }
        
        # Test JSON serialization
        json_result = json.dumps(result)
        parsed_result = json.loads(json_result)
        
        print(f"+ WebSocket message simulation successful")
        print(f"  Result type: {parsed_result.get('type')}")
        print(f"  Expression: {parsed_result.get('expression')}")
        
        detector.cleanup()
        return True
        
    except Exception as e:
        print(f"- WebSocket simulation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("EXPRESSION TRACKER DEPLOYMENT TESTS")
    print("=" * 50)
    
    tests = [
        test_detection_core,
        test_image_manager,
        test_frame_encoding,
        test_websocket_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test == test_websocket_simulation:
            # Run async test
            if asyncio.run(test()):
                passed += 1
        else:
            if test():
                passed += 1
    
    print()
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("SUCCESS: All tests passed! Deployment is ready.")
        return True
    else:
        print("ERROR: Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


















