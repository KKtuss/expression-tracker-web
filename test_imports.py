#!/usr/bin/env python3
"""
Minimal test to check if all imports work
"""

def test_imports():
    try:
        print("Testing imports...")
        
        # Test basic imports
        import cv2
        print("✓ OpenCV imported")
        
        import numpy as np
        print("✓ NumPy imported")
        
        import mediapipe as mp
        print("✓ MediaPipe imported")
        
        from PIL import Image
        print("✓ PIL imported")
        
        # Test MediaPipe initialization
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
        print("✓ MediaPipe FaceMesh initialized")
        
        mp_hands = mp.solutions.hands.Hands()
        print("✓ MediaPipe Hands initialized")
        
        print("✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()
