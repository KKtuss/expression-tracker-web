#!/usr/bin/env python3
"""
Optimized unified detection system for real-time performance
Replaces multiple separate detectors with a single efficient system
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import Dict, List, Tuple, Optional
from collections import deque

class OptimizedDetector:
    """
    Unified detection system that combines face, eye, and hand detection
    for maximum real-time performance
    """
    
    def __init__(self):
        # Initialize MediaPipe solutions with optimized settings
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=False,  # Disable for speed
            min_detection_confidence=0.3,  # Lower threshold for speed
            min_tracking_confidence=0.3,   # Lower threshold for speed
            static_image_mode=False,
            max_num_faces=1
        )
        
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Allow both hands for proper gesture detection
            min_detection_confidence=0.3,  # Lower threshold for speed
            min_tracking_confidence=0.3    # Lower threshold for speed
        )
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.last_performance_check = 0
        
        # Frame skipping for performance
        self.frame_skip = 1  # Process every frame
        self.frame_counter = 0
        
        # Cached results for temporal smoothing
        self.prev_landmarks = None
        self.prev_hands = None
        self.expression_history = deque(maxlen=5)
        self.last_result = None
        
        # Pre-computed landmark indices for performance
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.MOUTH_INDICES = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        
        print("OptimizedDetector initialized for real-time performance")
    
    def process_frame(self, frame):
        """Main processing function - optimized for real-time performance"""
        start_time = time.perf_counter()
        
        # Frame skipping for performance (disabled for debugging)
        self.frame_counter += 1
        # Temporarily disable frame skipping for debugging
        # if self.frame_counter % self.frame_skip != 0:
        #     # Return cached result for skipped frames
        #     if self.last_result:
        #         self.last_result["frame_time_ms"] = (time.perf_counter() - start_time) * 1000
        #         return self.last_result
        
        # Convert to RGB once
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face and hands simultaneously
        face_result = self.mp_face_mesh.process(rgb)
        hands_result = self.mp_hands.process(rgb)
        
        # Extract data efficiently
        landmarks = self._extract_landmarks_fast(face_result)
        hands_data = self._extract_hands_fast(hands_result)
        
        # Get face coordinates for hand touching head detection
        face_coords = self._get_face_coordinates(face_result, frame.shape)
        frame_shape = frame.shape
        
        # Apply temporal smoothing
        landmarks = self._smooth_landmarks(landmarks)
        hands_data = self._smooth_hands(hands_data)
        
        # Fast expression detection
        expressions = self._detect_expressions_fast(landmarks, hands_data, face_coords, frame_shape)
        
        # Debug logging for landmarks
        if landmarks is not None:
            print(f"Face detected - landmarks shape: {landmarks.shape}")
        else:
            print("No face detected")
        
        # Performance tracking
        frame_time = (time.perf_counter() - start_time) * 1000
        self.frame_times.append(frame_time)
        
        # Performance monitoring
        self._check_performance()
        
        # Cache result for frame skipping
        result = {
            "expressions": expressions,
            "landmarks": landmarks,
            "hands": hands_data,
            "frame_time_ms": frame_time,
            "debug": {
                "detected_expression": expressions.get("primary", "none"),
                "frame_time": f"{frame_time:.1f}ms",
                "performance": "good" if frame_time < 15 else "slow"
            }
        }
        
        self.last_result = result
        return result
    
    def _extract_landmarks_fast(self, face_result):
        """Fast landmark extraction"""
        if not face_result.multi_face_landmarks:
            return None
        
        landmarks = np.empty((468, 3), dtype=np.float32)
        for i, lm in enumerate(face_result.multi_face_landmarks[0].landmark):
            landmarks[i] = [lm.x, lm.y, lm.z]
        
        return landmarks
    
    def _extract_hands_fast(self, hands_result):
        """Fast hand extraction"""
        if not hands_result.multi_hand_landmarks:
            return []
        
        hands = []
        for hand_landmarks in hands_result.multi_hand_landmarks:
            landmarks = np.empty((21, 3), dtype=np.float32)
            for i, lm in enumerate(hand_landmarks.landmark):
                landmarks[i] = [lm.x, lm.y, lm.z]
            hands.append(landmarks)
        
        return hands
    
    def _get_face_coordinates(self, face_result, frame_shape):
        """Get face bounding box coordinates for hand touching head detection"""
        if not face_result.multi_face_landmarks:
            return None
        
        # Get face landmarks
        face_landmarks = face_result.multi_face_landmarks[0].landmark
        frame_h, frame_w = frame_shape[:2]
        
        # Find bounding box of face
        x_coords = [lm.x for lm in face_landmarks]
        y_coords = [lm.y for lm in face_landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Convert to pixel coordinates
        fx = int(x_min * frame_w)
        fy = int(y_min * frame_h)
        fw = int((x_max - x_min) * frame_w)
        fh = int((y_max - y_min) * frame_h)
        
        return (fx, fy, fw, fh)
    
    def _smooth_landmarks(self, landmarks, alpha=0.7):
        """Temporal smoothing for landmarks"""
        if landmarks is None:
            return None
        
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return landmarks
        
        smoothed = self.prev_landmarks + alpha * (landmarks - self.prev_landmarks)
        self.prev_landmarks = smoothed
        return smoothed
    
    def _smooth_hands(self, hands, alpha=0.7):
        """Temporal smoothing for hands"""
        if not hands:
            return []
        
        if self.prev_hands is None:
            self.prev_hands = hands
            return hands
        
        # Simple smoothing - just use current hands for now
        self.prev_hands = hands
        return hands
    
    def _detect_expressions_fast(self, landmarks, hands_data, face_coords, frame_shape):
        """Fast expression detection using MediaPipe landmarks"""
        if landmarks is None:
            return {"primary": "none", "confidence": 0.0}
        
        # Eye state detection
        eye_state = self._detect_eye_state(landmarks)
        
        # Face expressions
        face_expressions = self._detect_face_expressions(landmarks)
        
        # Hand gestures with face coordinates for hand touching head detection
        hand_gestures = self._detect_hand_gestures(hands_data, face_coords, frame_shape)
        
        # Combine results with priority system
        primary_expression = self._determine_primary_expression(
            eye_state, face_expressions, hand_gestures
        )
        
        return {
            "primary": primary_expression,
            "eye_state": eye_state,
            "face_expressions": face_expressions,
            "hand_gestures": hand_gestures,
            "confidence": 0.8  # Simplified confidence
        }
    
    def _detect_eye_state(self, landmarks):
        """Detect eye open/closed state"""
        if landmarks is None:
            return "unknown"
        
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2
        
        if avg_ear > 0.25:
            return "open"
        elif avg_ear < 0.20:
            return "closed"
        else:
            return "partial"
    
    def _calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio"""
        eye_points = landmarks[eye_indices]
        
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # EAR calculation
        ear = (A + B) / (2.0 * C + 1e-6)
        return ear
    
    def _detect_face_expressions(self, landmarks):
        """Detect facial expressions using improved logic"""
        if landmarks is None:
            return {}
        
        expressions = {}
        
        # Improved smile detection using multiple mouth landmarks
        # Get mouth corner landmarks
        left_corner = landmarks[61]   # Left mouth corner
        right_corner = landmarks[291] # Right mouth corner
        
        # Get upper lip center
        upper_center = landmarks[13]
        
        # Calculate mouth width
        mouth_width = np.linalg.norm(left_corner - right_corner)
        
        # Calculate mouth height (from corners to center)
        left_height = abs(left_corner[1] - upper_center[1])
        right_height = abs(right_corner[1] - upper_center[1])
        avg_height = (left_height + right_height) / 2
        
        # Calculate smile ratio (higher = more smiling)
        if mouth_width > 0:
            smile_ratio = avg_height / mouth_width
            expressions["smiling"] = smile_ratio > 0.08  # Lowered threshold for better detection
            # Debug logging
            print(f"Smile detection - ratio: {smile_ratio:.3f}, threshold: 0.08, smiling: {expressions['smiling']}")
        else:
            expressions["smiling"] = False
        
        # Mouth open detection (improved)
        # Get upper and lower lip landmarks
        upper_lip = landmarks[13]  # Upper lip center
        lower_lip = landmarks[14]  # Lower lip center
        
        mouth_open_distance = np.linalg.norm(upper_lip - lower_lip)
        expressions["mouth_open"] = mouth_open_distance > 0.080  # Much higher threshold to only detect truly wide open mouths
        # Debug logging
        print(f"Mouth open detection - distance: {mouth_open_distance:.3f}, threshold: 0.080, mouth_open: {expressions['mouth_open']}")
        
        return expressions
    
    def _detect_hand_gestures(self, hands_data, face_coords, frame_shape):
        """Detect hand gestures using proper logic with face coordinates"""
        if not hands_data:
            return {}
        
        gestures = {}
        left_hand_gesture = None
        right_hand_gesture = None
        left_raised = False
        right_raised = False
        
        for i, hand in enumerate(hands_data):
            # Detect individual finger extensions
            finger_info = self._get_finger_info(hand)
            num_fingers = finger_info["num_fingers"]
            
            # Detect specific gesture
            gesture = self._detect_gesture_from_fingers(finger_info)
            
            # Check if hand is raised (more sensitive)
            wrist_y = hand[0][1]  # Wrist is landmark 0
            hand_raised = wrist_y < 0.6  # Even more sensitive threshold for hands raised
            
            if i == 0:  # First hand (usually left)
                left_hand_gesture = gesture
                left_raised = hand_raised
                gestures["left_hand"] = gesture
                gestures["left_raised"] = left_raised
            else:  # Second hand (usually right)
                right_hand_gesture = gesture
                right_raised = hand_raised
                gestures["right_hand"] = gesture
                gestures["right_raised"] = right_raised
        
        # Check for hand touching head using proper face coordinates
        left_touching_head = False
        right_touching_head = False
        
        if face_coords is not None:
            # Use the same logic as the original hand tracker
            for i, hand in enumerate(hands_data):
                touching_head = self._is_hand_touching_head(hand, face_coords, frame_shape)
                
                if i == 0:  # Left hand
                    left_touching_head = touching_head
                else:  # Right hand
                    right_touching_head = touching_head
        
        # Determine special gestures (prioritize hand touching head)
        if left_touching_head or right_touching_head:
            gestures["special_gesture"] = "hand_touching_head"
        elif left_raised and right_raised:
            gestures["special_gesture"] = "both_hands_raised"
        elif left_raised or right_raised:
            gestures["special_gesture"] = "one_hand_raised"
        
        return gestures
    
    def _is_hand_touching_head(self, hand_landmarks, face_coords, frame_shape):
        """Check if hand landmarks are touching the head area (same logic as original)"""
        if face_coords is None:
            return False
        
        fx, fy, fw, fh = face_coords
        frame_h, frame_w = frame_shape[:2]
        
        # Convert face coordinates to normalized coordinates (0-1)
        face_left = fx / frame_w
        face_right = (fx + fw) / frame_w
        face_top = fy / frame_h
        face_bottom = (fy + fh) / frame_h
        
        # Expand the head area slightly for better detection
        head_expansion = 0.05  # 5% expansion (reduced for more precision)
        face_left = max(0, face_left - head_expansion)
        face_right = min(1, face_right + head_expansion)
        face_top = max(0, face_top - head_expansion)
        face_bottom = min(1, face_bottom + head_expansion)
        
        # Check if key hand landmarks (fingertips, thumb tip) are within the head area
        # Use only fingertips and thumb tip for more precise detection
        key_landmark_indices = [4, 8, 12, 16, 20]  # Thumb tip, Index tip, Middle tip, Ring tip, Pinky tip
        
        landmarks_in_head_area = 0
        for idx in key_landmark_indices:
            if idx < len(hand_landmarks):
                landmark = hand_landmarks[idx]
                x, y = landmark[0], landmark[1]  # x, y are already normalized (0-1)
                
                # Check if landmark is within head area
                if face_left <= x <= face_right and face_top <= y <= face_bottom:
                    landmarks_in_head_area += 1
        
        # Require at least 1 key landmark in head area for detection
        return landmarks_in_head_area >= 1
    
    def _get_finger_info(self, hand_landmarks):
        """Get detailed finger information like the original hand tracker"""
        # Check if fingers are extended using the same logic as original
        thumb_up = self._is_finger_extended(hand_landmarks, 4, 3, 2)  # Thumb tip, MCP, IP
        index_up = self._is_finger_extended(hand_landmarks, 8, 5, 6)  # Index tip, MCP, PIP
        middle_up = self._is_finger_extended(hand_landmarks, 12, 9, 10)  # Middle tip, MCP, PIP
        ring_up = self._is_finger_extended(hand_landmarks, 16, 13, 14)  # Ring tip, MCP, PIP
        pinky_up = self._is_finger_extended(hand_landmarks, 20, 17, 18)  # Pinky tip, MCP, PIP
        
        # Count extended fingers
        fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        num_fingers = sum(fingers_up)
        
        return {
            "thumb_up": thumb_up,
            "index_up": index_up,
            "middle_up": middle_up,
            "ring_up": ring_up,
            "pinky_up": pinky_up,
            "num_fingers": num_fingers,
            "fingers_up": fingers_up
        }
    
    def _is_finger_extended(self, landmarks, tip_idx, mcp_idx, pip_idx):
        """Improved finger extension detection using multiple joint checks (same as original)"""
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]
        pip = landmarks[pip_idx]
        
        # Check if fingertip is above MCP joint (basic check)
        tip_above_mcp = tip[1] < mcp[1]
        
        # Check if fingertip is above PIP joint (more strict)
        tip_above_pip = tip[1] < pip[1]
        
        # Check distance between tip and MCP (finger should be reasonably extended)
        tip_mcp_distance = np.sqrt((tip[0] - mcp[0])**2 + (tip[1] - mcp[1])**2)
        pip_mcp_distance = np.sqrt((pip[0] - mcp[0])**2 + (pip[1] - mcp[1])**2)
        
        # Finger is extended if:
        # 1. Tip is above MCP joint
        # 2. Tip is above PIP joint  
        # 3. Tip is far enough from MCP (not just slightly above)
        distance_ratio = tip_mcp_distance / (pip_mcp_distance + 1e-6) if pip_mcp_distance > 0 else 0
        
        # Require tip to be significantly above joints and reasonably extended
        is_extended = (tip_above_mcp and tip_above_pip and 
                      distance_ratio > 1.2 and tip_mcp_distance > 0.05)
        
        return is_extended
    
    def _detect_gesture_from_fingers(self, finger_info):
        """Detect gesture from finger information (same logic as original)"""
        num_fingers = finger_info["num_fingers"]
        thumb_up = finger_info["thumb_up"]
        index_up = finger_info["index_up"]
        middle_up = finger_info["middle_up"]
        ring_up = finger_info["ring_up"]
        pinky_up = finger_info["pinky_up"]
        
        # Gesture recognition (same as original)
        if num_fingers == 0:
            return "fist"
        elif num_fingers == 5:
            return "open_hand"
        elif num_fingers == 1:
            # One finger raised - determine which finger
            if thumb_up:
                return "thumbs_up"
            elif index_up:
                return "pointing"
            elif middle_up:
                return "one_finger_middle"
            elif ring_up:
                return "one_finger_ring"
            elif pinky_up:
                return "one_finger_pinky"
            else:
                return "one_finger_raised"  # Generic one finger raised
        elif num_fingers == 2:
            # Two fingers raised
            if index_up and middle_up:
                return "peace_sign"
            else:
                return "two_fingers_raised"
        elif num_fingers == 3:
            return "three_fingers_raised"
        elif num_fingers == 4:
            return "four_fingers_raised"
        else:
            return "unknown"
    
    def _determine_primary_expression(self, eye_state, face_expressions, hand_gestures):
        """Determine primary expression with priority system (same logic as original)"""
        
        # Priority 1: Eyes closed expressions
        if eye_state == "closed":
            if face_expressions.get("smiling", False):
                return "eyes_closed_smiling"
            else:
                return "eyes_closed_neutral"
        
        # Priority 2: Special hand gestures
        special_gesture = hand_gestures.get("special_gesture")
        if special_gesture == "hand_touching_head":
            return "hand_touching_head"
        elif special_gesture == "both_hands_raised":
            return "both_hands_raised"
        elif special_gesture == "one_hand_raised":
            return "one_hand_raised"
        
        # Priority 3: Both hands thumbs up (highest priority for hand gestures)
        left_hand = hand_gestures.get("left_hand", "None")
        right_hand = hand_gestures.get("right_hand", "None")
        
        if left_hand == "thumbs_up" and right_hand == "thumbs_up":
            return "both_hands_thumbs_up"
        elif left_hand == "fist" and right_hand == "fist":
            return "both_hands_fists"
        
        # Priority 4: Individual hand gestures (thumbs up has higher priority than fist)
        elif left_hand == "thumbs_up" or right_hand == "thumbs_up":
            return "thumbs_up"
        elif left_hand == "pointing" or right_hand == "pointing":
            return "pointing"
        elif left_hand == "fist" or right_hand == "fist":
            return "fist"
        elif left_hand == "open_hand" or right_hand == "open_hand":
            return "open_hand"
        
        # Priority 5: Face expressions
        # Prioritize smiling over mouth_open when both are detected
        if face_expressions.get("smiling", False):
            return "looking_center_smiling"
        elif face_expressions.get("mouth_open", False):
            return "shocked"
        
        # Default
        return "looking_center"
    
    def _check_performance(self):
        """Monitor and report performance with adaptive optimization"""
        current_time = time.time()
        if current_time - self.last_performance_check > 5.0:  # Every 5 seconds
            avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
            
            # Adaptive frame skipping based on performance
            if avg_frame_time > 20.0:
                self.frame_skip = 3  # Process every 3rd frame
                print(f"Performance slow: {avg_frame_time:.1f}ms - Using frame skip 3")
            elif avg_frame_time > 15.0:
                self.frame_skip = 2  # Process every 2nd frame
                print(f"Performance warning: {avg_frame_time:.1f}ms - Using frame skip 2")
            elif avg_frame_time < 10.0:
                self.frame_skip = 1  # Process every frame
                if self.frame_skip != 1:
                    print(f"Good performance: {avg_frame_time:.1f}ms - Using frame skip 1")
            
            self.last_performance_check = current_time
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.frame_times:
            return {"avg_frame_time": 0, "fps": 0}
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
        
        return {
            "avg_frame_time_ms": avg_frame_time,
            "fps": fps,
            "performance_status": "good" if avg_frame_time < 15 else "needs_optimization"
        }

# Test the optimized detector
if __name__ == "__main__":
    detector = OptimizedDetector()
    cap = cv2.VideoCapture(0)
    
    print("Testing optimized detector performance...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = detector.process_frame(frame)
        
        # Display performance info
        cv2.putText(frame, f"Frame time: {result['frame_time_ms']:.1f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Expression: {result['expressions']['primary']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Optimized Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final performance report
    stats = detector.get_performance_stats()
    print(f"\nFinal Performance Stats:")
    print(f"Average frame time: {stats['avg_frame_time_ms']:.1f}ms")
    print(f"FPS: {stats['fps']:.1f}")
    print(f"Status: {stats['performance_status']}")
