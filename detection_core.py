import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
from PIL import Image

from facial_landmarks import FacialLandmarks
from gaze_tracker import GazeTracker
from hand_tracker import HandTracker
from optimized_detector import OptimizedDetector

class ExpressionDetector:
    """
    Main detection orchestrator that combines all detection modules.
    This is the exact logic from the desktop app's SimpleImageViewer class.
    """
    
    def __init__(self):
        # Initialize optimized detector for better performance
        self.optimized_detector = OptimizedDetector()
        
        # Keep legacy detectors as fallback (but don't use them by default)
        self.landmarks_detector = FacialLandmarks()
        self.gaze_tracker = GazeTracker()
        self.hand_tracker = HandTracker()
        
        # Performance mode flag
        self.use_optimized_detector = True
        
        # Image storage - same as desktop app
        self.images = {
            'eyes_open': None,
            'eyes_closed': None,
            'looking_left': None,
            'looking_right': None,
            'looking_center': None,
            'smiling': None,
            'shocked': None,  # Mouth open without smiling
            # Combined expressions
            'eyes_closed_smiling': None,
            'eyes_open_smiling': None,
            'looking_left_smiling': None,
            'looking_right_smiling': None,
            'looking_center_smiling': None,
            'eyes_closed_neutral': None,
            # Hand gestures
            'thumbs_up': None,
            'thumbs_down': None,
            'open_hand': None,
            'fist': None,
            'pointing': None,
            # One finger raised options
            'one_finger_raised': None,
            'one_finger_middle': None,
            'one_finger_ring': None,
            'one_finger_pinky': None,
            # Special hand positions
            'one_hand_raised': None,
            'both_hands_raised': None,
            'hand_touching_head': None,
            'both_hands_thumbs_up': None,
            'both_hands_fists': None
        }
        
        # Current state (same as desktop app)
        self.current_expression = None
        self.last_valid_expression = None
    
    def process_frame_fast(self, frame: np.ndarray) -> Dict:
        """Fast processing using optimized detector"""
        if not self.use_optimized_detector:
            return self.process_frame_legacy(frame)
        
        try:
            # Use optimized detector
            result = self.optimized_detector.process_frame(frame)
            
            # Get primary expression
            primary_expression = result["expressions"]["primary"]
            
            # Check if we have an image for this expression
            if primary_expression in self.images and self.images[primary_expression] is not None:
                self.current_expression = primary_expression
                self.last_valid_expression = primary_expression
            
            # Create debug info
            debug_info = {
                "detected_expression": primary_expression,
                "frame_time_ms": result["frame_time_ms"],
                "performance": result["debug"]["performance"]
            }
            
            return {
                "expression": self.current_expression,
                "debug": debug_info,
                "frame": frame  # Return original frame
            }
            
        except Exception as e:
            print(f"Optimized detector error: {e}")
            # Fallback to legacy processing
            return self.process_frame_legacy(frame)
    
    def process_frame_legacy(self, frame: np.ndarray) -> Dict:
        """
        Legacy processing method - original desktop app logic
        """
        return self.process_frame(frame)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and return detection results.
        This is the exact logic from the desktop app's detection_loop method.
        """
        # Mirror the frame (same as desktop app)
        frame = cv2.flip(frame, 1)
        
        # Get landmark data (same as desktop app)
        landmark_data = self.landmarks_detector.get_landmark_data(frame)
        
        if landmark_data["faces_detected"] > 0:
            landmark = landmark_data["landmarks"][0]
            face_coords = landmark["face"]
            eyes = landmark["eyes"]
            eye_analysis = landmark["eye_analysis"]
            
            # Analyze gaze direction (same as desktop app)
            gaze_result = self.gaze_tracker.analyze_gaze_direction(frame, eyes, face_coords)
            
            # Analyze smile using facial landmarks (same as desktop app)
            smile_result = self.landmarks_detector.detect_smile_simple(frame, face_coords)
            
            # Analyze mouth opening (same as desktop app)
            mouth_result = self.landmarks_detector.detect_mouth_opening(frame, face_coords)
            
            # Analyze hand gestures (same as desktop app)
            hand_result = self.hand_tracker.get_hand_gestures(frame, face_coords)
            hands_data = self.hand_tracker.detect_hands(frame)
            
            # Determine current expression (same as desktop app)
            new_expression = self.determine_expression(
                eye_analysis, gaze_result, smile_result, mouth_result, hand_result
            )
            
            # Update expression state (same as desktop app)
            if new_expression != self.current_expression:
                if new_expression is not None:
                    self.current_expression = new_expression
                    self.last_valid_expression = new_expression
                elif self.last_valid_expression is not None:
                    self.current_expression = self.last_valid_expression
            
            # Draw debug overlays on frame (same as desktop app)
            frame = self.landmarks_detector.draw_landmarks(frame, landmark_data)
            frame = self.hand_tracker.draw_hands(frame, hands_data)
            
            # Add expression text to frame (same as desktop app)
            display_expression = self.current_expression if self.current_expression else "None (no image set)"
            cv2.putText(frame, f"Expression: {display_expression}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Smile: {smile_result.get('is_smiling', False)} (count: {smile_result.get('smile_count', 0)})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Mouth Open: {mouth_result.get('is_mouth_open', False)} (ratio: {mouth_result.get('mouth_ratio', 0):.3f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Add gaze debugging info (same as desktop app)
            raw_gaze = gaze_result.get('raw_gaze', {})
            cv2.putText(frame, f"Gaze: {raw_gaze.get('horizontal', 0.5):.2f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Add detailed hand tracking info (same as desktop app)
            left_hand = hand_result.get('left_hand', 'None')
            right_hand = hand_result.get('right_hand', 'None')
            special_gesture = hand_result.get('special_gesture', 'None')
            
            hand_count = len(hands_data)
            cv2.putText(frame, f"Hands Detected: {hand_count}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Show individual hand data (same as desktop app)
            for i, hand in enumerate(hands_data):
                landmarks = hand['landmarks']
                wrist_y = landmarks[0][1]  # WRIST = 0, get Y coordinate
                hand_label = hand['label']
                gesture = hand['gesture']
                
                cv2.putText(frame, f"{hand_label}: {gesture} (Y: {wrist_y:.3f})", 
                           (10, 180 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            
            # Show special gesture info (same as desktop app)
            cv2.putText(frame, f"Special Gesture: {special_gesture}", 
                       (10, 240 + len(hands_data)*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 255), 2)
            
            # Prepare response with JSON-serializable data
            def make_json_safe(obj):
                """Convert numpy types and other non-JSON types to JSON-safe types"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, tuple):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_safe(item) for item in obj]
                else:
                    return obj
            
            # Make debug data JSON-safe
            safe_eye_analysis = make_json_safe(eye_analysis)
            safe_gaze_result = make_json_safe(gaze_result)
            safe_smile_result = make_json_safe(smile_result)
            safe_mouth_result = make_json_safe(mouth_result)
            safe_hand_result = make_json_safe(hand_result)
            
            safe_hands_data = []
            for hand in hands_data:
                safe_hands_data.append({
                    "label": hand["label"],
                    "gesture": hand["gesture"],
                    "landmarks_count": len(hand["landmarks"])
                })
            
            safe_face_coords = make_json_safe(face_coords)
            
            # Get the detected expression (regardless of image availability)
            detected_expression = self._detect_expression(eye_analysis, gaze_result, smile_result, mouth_result, hand_result)
            
            return {
                "success": True,
                "expression": self.current_expression,
                "debug": {
                    "face_detected": True,
                    "detected_expression": detected_expression,
                    "face_coords": safe_face_coords,
                    "eye_analysis": safe_eye_analysis,
                    "gaze_result": safe_gaze_result,
                    "smile_result": safe_smile_result,
                    "mouth_result": safe_mouth_result,
                    "hand_result": safe_hand_result,
                    "hands_data": safe_hands_data
                },
                "frame_with_overlay": self._frame_to_base64(frame)
            }
        else:
            # No face detected
            return {
                "success": False,
                "expression": None,
                "debug": {
                    "face_detected": False,
                    "message": "No face detected"
                },
                "frame_with_overlay": self._frame_to_base64(frame)
            }
    
    def determine_expression(self, eye_analysis, gaze_result, smile_result, mouth_result, hand_result):
        """
        Determine the current facial expression based on detection results.
        This is the EXACT logic from the desktop app's determine_expression method.
        """
        # First, determine what expression is detected (regardless of image availability)
        detected_expression = self._detect_expression(eye_analysis, gaze_result, smile_result, mouth_result, hand_result)
        
        # Then, only return it if an image is assigned (for image display)
        if detected_expression and self.images.get(detected_expression) is not None:
            return detected_expression
        else:
            return None  # No image assigned, but we still track the detected expression
    
    def _detect_expression(self, eye_analysis, gaze_result, smile_result, mouth_result, hand_result):
        """
        Detect the expression without checking for image availability.
        This is used for debug info and testing purposes.
        """
        # Get smile status
        is_smiling = smile_result.get("is_smiling", False)
        
        # Get mouth opening status
        is_mouth_open = mouth_result.get("is_mouth_open", False)
        
        # Check if eyes are closed
        eyes_closed = gaze_result.get("is_eyes_closed", False)
        
        # Check gaze direction
        gaze_direction = gaze_result.get("direction", "center")
        
        # Check if eyes are open
        eyes_open = eye_analysis.get("both_eyes_open", False)
        
        # Get hand gestures (prioritize hand gestures over facial expressions)
        left_hand_gesture = hand_result.get("left_hand")
        right_hand_gesture = hand_result.get("right_hand")
        special_gesture = hand_result.get("special_gesture")
        
        # Prioritize special gestures first (regardless of image availability)
        if special_gesture and special_gesture != "None":
            return special_gesture
        
        # Check for both hands thumbs up (highest priority after special gestures)
        if (left_hand_gesture == "thumbs_up" and right_hand_gesture == "thumbs_up"):
            return "both_hands_thumbs_up"
        
        # Check for both hands fists
        if (left_hand_gesture == "fist" and right_hand_gesture == "fist"):
            return "both_hands_fists"
        
        # Then check individual hand gestures (regardless of image availability)
        if left_hand_gesture and left_hand_gesture != "unknown" and left_hand_gesture != "None":
            return left_hand_gesture
        elif right_hand_gesture and right_hand_gesture != "unknown" and right_hand_gesture != "None":
            return right_hand_gesture
        
        # Create combined expressions (facial expressions only if no hand gestures)
        if eyes_closed:
            if is_smiling:
                return "eyes_closed_smiling"
            else:
                return "eyes_closed_neutral"
        elif is_smiling:
            if gaze_direction == "left":
                return "looking_left_smiling"
            elif gaze_direction == "right":
                return "looking_right_smiling"
            elif gaze_direction == "center":
                return "looking_center_smiling"
            elif eyes_open:
                return "eyes_open_smiling"
            else:
                return "smiling"
        elif is_mouth_open and not is_smiling:
            # Mouth open without smiling = shocked
            return "shocked"
        elif eyes_open:
            if gaze_direction == "left":
                return "looking_left"
            elif gaze_direction == "right":
                return "looking_right"
            elif gaze_direction == "center":
                return "looking_center"
        
        # Default to looking_center if no specific expression detected
        return "looking_center"
    
    def set_image(self, expression_type: str, image_data: bytes) -> bool:
        """
        Set an image for a specific expression type.
        image_data should be the raw bytes of an image file.
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_data))
            # Resize to fit display (same as desktop app)
            image = image.resize((400, 300), Image.Resampling.LANCZOS)
            self.images[expression_type] = image
            return True
        except Exception as e:
            print(f"Error setting image for {expression_type}: {e}")
            return False
    
    def get_current_image(self) -> Optional[str]:
        """
        Get the current expression's image as base64 string.
        Returns None if no image is set for current expression.
        """
        if self.current_expression and self.images.get(self.current_expression) is not None:
            return self._image_to_base64(self.images[self.current_expression])
        return None
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame to base64 string"""
        try:
            # Create a copy to avoid modifying the original
            frame_copy = frame.copy()
            
            # Ensure frame is in BGR format (3 channels) for JPEG compatibility
            if len(frame_copy.shape) == 3 and frame_copy.shape[2] == 4:
                # Convert RGBA to BGR
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
                print(f"DEBUG: Converted RGBA to BGR, new shape: {frame_copy.shape}")
            elif len(frame_copy.shape) == 3 and frame_copy.shape[2] == 3:
                # Already BGR, no conversion needed
                pass
            elif len(frame_copy.shape) == 2:
                # Convert grayscale to BGR
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
                print(f"DEBUG: Converted grayscale to BGR, new shape: {frame_copy.shape}")
            else:
                print(f"DEBUG: Unexpected frame shape: {frame_copy.shape}")
                # Force conversion to BGR
                if len(frame_copy.shape) == 3:
                    frame_copy = frame_copy[:, :, :3]  # Take only first 3 channels
                else:
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
            
            # Ensure the frame is uint8
            if frame_copy.dtype != np.uint8:
                frame_copy = frame_copy.astype(np.uint8)
            
            # Double-check the frame shape before encoding
            if len(frame_copy.shape) != 3 or frame_copy.shape[2] != 3:
                print(f"DEBUG: Frame still not BGR after conversion: {frame_copy.shape}")
                # Force to BGR by taking first 3 channels or converting
                if len(frame_copy.shape) == 3 and frame_copy.shape[2] >= 3:
                    frame_copy = frame_copy[:, :, :3]
                else:
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
            
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame_copy)
            if not success:
                print(f"DEBUG: JPEG encoding failed for frame shape: {frame_copy.shape}, dtype: {frame_copy.dtype}")
                # Try one more time with a clean BGR conversion
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2BGR)  # This forces a clean copy
                success, buffer = cv2.imencode('.jpg', frame_copy)
                if not success:
                    raise ValueError("Failed to encode frame as JPEG after multiple attempts")
            
            frame_bytes = buffer.tobytes()
            return base64.b64encode(frame_bytes).decode('utf-8')
            
        except Exception as e:
            print(f"DEBUG: Error in _frame_to_base64: {e}")
            print(f"DEBUG: Frame shape: {frame.shape}, dtype: {frame.dtype}")
            # Return a minimal black frame as fallback
            try:
                fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, fallback_buffer = cv2.imencode('.jpg', fallback_frame)
                fallback_bytes = fallback_buffer.tobytes()
                return base64.b64encode(fallback_bytes).decode('utf-8')
            except:
                # Ultimate fallback - return empty string
                return ""
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        try:
            buffer = BytesIO()
            
            # Convert RGBA to RGB if necessary (for PNG images with transparency)
            if image.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            print(f"DEBUG: Error in _image_to_base64: {e}")
            print(f"DEBUG: Image mode: {image.mode if hasattr(image, 'mode') else 'unknown'}")
            # Return a minimal black image as fallback
            try:
                fallback_image = Image.new('RGB', (100, 100), (0, 0, 0))
                fallback_buffer = BytesIO()
                fallback_image.save(fallback_buffer, format='JPEG')
                fallback_bytes = fallback_buffer.getvalue()
                return base64.b64encode(fallback_bytes).decode('utf-8')
            except:
                return ""
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'hand_tracker'):
            self.hand_tracker.cleanup()
