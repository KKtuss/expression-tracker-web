import os
import json
import time
from typing import Dict, Optional, List
from PIL import Image
import base64
from io import BytesIO

class ImageManager:
    """
    Manages expression images and presets.
    This handles the same functionality as the desktop app's image management.
    """
    
    def __init__(self, images_storage: Dict):
        self.images = images_storage
        self.presets_dir = "presets"
        
        # Create presets directory if it doesn't exist
        if not os.path.exists(self.presets_dir):
            os.makedirs(self.presets_dir)
    
    def create_default_image(self, text: str, color: str) -> Image.Image:
        """Create a default colored image with text (same as desktop app)"""
        img = Image.new('RGB', (400, 300), color)
        return img
    
    def load_default_images(self):
        """Load default placeholder images (same as desktop app)"""
        default_images = {
            'eyes_open': self.create_default_image("Eyes Open!", "green"),
            'eyes_closed': self.create_default_image("Eyes Closed", "blue"),
            'looking_left': self.create_default_image("Looking Left", "orange"),
            'looking_right': self.create_default_image("Looking Right", "purple"),
            'looking_center': self.create_default_image("Looking Center", "yellow"),
            'neutral': self.create_default_image("Neutral", "gray")
        }
        
        for key, img in default_images.items():
            if key in self.images:  # Only set if key exists in images dict
                self.images[key] = img
    
    def clear_all_images(self):
        """Clear all loaded images (same as desktop app)"""
        for key in self.images.keys():
            self.images[key] = None
        print("All images cleared")
    
    def set_image_from_bytes(self, expression_type: str, image_data: bytes) -> bool:
        """Set an image for a specific expression type from raw bytes"""
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
    
    def set_image_from_base64(self, expression_type: str, base64_data: str) -> bool:
        """Set an image for a specific expression type from base64 string"""
        try:
            # Decode base64 to bytes
            image_data = base64.b64decode(base64_data)
            return self.set_image_from_bytes(expression_type, image_data)
        except Exception as e:
            print(f"Error setting image from base64 for {expression_type}: {e}")
            return False
    
    def get_image_base64(self, expression_type: str) -> Optional[str]:
        """Get an image as base64 string"""
        if expression_type in self.images and self.images[expression_type] is not None:
            return self._image_to_base64(self.images[expression_type])
        return None
    
    def get_all_images_status(self) -> Dict[str, bool]:
        """Get status of all images (whether they're set or not)"""
        return {key: image is not None for key, image in self.images.items()}
    
    def save_preset(self, preset_name: str) -> bool:
        """Save current image configuration as a preset (same as desktop app)"""
        try:
            # Validate preset name (same as desktop app)
            if not preset_name.replace("_", "").replace("-", "").isalnum():
                return False
            
            preset_path = os.path.join(self.presets_dir, f"{preset_name}.json")
            
            # Collect current image paths (same as desktop app logic)
            preset_data = {}
            for key, image_data in self.images.items():
                # For web version, we store base64 data instead of file paths
                if image_data is not None:
                    preset_data[key] = self._image_to_base64(image_data)
            
            # Count actual images (before adding metadata)
            image_count = len(preset_data)
            
            # Add metadata (same as desktop app)
            preset_data["_metadata"] = {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_images": image_count
            }
            
            with open(preset_path, 'w') as f:
                json.dump(preset_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save preset: {e}")
            return False
    
    def load_preset(self, preset_name: str) -> bool:
        """Load a preset configuration (same as desktop app)"""
        try:
            preset_path = os.path.join(self.presets_dir, f"{preset_name}.json")
            
            with open(preset_path, 'r') as f:
                preset_data = json.load(f)
            
            # Clear current images
            self.clear_all_images()
            
            # Load preset images
            loaded_count = 0
            for key, image_data in preset_data.items():
                if key.startswith("_"):  # Skip metadata
                    continue
                if self.set_image_from_base64(key, image_data):
                    loaded_count += 1
            
            print(f"Loaded preset with {loaded_count} images!")
            return True
        except Exception as e:
            print(f"Failed to load preset: {e}")
            return False
    
    def delete_preset(self, preset_name: str) -> bool:
        """Delete a preset (same as desktop app)"""
        try:
            preset_path = os.path.join(self.presets_dir, f"{preset_name}.json")
            os.remove(preset_path)
            return True
        except Exception as e:
            print(f"Failed to delete preset: {e}")
            return False
    
    def list_presets(self) -> List[Dict]:
        """List all available presets with metadata"""
        preset_files = [f for f in os.listdir(self.presets_dir) if f.endswith('.json')]
        presets = []
        
        for preset_file in preset_files:
            preset_name = preset_file[:-5]  # Remove .json extension
            preset_path = os.path.join(self.presets_dir, preset_file)
            
            try:
                with open(preset_path, 'r') as f:
                    preset_data = json.load(f)
                metadata = preset_data.get("_metadata", {})
                
                presets.append({
                    "name": preset_name,
                    "created": metadata.get("created", "Unknown"),
                    "total_images": metadata.get("total_images", 0)
                })
            except:
                presets.append({
                    "name": preset_name,
                    "created": "Unknown",
                    "total_images": 0
                })
        
        return sorted(presets, key=lambda x: x["name"])
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
