#!/usr/bin/env python3
"""
Script to create the MONKE preset with the provided monkey images
"""

import json
import base64
import os
from datetime import datetime
from PIL import Image
from io import BytesIO

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting {image_path}: {e}")
        return None

def create_monke_preset():
    """Create the MONKE preset JSON file"""
    
    # Create the preset data structure
    preset_data = {
        "_metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": 3,
            "description": "Monkey expressions preset - Thinking, Idea, and Neutral"
        }
    }
    
    # Map the images to expressions
    image_mappings = {
        "hand_touching_head": "monkey thinking.jpeg",
        "one_hand_raised": "monkey idea.jpg", 
        "looking_center": "monkey neutral.png"
    }
    
    # Convert images to base64 and add to preset
    for expression, image_file in image_mappings.items():
        # Try different possible paths for the image files
        possible_paths = [
            image_file,  # Same directory
            f"../{image_file}",  # Parent directory
            f"images/{image_file}",  # Images subdirectory
            f"../images/{image_file}",  # Parent images subdirectory
        ]
        
        image_base64 = None
        for path in possible_paths:
            if os.path.exists(path):
                image_base64 = image_to_base64(path)
                if image_base64:
                    print(f"Successfully converted {path} to base64")
                    break
        
        if image_base64:
            preset_data[expression] = image_base64
            print(f"Added {expression} -> {image_file}")
        else:
            print(f"Warning: Could not find or convert {image_file}")
    
    # Save the preset to JSON file
    preset_file = "presets/MONKE.json"
    with open(preset_file, 'w') as f:
        json.dump(preset_data, f, indent=2)
    
    print(f"\nCreated preset: {preset_file}")
    print(f"Total images in preset: {len(preset_data) - 1}")  # Subtract metadata
    
    return preset_file

if __name__ == "__main__":
    create_monke_preset()
