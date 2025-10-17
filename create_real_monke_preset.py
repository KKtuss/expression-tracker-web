#!/usr/bin/env python3
"""
Script to create the MONKE preset with the actual monkey images from Base_Presets/Monkey/
"""

import json
import base64
import os
from datetime import datetime

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting {image_path}: {e}")
        return None

def create_real_monke_preset():
    """Create the MONKE preset with actual monkey images"""
    
    # Paths to the actual monkey images
    image_paths = {
        "hand_touching_head": "../Base_Presets/Monkey/monkey thinking.jpeg",
        "one_hand_raised": "../Base_Presets/Monkey/monkey idea.jpg", 
        "looking_center": "../Base_Presets/Monkey/monkey neutral.png"
    }
    
    # Create the preset data structure
    preset_data = {
        "_metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": 3,
            "description": "Monkey expressions preset - Thinking, Idea, and Neutral (Real Images)"
        }
    }
    
    # Convert images to base64 and add to preset
    for expression, image_path in image_paths.items():
        if os.path.exists(image_path):
            image_base64 = image_to_base64(image_path)
            if image_base64:
                preset_data[expression] = image_base64
                print(f"[SUCCESS] Added {expression} -> {os.path.basename(image_path)}")
            else:
                print(f"[ERROR] Failed to convert {image_path}")
        else:
            print(f"[ERROR] File not found: {image_path}")
    
    # Save the preset to JSON file
    preset_file = "presets/MONKE.json"
    with open(preset_file, 'w') as f:
        json.dump(preset_data, f, indent=2)
    
    print(f"\n[SUCCESS] Created MONKE preset with real monkey images")
    print(f"[FILE] Saved to: {preset_file}")
    print(f"[COUNT] Total images in preset: {len(preset_data) - 1}")  # Subtract metadata
    
    return preset_file

if __name__ == "__main__":
    create_real_monke_preset()
