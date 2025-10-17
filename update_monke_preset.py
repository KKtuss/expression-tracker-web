#!/usr/bin/env python3
"""
Script to update the MONKE preset with actual image base64 data
Usage: python update_monke_preset.py
"""

import json
import base64
import os

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting {image_path}: {e}")
        return None

def update_monke_preset():
    """Update the MONKE preset with actual image data"""
    
    preset_file = "presets/MONKE.json"
    
    # Load existing preset
    try:
        with open(preset_file, 'r') as f:
            preset_data = json.load(f)
    except Exception as e:
        print(f"Error loading preset: {e}")
        return
    
    print("MONKE Preset Updater")
    print("===================")
    print("\nPlease provide the paths to your monkey images:")
    print("1. Monkey thinking (for hand_touching_head)")
    print("2. Monkey idea (for one_hand_raised)")
    print("3. Monkey neutral (for looking_center)")
    print("\nOr press Enter to skip an image.\n")
    
    # Get image paths from user
    image_mappings = {
        "hand_touching_head": input("Path to monkey thinking image (hand_touching_head): ").strip(),
        "one_hand_raised": input("Path to monkey idea image (one_hand_raised): ").strip(),
        "looking_center": input("Path to monkey neutral image (looking_center): ").strip()
    }
    
    # Convert images to base64
    updated_count = 0
    for expression, image_path in image_mappings.items():
        if image_path and os.path.exists(image_path):
            image_base64 = image_to_base64(image_path)
            if image_base64:
                preset_data[expression] = image_base64
                print(f"✅ Updated {expression} with {image_path}")
                updated_count += 1
            else:
                print(f"❌ Failed to convert {image_path}")
        elif image_path:
            print(f"❌ File not found: {image_path}")
        else:
            print(f"⏭️  Skipped {expression}")
    
    # Save updated preset
    try:
        with open(preset_file, 'w') as f:
            json.dump(preset_data, f, indent=2)
        print(f"\n✅ Preset updated successfully!")
        print(f"📊 Total images in preset: {updated_count}")
        print(f"📁 Preset saved to: {preset_file}")
    except Exception as e:
        print(f"❌ Error saving preset: {e}")

if __name__ == "__main__":
    update_monke_preset()
