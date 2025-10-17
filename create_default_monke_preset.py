#!/usr/bin/env python3
"""
Create a default MONKE preset with placeholder images
"""

import json
import base64
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import os

def create_monkey_placeholder_image(text, size=(400, 300), bg_color=(139, 69, 19)):
    """Create a placeholder image with monkey-themed colors"""
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

def create_monke_preset():
    """Create the MONKE preset with placeholder images"""
    
    # Create placeholder images
    placeholders = {
        "hand_touching_head": create_monkey_placeholder_image("MONKEY\nTHINKING", bg_color=(139, 69, 19)),
        "one_hand_raised": create_monkey_placeholder_image("MONKEY\nIDEA!", bg_color=(160, 82, 45)),
        "looking_center": create_monkey_placeholder_image("MONKEY\nNEUTRAL", bg_color=(101, 67, 33))
    }
    
    # Create the preset data structure
    preset_data = {
        "_metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": 3,
            "description": "Monkey expressions preset - Thinking, Idea, and Neutral"
        }
    }
    
    # Add the placeholder images
    preset_data.update(placeholders)
    
    # Save the preset to JSON file
    preset_file = "presets/MONKE.json"
    with open(preset_file, 'w') as f:
        json.dump(preset_data, f, indent=2)
    
    print(f"[SUCCESS] Created MONKE preset with placeholder images")
    print(f"[FILE] Saved to: {preset_file}")
    print(f"[COUNT] Total images: {len(placeholders)}")
    
    return preset_file

if __name__ == "__main__":
    create_monke_preset()
