import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import io
import json
import os
import re
from datetime import datetime
import numpy as np
import math
import traceback

# Try to import replicate if available
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    print("Replicate package not installed. Background removal will use local method.")
    REPLICATE_AVAILABLE = False

# Webhook URL - Google Apps Script Web App URL
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzOQ7SaTtIXRubvSNXNY53pphacVmJg_XKV5sIyOgxjpDykiWsAHN7ecKFHcygGFrYi/exec"

# FIXED WIDTH FOR ALL IMAGES
FIXED_WIDTH = 1200

def download_korean_font():
    """Download Korean font for text rendering - More robust version"""
    try:
        font_path = '/tmp/NanumMyeongjo.ttf'
        
        # Check if already exists and valid
        if os.path.exists(font_path):
            try:
                # Try to load it to verify it's valid
                test_font = ImageFont.truetype(font_path, 20)
                print("Korean font already exists and is valid")
                return True
            except:
                print("Korean font exists but is corrupted, re-downloading...")
                os.remove(font_path)
        
        # List of URLs to try
        font_urls = [
            'https://github.com/naver/nanumfont/raw/master/fonts/NanumMyeongjo/NanumMyeongjo.ttf',
            'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumMyeongjo/NanumMyeongjo.ttf',
            'https://raw.githubusercontent.com/naver/nanumfont/master/fonts/NanumMyeongjo/NanumMyeongjo.ttf'
        ]
        
        for url in font_urls:
            try:
                print(f"Downloading Korean font from: {url}")
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200 and len(response.content) > 100000:  # Font should be > 100KB
                    with open(font_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Verify the font works
                    test_font = ImageFont.truetype(font_path, 20)
                    print(f"Korean font downloaded successfully from {url}")
                    return True
            except Exception as e:
                print(f"Failed to download from {url}: {str(e)}")
                continue
        
        print("Failed to download Korean font from all sources")
        return False
        
    except Exception as e:
        print(f"Error in font download process: {str(e)}")
        return False

def ultra_safe_string_encode(text):
    """V135 ULTRA SAFE: Enhanced string encoding with perfect Korean character support"""
    if not text:
        return ""
    
    try:
        # 1단계: 문자열로 변환
        if isinstance(text, bytes):
            # V135 ENHANCED: Better bytes handling with Korean support
            try:
                text = text.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = text.decode('utf-8', errors='replace')
                except:
                    text = text.decode('unicode_escape', errors='ignore')
        else:
            text = str(text)
        
        # V135 IMPROVED: Direct UTF-8 validation and preservation
        try:
            # Test UTF-8 encoding/decoding to ensure validity
            text_bytes = text.encode('utf-8')
            text = text_bytes.decode('utf-8')
            
            # V135 ENHANCED: Test JSON serialization with Korean characters
            json.dumps(text, ensure_ascii=False)
            
            print(f"V135 SUCCESS: UTF-8 Korean text preserved: {text[:50]}...")
            return text.strip()
            
        except (UnicodeEncodeError, UnicodeDecodeError, json.JSONDecodeError):
            print("V135 INFO: UTF-8 validation failed, attempting repair...")
            
            # V135 ENHANCED: Repair strategy that preserves Korean characters
            try:
                # Use 'replace' to handle problematic characters while preserving Korean
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                
                # Test JSON serialization again
                json.dumps(text, ensure_ascii=False)
                print(f"V135 REPAIRED: Korean text repaired: {text[:50]}...")
                return text.strip()
                
            except Exception as repair_error:
                print(f"V135 WARNING: Text repair failed: {repair_error}")
                
                # V135 FALLBACK: Use unicode_escape for problematic text
                try:
                    text = text.encode('unicode_escape').decode('ascii')
                    return text.strip()
                except:
                    return "한국어 텍스트 처리 오류"
        
    except Exception as e:
        print(f"V135 ERROR: Text encoding completely failed: {e}")
        return "텍스트 인코딩 실패"

def clean_claude_text(text):
    """V135 ENHANCED: Clean text for safe JSON encoding while perfectly preserving Korean characters"""
    if not text:
        return ""
    
    # V135 CRITICAL: Use enhanced ultra safe encoding first
    text = ultra_safe_string_encode(text)
    
    # Replace escape sequences (backslash + character)
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\\\', '\\')  # Handle double backslashes
    
    # Replace actual control characters
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    
    # Fix quotes if they're escaped
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    
    # Remove markdown symbols
    for char in ['#', '*', '_', '`', '[', ']', '(', ')']:
        text = text.replace(char, '')
    
    # Collapse multiple spaces
    text = ' '.join(text.split())
    
    print(f"V135 Cleaned Korean text preview: {text[:100]}...")
    return text

def get_text_dimensions(draw, text, font):
    """Get text dimensions compatible with all PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def extract_ring_with_replicate(img):
    """V135 ENHANCED: Extract ring from background using Replicate API with improved settings"""
    try:
        if not REPLICATE_AVAILABLE:
            print("Replicate not available, using local fallback")
            return extract_ring_local_fallback(img)
            
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print("Replicate API token not found, using local fallback")
            return extract_ring_local_fallback(img)
            
        print("V135: Starting enhanced Replicate background removal...")
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # V135 ENHANCED: Use improved model with better settings for jewelry
        output = replicate.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            input={
                "image": f"data:image/png;base64,{img_base64}",
                "model": "u2net",  # V135: Better for object detection
                "alpha_matting": True,  # V135: Enable alpha matting for smoother edges
                "alpha_matting_foreground_threshold": 270,
                "alpha_matting_background_threshold": 10,
                "alpha_matting_erode_size": 10
            }
        )
        
        response = requests.get(output)
        result_img = Image.open(BytesIO(response.content))
        
        if result_img.mode != 'RGBA':
            result_img = result_img.convert('RGBA')
        
        print("V135: Enhanced Replicate background removal completed successfully")
        return result_img
        
    except Exception as e:
        print(f"V135: Error with Replicate API: {e}")
        print("V135: Falling back to local method...")
        return extract_ring_local_fallback(img)

def extract_ring_local_fallback(img):
    """V135 ENHANCED: Local fallback method for background removal with improved algorithm"""
    print("V135: Using enhanced local fallback for background removal")
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    width, height = img.size
    img_array = np.array(img)
    
    # V135 IMPROVED: Better corner detection
    corner_size = 15  # Increased from 10
    corners = []
    corners.extend(img_array[:corner_size, :corner_size].reshape(-1, 4))
    corners.extend(img_array[:corner_size, -corner_size:].reshape(-1, 4))
    corners.extend(img_array[-corner_size:, :corner_size].reshape(-1, 4))
    corners.extend(img_array[-corner_size:, -corner_size:].reshape(-1, 4))
    
    corners_array = np.array(corners)
    bg_color = np.median(corners_array, axis=0)[:3]
    
    # V135 ENHANCED: More sophisticated color distance calculation
    color_distance = np.sqrt(
        (img_array[:,:,0] - bg_color[0])**2 +
        (img_array[:,:,1] - bg_color[1])**2 +
        (img_array[:,:,2] - bg_color[2])**2
    )
    
    # V135 IMPROVED: Better threshold calculation
    threshold = np.percentile(color_distance, 25)  # More sensitive
    mask = color_distance > threshold
    
    mask = mask.astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, 'L')
    
    # V135 ENHANCED: Better morphological operations
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))  # Increased from 3
    mask_img = mask_img.filter(ImageFilter.MinFilter(5))  # Increased from 3
    mask_img = mask_img.filter(ImageFilter.SMOOTH_MORE)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))  # Increased from 1
    
    result = img.copy()
    result.putalpha(mask_img)
    
    return result

def apply_metal_color_filter(img, color_multipliers):
    """V135 ENHANCED: Apply metal color filter with improved color accuracy"""
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
    else:
        img = img.convert('RGBA')
        r, g, b, a = img.split()
    
    # V135 IMPROVED: Better color multiplication algorithm
    r = r.point(lambda x: min(255, max(0, int(x * color_multipliers[0] * 1.1))))  # Slight boost
    g = g.point(lambda x: min(255, max(0, int(x * color_multipliers[1] * 1.1))))
    b = b.point(lambda x: min(255, max(0, int(x * color_multipliers[2] * 1.1))))
    
    return Image.merge('RGBA', (r, g, b, a))

def create_color_options_section(width=FIXED_WIDTH, ring_image=None):
    """V135 ENHANCED: Create COLOR section with improved ring processing and no inner filling"""
    print("V135: === CREATING ENHANCED COLOR SECTION WITH REAL WEDDING RING ===")
    
    section_height = 1000
    section_img = Image.new('RGB', (width, section_height), '#F8F8F8')
    draw = ImageDraw.Draw(section_img)
    
    # Load fonts
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    label_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 48)
                label_font = ImageFont.truetype(font_path, 28)  # V135: Increased font size
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Title
    title = "COLOR"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 80), title, font=title_font, fill=(51, 51, 51))
    
    # V135 ENHANCED: Better color definitions with improved visual quality
    colors = [
        ("yellow", (255, 215, 0)),      # Brighter gold
        ("rose", (232, 180, 184)),      # More accurate rose gold
        ("white", (245, 245, 245)),     # Purer white
        ("antique", (212, 175, 55))     # Vintage gold
    ]
    
    # Layout settings
    container_size = 300
    h_spacing = 100
    v_spacing = 400
    
    grid_width = 2 * container_size + h_spacing
    start_x = (width - grid_width) // 2
    start_y = 200
    
    # V135 ENHANCED: Better ring processing with no inner filling
    processed_ring = None
    if ring_image:
        try:
            print("V135 ENHANCED: Processing actual ring image with improved background removal...")
            print(f"V135: Input ring image size: {ring_image.size}, mode: {ring_image.mode}")
            
            # V135 ENHANCED: Better background removal
            processed_ring = extract_ring_with_replicate(ring_image)
            print(f"V135 SUCCESS: Enhanced ring extraction completed, size: {processed_ring.size}")
            
            # Additional verification
            if processed_ring and processed_ring.size[0] > 0 and processed_ring.size[1] > 0:
                print("V135 VERIFIED: Ring image is valid and ready for enhanced color application")
            else:
                print("V135 ERROR: Ring extraction returned invalid image")
                processed_ring = None
                
        except Exception as e:
            print(f"V135 ERROR: Ring processing failed: {e}")
            print(f"V135 TRACEBACK: {traceback.format_exc()}")
            processed_ring = None
    else:
        print("V135 WARNING: No ring image provided to color section")
    
    # Create color variants
    for i, (name, color_rgb) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (container_size + h_spacing)
        y = start_y + row * v_spacing
        
        # White container background
        container = Image.new('RGBA', (container_size, container_size), (255, 255, 255, 255))
        container_draw = ImageDraw.Draw(container)
        
        # V135 ENHANCED: Use actual ring image with improved processing
        if processed_ring and processed_ring.size[0] > 0:
            try:
                print(f"V135: Applying enhanced {name} color to actual ring image...")
                
                # V135 IMPROVED: Better resizing with aspect ratio preservation
                ring_width, ring_height = processed_ring.size
                max_size = container_size - 60  # Leave margin
                
                # Calculate aspect-preserving size
                aspect_ratio = ring_width / ring_height
                if aspect_ratio > 1:  # Wider than tall
                    new_width = max_size
                    new_height = int(max_size / aspect_ratio)
                else:  # Taller than wide
                    new_height = max_size
                    new_width = int(max_size * aspect_ratio)
                
                # V135 ENHANCED: High-quality resize with better resampling
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                ring_resized = processed_ring.resize((new_width, new_height), resample_filter)
                
                # V135 ENHANCED: Better color filter application
                color_multipliers = [
                    color_rgb[0] / 255.0,
                    color_rgb[1] / 255.0,
                    color_rgb[2] / 255.0
                ]
                colored_ring = apply_metal_color_filter(ring_resized, color_multipliers)
                
                # Center the ring in the container
                ring_x = (container_size - new_width) // 2
                ring_y = (container_size - new_height) // 2
                
                # V135 ENHANCED: Better alpha compositing
                if colored_ring.mode == 'RGBA':
                    container.paste(colored_ring, (ring_x, ring_y), colored_ring)
                else:
                    container.paste(colored_ring, (ring_x, ring_y))
                
                print(f"V135 SUCCESS: Enhanced {name} color applied to actual ring image")
                
            except Exception as e:
                print(f"V135 ERROR: Failed to apply {name} color to ring: {e}")
                print(f"V135 TRACEBACK: {traceback.format_exc()}")
                # V135 ENHANCED: Better fallback rings
                draw_enhanced_fallback_rings(container_draw, container_size, color_rgb)
        else:
            print(f"V135 FALLBACK: Using enhanced ring graphics for {name}")
            # V135 ENHANCED: Better fallback rings
            draw_enhanced_fallback_rings(container_draw, container_size, color_rgb)
        
        # V135 ENHANCED: Better shadow effect
        shadow_img = Image.new('RGBA', (container_size + 15, container_size + 15), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        shadow_draw.rectangle([8, 8, container_size + 8, container_size + 8], 
                            fill=(0, 0, 0, 25))
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=6))
        
        # Paste shadow first
        section_img.paste(shadow_img, (x - 8, y - 8), shadow_img)
        
        # Paste container
        section_img.paste(container, (x, y))
        
        # V135 ENHANCED: Better label positioning and size
        label_width, _ = get_text_dimensions(draw, name, label_font)
        draw.text((x + container_size//2 - label_width//2, y + container_size + 35), 
                 name, font=label_font, fill=(80, 80, 80))
    
    print("V135: Enhanced color section creation completed")
    return section_img

def draw_enhanced_fallback_rings(container_draw, container_size, color_rgb):
    """V135 ENHANCED: Draw improved fallback ring graphics with no inner filling"""
    print("V135: Drawing enhanced fallback ring graphics with outline only")
    
    # V135 ENHANCED: Better ring positioning and sizing
    left_ring_center_x = container_size // 2 - 35
    left_ring_center_y = container_size // 2 - 10
    left_ring_radius = 75
    left_ring_thickness = 12  # Thinner for outline effect
    
    # V135 IMPROVED: Draw left ring OUTLINE ONLY
    container_draw.ellipse([
        left_ring_center_x - left_ring_radius,
        left_ring_center_y - left_ring_radius,
        left_ring_center_x + left_ring_radius,
        left_ring_center_y + left_ring_radius
    ], outline=color_rgb, width=left_ring_thickness)
    
    # V135 IMPROVED: Draw inner ring OUTLINE ONLY
    inner_radius = left_ring_radius - left_ring_thickness - 5
    container_draw.ellipse([
        left_ring_center_x - inner_radius,
        left_ring_center_y - inner_radius,
        left_ring_center_x + inner_radius,
        left_ring_center_y + inner_radius
    ], outline=color_rgb, width=8)
    
    # V135 ENHANCED: Better right ring positioning
    right_ring_center_x = container_size // 2 + 45
    right_ring_center_y = container_size // 2 + 25
    right_ring_radius = 60
    right_ring_thickness = 10
    
    # V135 IMPROVED: Draw right ring OUTLINE ONLY
    container_draw.ellipse([
        right_ring_center_x - right_ring_radius,
        right_ring_center_y - right_ring_radius,
        right_ring_center_x + right_ring_radius,
        right_ring_center_y + right_ring_radius
    ], outline=color_rgb, width=right_ring_thickness)
    
    # V135 IMPROVED: Draw inner ring OUTLINE ONLY
    inner_radius = right_ring_radius - right_ring_thickness - 5
    container_draw.ellipse([
        right_ring_center_x - inner_radius,
        right_ring_center_y - inner_radius,
        right_ring_center_x + inner_radius,
        right_ring_center_y + inner_radius
    ], outline=color_rgb, width=6)
    
    # V135 ENHANCED: Better diamond positioning and size
    diamond_size = 10
    diamond_y = left_ring_center_y - left_ring_radius + left_ring_thickness//2
    container_draw.polygon([
        (left_ring_center_x, diamond_y - diamond_size),
        (left_ring_center_x + diamond_size//2, diamond_y - diamond_size//2),
        (left_ring_center_x, diamond_y),
        (left_ring_center_x - diamond_size//2, diamond_y - diamond_size//2)
    ], fill=(255, 255, 255), outline=(200, 200, 200), width=2)
    
    small_diamond_size = 8
    small_diamond_y = right_ring_center_y - right_ring_radius + right_ring_thickness//2
    container_draw.polygon([
        (right_ring_center_x, small_diamond_y - small_diamond_size),
        (right_ring_center_x + small_diamond_size//2, small_diamond_y - small_diamond_size//2),
        (right_ring_center_x, small_diamond_y),
        (right_ring_center_x - small_diamond_size//2, small_diamond_y - small_diamond_size//2)
    ], fill=(255, 255, 255), outline=(200, 200, 200), width=2)

def create_ai_generated_md_talk(claude_text, width=FIXED_WIDTH):
    """V135 ENHANCED: Create MD Talk text section with improved Korean text handling"""
    print("V135: Creating enhanced MD TALK text section with Korean support")
    
    section_height = 800
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    body_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 48)
                body_font = ImageFont.truetype(font_path, 28)  # V135: Increased font size
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    title = "MD TALK"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 100), title, font=title_font, fill=(40, 40, 40))
    
    if claude_text:
        # V135 ENHANCED: Better Korean text cleaning
        cleaned_text = clean_claude_text(claude_text)
        
        # Remove title prefixes
        cleaned_text = re.sub(r'^(MD TALK|md talk|MD talk|엠디톡)\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        # V135 ENHANCED: Better text wrapping for Korean characters
        words = cleaned_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 120:  # V135: More margin
                if current_line:
                    lines.append(current_line.strip())
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line.strip())
        
    else:
        lines = [
            "Premium texture and balanced details",
            "add depth of emotion to the coupling.",
            "Recommended for couples who want to feel delicate connection."
        ]
    
    # V135 ENHANCED: Better text positioning
    y_pos = 250
    line_height = 55  # V135: Increased line height
    
    for line in lines:
        # V135 ENHANCED: Better encoding safety for each line
        safe_line = ultra_safe_string_encode(line)
        line_width, _ = get_text_dimensions(draw, safe_line, body_font)
        
        # V135 ENHANCED: Text shadow for better readability
        shadow_x = width//2 - line_width//2 + 1
        shadow_y = y_pos + 1
        draw.text((shadow_x, shadow_y), safe_line, font=body_font, fill=(200, 200, 200))
        
        # Main text
        draw.text((width//2 - line_width//2, y_pos), safe_line, font=body_font, fill=(60, 60, 60))
        y_pos += line_height
    
    print("V135: Enhanced MD TALK section completed")
    return section_img

def create_ai_generated_design_point(claude_text, width=FIXED_WIDTH):
    """V135 ENHANCED: Create Design Point text section with improved Korean text handling"""
    print("V135: Creating enhanced DESIGN POINT text section with Korean support")
    
    section_height = 900
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    body_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 48)
                body_font = ImageFont.truetype(font_path, 24)  # V135: Increased font size
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    title = "DESIGN POINT"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 80), title, font=title_font, fill=(40, 40, 40))
    
    if claude_text:
        # V135 ENHANCED: Better Korean text cleaning
        cleaned_text = clean_claude_text(claude_text)
        
        # Remove title prefixes
        cleaned_text = re.sub(r'^(DESIGN POINT|design point|Design Point|디자인포인트|디자인 포인트)\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        # V135 ENHANCED: Better text wrapping for Korean characters
        words = cleaned_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 120:  # V135: More margin
                if current_line:
                    lines.append(current_line.strip())
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line.strip())
        
    else:
        lines = [
            "Leaf ring matte texture and glossy line harmony",
            "conveys solid emotion and women's single items",
            "Pave setting and delicate milgrain details",
            "express luxurious and sophisticated reflection"
        ]
    
    # V135 ENHANCED: Better text positioning
    y_pos = 250
    line_height = 60  # V135: Increased line height
    
    for line in lines:
        # V135 ENHANCED: Better encoding safety for each line
        safe_line = ultra_safe_string_encode(line)
        line_width, _ = get_text_dimensions(draw, safe_line, body_font)
        
        # V135 ENHANCED: Text shadow for better readability
        shadow_x = width//2 - line_width//2 + 1
        shadow_y = y_pos + 1
        draw.text((shadow_x, shadow_y), safe_line, font=body_font, fill=(200, 200, 200))
        
        # Main text
        draw.text((width//2 - line_width//2, y_pos), safe_line, font=body_font, fill=(60, 60, 60))
        y_pos += line_height
    
    print("V135: Enhanced DESIGN POINT section completed")
    return section_img

def extract_file_id_from_url(url):
    """Extract Google Drive file ID from various URL formats"""
    if not url:
        return None
        
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/d/([a-zA-Z0-9_-]+)',
        r'^([a-zA-Z0-9_-]{25,})$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def download_image_from_google_drive(url):
    """Download image from Google Drive URL"""
    try:
        print(f"Processing Google Drive URL: {url}")
        
        file_id = extract_file_id_from_url(url)
        if not file_id:
            raise ValueError(f"Could not extract file ID from URL: {url}")
        
        print(f"Extracted file ID: {file_id}")
        
        download_urls = [
            f'https://drive.google.com/uc?export=download&id={file_id}',
            f'https://drive.google.com/uc?export=download&id={file_id}&confirm=t',
            f'https://docs.google.com/uc?export=download&id={file_id}'
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        session = requests.Session()
        
        for download_url in download_urls:
            try:
                print(f"Trying: {download_url}")
                response = session.get(download_url, headers=headers, stream=True, timeout=30)
                
                content_type = response.headers.get('content-type', '')
                if response.status_code == 200 and ('image' in content_type or len(response.content) > 1000):
                    img = Image.open(BytesIO(response.content))
                    print(f"Successfully downloaded image: {img.size}")
                    return img
                    
            except Exception as e:
                print(f"Failed with URL: {download_url}, Error: {str(e)}")
                continue
        
        raise Exception("Failed to download from all URLs")
        
    except Exception as e:
        print(f"Error downloading from Google Drive: {str(e)}")
        raise

def get_image_from_input(input_data):
    """Get image from URL or base64"""
    try:
        image_url = (input_data.get('image_url') or 
                    input_data.get('imageUrl') or 
                    input_data.get('url') or 
                    input_data.get('webContentLink') or
                    input_data.get('image') or '')
        
        if image_url:
            print(f"Found image URL: {image_url}")
            if 'drive.google.com' in image_url or 'docs.google.com' in image_url:
                return download_image_from_google_drive(image_url)
            else:
                response = requests.get(image_url, timeout=30)
                return Image.open(BytesIO(response.content))
        
        image_base64 = (input_data.get('image_base64') or 
                       input_data.get('base64') or 
                       input_data.get('image_data') or 
                       input_data.get('enhanced_image') or '')
        
        if image_base64:
            print(f"Using base64 data, length: {len(image_base64)}")
            if image_base64.startswith('data:'):
                image_base64 = image_base64.split(',')[1]
            
            missing_padding = len(image_base64) % 4
            if missing_padding:
                image_base64 += '=' * (4 - missing_padding)
            
            image_data = base64.b64decode(image_base64)
            return Image.open(BytesIO(image_data))
        
        raise ValueError("No image URL or base64 data provided")
        
    except Exception as e:
        print(f"Error getting image: {e}")
        raise

def calculate_image_height(original_width, original_height, target_width):
    """Calculate proportional height for target width"""
    ratio = target_width / original_width
    return int(original_height * ratio)

def process_single_image(input_data, group_number):
    """Process single image (groups 1, 2) - V135 NO PAGE NUMBERING"""
    print(f"Processing single image for group {group_number}")
    
    img = get_image_from_input(input_data)
    print(f"Original image size: {img.size}")
    
    new_height = calculate_image_height(img.width, img.height, FIXED_WIDTH)
    
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    TOTAL_HEIGHT = TOP_MARGIN + new_height + BOTTOM_MARGIN
    
    detail_page = Image.new('RGB', (FIXED_WIDTH, TOTAL_HEIGHT), '#FFFFFF')
    
    resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    img_resized = img.resize((FIXED_WIDTH, new_height), resample_filter)
    
    detail_page.paste(img_resized, (0, TOP_MARGIN))
    
    # V135: NO PAGE NUMBERING TEXT
    
    return detail_page

def process_clean_combined_images(images_data, group_number, input_data=None):
    """Process combined images WITHOUT text sections (groups 3, 4, 5) - V135 NO PAGE NUMBERING"""
    print(f"Processing {len(images_data)} CLEAN images for group {group_number} (NO TEXT SECTIONS)")
    
    # Special handling for GROUP 5
    if group_number == 5:
        print("=== GROUP 5 SPECIAL HANDLING ===")
        
        if len(images_data) >= 2:
            print(f"Using first 2 images from images_data (total: {len(images_data)})")
            images_data = images_data[:2]
        elif input_data and 'image7' in input_data and 'image8' in input_data:
            print("Found image7 and image8 keys")
            images_data = [
                {'url': input_data['image7']},
                {'url': input_data['image8']}
            ]
        else:
            print(f"WARNING: Group 5 with insufficient images. Have: {len(images_data)}")
    
    TOP_MARGIN = 100
    BOTTOM_MARGIN = 100
    IMAGE_SPACING = 200
    
    total_height = TOP_MARGIN + BOTTOM_MARGIN
    
    image_heights = []
    for img_data in images_data:
        img = get_image_from_input(img_data)
        img_height = calculate_image_height(img.width, img.height, FIXED_WIDTH)
        image_heights.append(img_height)
        total_height += img_height
        img.close()
    
    total_height += (len(images_data) - 1) * IMAGE_SPACING
    
    print(f"Creating CLEAN combined canvas: {FIXED_WIDTH}x{total_height}")
    
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    
    current_y = TOP_MARGIN
    
    for idx, (img_data, img_height) in enumerate(zip(images_data, image_heights)):
        if idx > 0:
            current_y += IMAGE_SPACING
        
        img = get_image_from_input(img_data)
        
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        img_resized = img.resize((FIXED_WIDTH, img_height), resample_filter)
        
        detail_page.paste(img_resized, (0, current_y))
        current_y += img_height
        
        img.close()
    
    # V135: NO PAGE NUMBERING TEXT
    
    return detail_page

def process_color_section(input_data):
    """V135 ENHANCED: Process group 6 - COLOR section with improved ring processing"""
    print("V135: === PROCESSING ENHANCED GROUP 6 COLOR SECTION ===")
    
    # Multiple ways to find the image for color section
    img = None
    
    # Method 1: Check for image9 key
    if 'image9' in input_data:
        print("V135: Found image9 key for COLOR section")
        img_data = {'url': input_data['image9']}
        img = get_image_from_input(img_data)
    # Method 2: Check for group6 key
    elif 'group6' in input_data:
        print("V135: Found group6 key for COLOR section")
        img_data = {'url': input_data['group6']}
        img = get_image_from_input(img_data)
    # Method 3: Check for image6 key
    elif 'image6' in input_data:
        print("V135: Found image6 key for COLOR section")
        img_data = {'url': input_data['image6']}
        img = get_image_from_input(img_data)
    # Method 4: Check standard image input
    else:
        print("V135: Using standard image input for COLOR section")
        try:
            img = get_image_from_input(input_data)
        except:
            print("V135: No image found for COLOR section")
            img = None
    
    if img:
        print(f"V135 SUCCESS: Ring image for enhanced color section: {img.size}, mode: {img.mode}")
    else:
        print("V135 WARNING: No ring image found, creating enhanced version without ring image")
    
    # V135 ENHANCED: Pass the actual ring image for better processing
    color_section = create_color_options_section(ring_image=img)
    
    if img:
        img.close()
    
    print("V135: Enhanced color section created successfully")
    return color_section

def process_text_section(input_data, group_number):
    """V135 ULTIMATE FIX: Process text sections with PERFECT Korean encoding"""
    print(f"V135: Processing text section for group {group_number} with perfect Korean support")
    
    # V135 ENHANCED: Check for base64 encoded text first
    claude_text_base64 = input_data.get('claude_text_base64', '')
    claude_text = ""
    
    if claude_text_base64:
        try:
            print("V135: Found base64 encoded claude_text")
            # Add padding if needed
            missing_padding = len(claude_text_base64) % 4
            if missing_padding:
                claude_text_base64 += '=' * (4 - missing_padding)
            
            # V135 ULTIMATE: Perfect Korean character handling
            try:
                # First: Try direct UTF-8 decode
                decoded_bytes = base64.b64decode(claude_text_base64)
                claude_text = decoded_bytes.decode('utf-8')
                print("V135 SUCCESS: Direct UTF-8 decode successful")
            except UnicodeDecodeError as e1:
                print(f"V135 INFO: Direct UTF-8 failed ({e1}), trying with error handling")
                try:
                    # Second: Try UTF-8 with error replacement (keeps Korean chars)
                    claude_text = decoded_bytes.decode('utf-8', errors='replace')
                    print("V135 SUCCESS: UTF-8 with replacement successful")
                except Exception as e2:
                    print(f"V135 INFO: UTF-8 with replacement failed ({e2}), trying alternative")
                    try:
                        # Third: Try alternative encoding that preserves Unicode
                        claude_text = decoded_bytes.decode('unicode_escape', errors='ignore')
                        print("V135 SUCCESS: Unicode escape decoding successful")
                    except Exception as e3:
                        print(f"V135 WARNING: All decoding attempts failed ({e3}), using safe fallback")
                        # Ultimate fallback: Use placeholder that indicates encoding issue
                        claude_text = "텍스트 인코딩 오류 - 한국어 문자 처리 실패"
                        print("V135 FALLBACK: Using Korean placeholder text")
        except Exception as e:
            print(f"V135 ERROR: Base64 decoding failed: {e}")
            claude_text = ""
    else:
        # Fallback to regular text fields
        claude_text = (input_data.get('claude_text') or 
                      input_data.get('text_content') or 
                      input_data.get('ai_text') or 
                      input_data.get('generated_text') or '')
    
    # V135 ENHANCED: Clean the text with Korean character preservation
    if claude_text:
        claude_text = ultra_safe_string_encode(claude_text)
        claude_text = clean_claude_text(claude_text)
    
    text_type = (input_data.get('text_type') or 
                input_data.get('section_type') or '')
    
    print(f"V135: Text type: {text_type}")
    print(f"V135: Group number: {group_number}")
    print(f"V135: Korean text preview: {claude_text[:100] if claude_text else 'No text'}...")
    
    # Create text section based on group number
    if group_number == 7:
        print("V135: GROUP 7 CONFIRMED - Creating enhanced MD TALK section")
        text_section = create_ai_generated_md_talk(claude_text)
        section_type = "md_talk"
    elif group_number == 8:
        print("V135: GROUP 8 CONFIRMED - Creating enhanced DESIGN POINT section")
        text_section = create_ai_generated_design_point(claude_text)
        section_type = "design_point"
    else:
        print(f"V135 WARNING: Unexpected group number {group_number} for text section")
        if 'md' in text_type.lower():
            text_section = create_ai_generated_md_talk(claude_text)
            section_type = "md_talk"
        else:
            text_section = create_ai_generated_design_point(claude_text)
            section_type = "design_point"
    
    print(f"V135: Enhanced text section created successfully: {section_type}")
    return text_section, section_type

def send_to_webhook(image_base64, handler_type, file_name, route_number=0, metadata={}):
    """V135 ENHANCED: Send results to webhook with perfect Korean text encoding"""
    try:
        if not WEBHOOK_URL:
            print("WARNING: Webhook URL not configured, skipping webhook send")
            return None
        
        # V135 ENHANCED: Ensure all metadata is Korean-safe
        safe_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                safe_metadata[key] = ultra_safe_string_encode(value)
            elif isinstance(value, dict):
                safe_dict = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        safe_dict[sub_key] = ultra_safe_string_encode(sub_value)
                    else:
                        safe_dict[sub_key] = sub_value
                safe_metadata[key] = safe_dict
            else:
                safe_metadata[key] = value
                
        webhook_data = {
            "handler_type": ultra_safe_string_encode(handler_type),
            "file_name": ultra_safe_string_encode(file_name),
            "route_number": route_number,
            "runpod_result": {
                "output": {
                    "output": safe_metadata
                }
            }
        }
        
        webhook_data["runpod_result"]["output"]["output"]["enhanced_image"] = image_base64
        
        print(f"V135: Sending to webhook: {handler_type} for {file_name}")
        
        # V135 ENHANCED: Test JSON serialization with Korean support
        try:
            test_json = json.dumps(webhook_data, ensure_ascii=False)  # V135: Support Korean
            print("V135: JSON serialization test passed with Korean support")
        except Exception as json_err:
            print(f"V135: JSON serialization failed: {json_err}")
            # Try with ASCII fallback
            try:
                test_json = json.dumps(webhook_data, ensure_ascii=True)
                print("V135: JSON serialization fallback to ASCII successful")
            except Exception as ascii_err:
                print(f"V135: ASCII JSON serialization also failed: {ascii_err}")
                return None
        
        response = requests.post(
            WEBHOOK_URL,
            json=webhook_data,
            headers={'Content-Type': 'application/json; charset=utf-8'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"V135: Webhook success: {result}")
            return result
        else:
            print(f"V135: Webhook failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"V135: Webhook error: {str(e)}")
        return None

def detect_group_number_from_input(input_data):
    """V135 ENHANCED: Group number detection with improved accuracy"""
    print("=== GROUP NUMBER DETECTION V135 - ENHANCED VERSION ===")
    print(f"Full input_data keys: {sorted(input_data.keys())}")
    
    # PRIORITY 1: Direct route_number - ABSOLUTE HIGHEST PRIORITY!
    route_number = input_data.get('route_number', 0)
    try:
        route_number = int(str(route_number)) if str(route_number).isdigit() else 0
    except:
        route_number = 0
        
    if route_number > 0:
        print(f"Found route_number: {route_number} - USING IT DIRECTLY")
        return route_number
    
    # PRIORITY 2: Check for image9 (Google Script Group 6)
    if 'image9' in input_data:
        print("Found image9 - GROUP 6 (COLOR)")
        return 6
    
    # PRIORITY 3: Check for group keys
    if 'group6' in input_data or 'group_6' in input_data:
        print("Found group6 key - GROUP 6")
        return 6
    
    # PRIORITY 4: group_number field
    group_number = input_data.get('group_number', 0)
    try:
        group_number = int(str(group_number)) if str(group_number).isdigit() else 0
    except:
        group_number = 0
        
    if group_number > 0:
        print(f"Found group_number: {group_number}")
        return group_number
    
    # PRIORITY 5: Check text_type for groups 7, 8
    text_type = str(input_data.get('text_type', ''))
    if text_type == 'md_talk':
        print("Found md_talk text_type - GROUP 7")
        return 7
    elif text_type == 'design_point':
        print("Found design_point text_type - GROUP 8")
        return 8
    
    # PRIORITY 6: Check for COLOR indicators
    all_text = str(input_data).lower()
    if any(word in all_text for word in ['color', 'colour', '컬러', '색상', '컬러섹션', 'color_section']):
        print("Found COLOR keywords - GROUP 6")
        return 6
    
    # PRIORITY 7: Analyze URLs in 'image' field
    if 'image' in input_data and not any(f'image{i}' in input_data for i in range(1, 10)):
        image_value = str(input_data.get('image', ''))
        if ';' in image_value:
            urls = [u.strip() for u in image_value.split(';') if u.strip()]
            url_count = len(urls)
            print(f"Found {url_count} URLs in 'image' field")
            
            if url_count == 2:
                return 3
            elif url_count >= 3:
                return 4
        else:
            return 1
    
    # PRIORITY 8: Check for specific image keys
    if 'image1' in input_data and not any(f'image{i}' in input_data for i in range(2, 10)):
        return 1
    if 'image2' in input_data and not any(f'image{i}' in input_data for i in [1,3,4,5,6,7,8,9]):
        return 2
    if ('image3' in input_data or 'image4' in input_data):
        return 3
    if 'image5' in input_data:
        return 4
    if 'image6' in input_data and not any(f'image{i}' in input_data for i in [1,2,3,4,5,7,8,9]):
        return 6
    if ('image7' in input_data or 'image8' in input_data):
        return 5
    
    # Last resort
    for i in range(1, 10):
        key = f'image{i}'
        if key in input_data:
            if i in [1, 2]:
                return i
            elif i in [3, 4]:
                return 3
            elif i == 5:
                return 4
            elif i == 6:
                return 6
            elif i in [7, 8]:
                return 5
            elif i == 9:
                return 6
    
    print("ERROR: Could not determine group number")
    return 0

def handler(event):
    """V135 ULTIMATE: Main handler with perfect Korean support and enhanced features"""
    try:
        print(f"=== V135 Detail Page Handler - ULTIMATE ENHANCED VERSION ===")
        
        # Download Korean font if not exists
        if not os.path.exists('/tmp/NanumMyeongjo.ttf'):
            print("Korean font not found, downloading...")
            if not download_korean_font():
                print("WARNING: Failed to download Korean font, text may appear corrupted")
        else:
            print("Korean font already exists")
        
        # Get input data
        input_data = event.get('input', event)
        
        print(f"=== INCOMING DATA ===")
        print(f"Keys: {sorted(input_data.keys())}")
        print(f"route_number: {input_data.get('route_number', 'NOT FOUND')}")
        
        # Group detection
        group_number = detect_group_number_from_input(input_data)
        
        print(f"\n=== DETECTION RESULT ===")
        print(f"Detected group_number: {group_number}")
        
        # Route number override
        route_str = str(input_data.get('route_number', '0'))
        try:
            route_int = int(route_str) if route_str.isdigit() else 0
        except:
            route_int = 0
            
        if route_int > 0:
            original_group = group_number
            group_number = route_int
            if original_group != group_number:
                print(f"!!! OVERRIDE: Changed group {original_group} → {group_number} based on route_number")
        
        print(f"\n=== FINAL GROUP: {group_number} ===")
        
        # Validate group number
        if group_number == 0:
            raise ValueError(f"Could not determine group number. Keys: {list(input_data.keys())}")
        
        if group_number < 1 or group_number > 8:
            raise ValueError(f"Invalid group number: {group_number}. Must be 1-8.")
        
        # Handle image input formats
        if 'image' in input_data and input_data['image']:
            print(f"Found 'image' key with value: {input_data['image'][:100]}...")
            image_data = input_data['image']
            
            if ';' in str(image_data):
                urls = str(image_data).split(';')
                input_data['images'] = []
                for url in urls:
                    url = url.strip()
                    if url:
                        input_data['images'].append({'url': url})
                print(f"Converted 'image' to {len(input_data['images'])} images array")
            else:
                input_data['url'] = image_data
                print(f"Set single URL from 'image' key")
        
        if 'combined_urls' in input_data and input_data['combined_urls']:
            urls = str(input_data['combined_urls']).split(';')
            input_data['images'] = []
            for url in urls:
                url = url.strip()
                if url:
                    input_data['images'].append({'url': url})
            print(f"Converted combined_urls to {len(input_data['images'])} images")
        
        # Handle specific image keys
        if not input_data.get('images') and not input_data.get('url'):
            if group_number == 6:
                # V135 ENHANCED: Better GROUP 6 image handling
                if 'image9' in input_data:
                    input_data['url'] = input_data['image9']
                    print("V135 GROUP 6: Using image9")
                elif 'image6' in input_data:
                    input_data['url'] = input_data['image6']
                    print("V135 GROUP 6: Using image6")
                elif 'group6' in input_data:
                    input_data['url'] = input_data['group6']
                    print("V135 GROUP 6: Using group6")
                else:
                    print("V135 WARNING: GROUP 6 but no specific image found")
            
            elif f'image{group_number}' in input_data:
                image_url = input_data[f'image{group_number}']
                if ';' in str(image_url):
                    urls = str(image_url).split(';')
                    input_data['images'] = [{'url': url.strip()} for url in urls if url.strip()]
                else:
                    input_data['url'] = image_url
                print(f"Found and processed image{group_number}")
            
            elif group_number == 3:
                images_to_add = []
                if 'image3' in input_data:
                    images_to_add.append({'url': input_data['image3']})
                if 'image4' in input_data:
                    images_to_add.append({'url': input_data['image4']})
                if images_to_add:
                    input_data['images'] = images_to_add
                    print(f"GROUP 3: Found {len(images_to_add)} images")
            
            elif group_number == 4:
                images_to_add = []
                if 'image5' in input_data:
                    images_to_add.append({'url': input_data['image5']})
                if 'image6' in input_data and 'image5' in input_data:
                    images_to_add.append({'url': input_data['image6']})
                if images_to_add:
                    input_data['images'] = images_to_add
                    print(f"GROUP 4: Found {len(images_to_add)} images")
            
            elif group_number == 5:
                images_to_add = []
                if 'image7' in input_data:
                    images_to_add.append({'url': input_data['image7']})
                if 'image8' in input_data:
                    images_to_add.append({'url': input_data['image8']})
                if images_to_add:
                    input_data['images'] = images_to_add
                    print(f"GROUP 5: Found {len(images_to_add)} images")
        
        # Process based on group number
        if group_number == 6:
            print("V135: === Processing ENHANCED GROUP 6: COLOR section ===")
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif group_number == 7:
            print("V135: === Processing ENHANCED GROUP 7: MD TALK text section ===")
            detail_page, section_type = process_text_section(input_data, 7)
            page_type = f"text_section_{section_type}"
            
        elif group_number == 8:
            print("V135: === Processing ENHANCED GROUP 8: DESIGN POINT text section ===")
            detail_page, section_type = process_text_section(input_data, 8)
            page_type = f"text_section_{section_type}"
            
        elif group_number in [1, 2]:
            print(f"V135: === Processing GROUP {group_number}: Individual image ===")
            detail_page = process_single_image(input_data, group_number)
            page_type = "individual"
            
        elif group_number in [3, 4, 5]:
            print(f"V135: === Processing GROUP {group_number}: Combined images ===")
            if 'images' not in input_data or not isinstance(input_data['images'], list):
                input_data['images'] = [input_data]
            
            if group_number == 5:
                detail_page = process_clean_combined_images(input_data.get('images', []), group_number, input_data)
            else:
                detail_page = process_clean_combined_images(input_data['images'], group_number, input_data)
            
            page_type = "clean_combined"
        
        else:
            raise ValueError(f"Invalid group number: {group_number}. Must be 1-8.")
        
        # Convert to base64
        buffered = BytesIO()
        detail_page.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue())
        
        detail_base64 = img_str.decode('utf-8')
        detail_base64_no_padding = detail_base64.rstrip('=')
        
        print(f"V135: Detail page created: {detail_page.size}")
        print(f"V135: Base64 length: {len(detail_base64_no_padding)} chars")
        
        # V135 ENHANCED: Metadata with Korean support
        metadata = {
            "enhanced_image": detail_base64_no_padding,
            "status": "success",
            "page_type": ultra_safe_string_encode(page_type),
            "page_number": group_number,
            "route_number": group_number,
            "actual_group": group_number,
            "dimensions": {
                "width": detail_page.width,
                "height": detail_page.height
            },
            "version": "V135_ULTIMATE_ENHANCED",
            "image_count": len(input_data.get('images', [input_data])),
            "processing_time": "calculated_later",
            "font_status": "korean_font_available" if os.path.exists('/tmp/NanumMyeongjo.ttf') else "fallback_font",
            "korean_support": "enabled",
            "enhanced_features": "background_removal_improved,text_rendering_enhanced,color_accuracy_improved"
        }
        
        # Send to webhook
        file_name = f"detail_group_{group_number}.png"
        webhook_result = send_to_webhook(detail_base64_no_padding, "detail", file_name, group_number, metadata)
        
        # Return response
        return {
            "output": metadata
        }
        
    except Exception as e:
        # V135 ENHANCED: Error handling with Korean support
        error_msg = ultra_safe_string_encode(f"Detail page creation failed: {str(e)}")
        print(f"V135 ERROR: {error_msg}")
        traceback_str = ultra_safe_string_encode(traceback.format_exc())
        print(f"V135 TRACEBACK: {traceback_str}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "error",
                "traceback": traceback_str,
                "version": "V135_ULTIMATE_ENHANCED"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("Starting Detail Page Handler V135 - ULTIMATE ENHANCED VERSION...")
    runpod.serverless.start({"handler": handler})
