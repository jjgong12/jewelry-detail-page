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

def clean_claude_text(text):
    """Clean text for safe JSON encoding while preserving Korean characters"""
    if not text:
        return ""
    
    # Convert to string if needed
    text = str(text) if text is not None else ""
    
    # CRITICAL: Never use decode('unicode_escape') - it destroys Korean text!
    # Just handle the text as-is
    
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
    
    # Trim length
    if len(text) > 500:
        text = text[:497] + "..."
    
    # IMPORTANT: No character filtering by Unicode value!
    # Korean characters (한글) have values > 0xAC00
    
    print(f"Cleaned text preview: {text[:100]}...")
    return text.strip()

def get_text_dimensions(draw, text, font):
    """Get text dimensions compatible with all PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def extract_ring_with_replicate(img):
    """Extract ring from background using Replicate API"""
    try:
        if not REPLICATE_AVAILABLE:
            print("Replicate not available, using local fallback")
            return extract_ring_local_fallback(img)
            
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print("Replicate API token not found, using local fallback")
            return extract_ring_local_fallback(img)
            
        print("Starting Replicate background removal...")
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        output = replicate.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            input={
                "image": f"data:image/png;base64,{img_base64}"
            }
        )
        
        response = requests.get(output)
        result_img = Image.open(BytesIO(response.content))
        
        if result_img.mode != 'RGBA':
            result_img = result_img.convert('RGBA')
        
        print("Replicate background removal completed successfully")
        return result_img
        
    except Exception as e:
        print(f"Error with Replicate API: {e}")
        print("Falling back to local method...")
        return extract_ring_local_fallback(img)

def extract_ring_local_fallback(img):
    """Local fallback method for background removal"""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    width, height = img.size
    img_array = np.array(img)
    
    corner_size = 10
    corners = []
    corners.extend(img_array[:corner_size, :corner_size].reshape(-1, 4))
    corners.extend(img_array[:corner_size, -corner_size:].reshape(-1, 4))
    corners.extend(img_array[-corner_size:, :corner_size].reshape(-1, 4))
    corners.extend(img_array[-corner_size:, -corner_size:].reshape(-1, 4))
    
    corners_array = np.array(corners)
    bg_color = np.median(corners_array, axis=0)[:3]
    
    color_distance = np.sqrt(
        (img_array[:,:,0] - bg_color[0])**2 +
        (img_array[:,:,1] - bg_color[1])**2 +
        (img_array[:,:,2] - bg_color[2])**2
    )
    
    threshold = np.percentile(color_distance, 30)
    mask = color_distance > threshold
    
    mask = mask.astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, 'L')
    
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    mask_img = mask_img.filter(ImageFilter.SMOOTH_MORE)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))
    
    result = img.copy()
    result.putalpha(mask_img)
    
    return result

def apply_metal_color_filter(img, color_multipliers):
    """Apply metal color filter to image"""
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
    else:
        img = img.convert('RGBA')
        r, g, b, a = img.split()
    
    r = r.point(lambda x: min(255, int(x * color_multipliers[0])))
    g = g.point(lambda x: min(255, int(x * color_multipliers[1])))
    b = b.point(lambda x: min(255, int(x * color_multipliers[2])))
    
    return Image.merge('RGBA', (r, g, b, a))

def create_color_options_section(width=FIXED_WIDTH, ring_image=None):
    """Create COLOR section with simple CSS-style rings"""
    section_height = 1000
    section_img = Image.new('RGB', (width, section_height), '#F8F8F8')  # Light gray background like HTML
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    label_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 48)  # Smaller like HTML
                label_font = ImageFont.truetype(font_path, 24)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Title with letter spacing effect
    title = "COLOR"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 80), title, font=title_font, fill=(51, 51, 51))
    
    # Color definitions - matching HTML exactly
    colors = [
        ("yellow", (255, 215, 0)),      # #FFD700
        ("rose", (255, 192, 203)),      # #FFC0CB  
        ("white", (229, 229, 229)),     # #E5E5E5
        ("antique", (212, 175, 55))     # #D4AF37
    ]
    
    # Layout settings
    container_size = 300  # Size for each ring container
    h_spacing = 100
    v_spacing = 400
    
    grid_width = 2 * container_size + h_spacing
    start_x = (width - grid_width) // 2
    start_y = 200
    
    for i, (name, color_rgb) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (container_size + h_spacing)
        y = start_y + row * v_spacing
        
        # White container background
        container = Image.new('RGBA', (container_size, container_size), (255, 255, 255, 255))
        container_draw = ImageDraw.Draw(container)
        
        # Draw ring pair like HTML - overlapping circles
        # Left ring (larger)
        left_ring_center_x = container_size // 2 - 30
        left_ring_center_y = container_size // 2
        left_ring_radius = 70
        left_ring_thickness = 20
        
        # Draw left ring
        container_draw.ellipse([
            left_ring_center_x - left_ring_radius,
            left_ring_center_y - left_ring_radius,
            left_ring_center_x + left_ring_radius,
            left_ring_center_y + left_ring_radius
        ], fill=color_rgb, outline=(0, 0, 0, 30), width=1)
        
        # Inner circle for left ring
        container_draw.ellipse([
            left_ring_center_x - (left_ring_radius - left_ring_thickness),
            left_ring_center_y - (left_ring_radius - left_ring_thickness),
            left_ring_center_x + (left_ring_radius - left_ring_thickness),
            left_ring_center_y + (left_ring_radius - left_ring_thickness)
        ], fill=(255, 255, 255))
        
        # Right ring (smaller)
        right_ring_center_x = container_size // 2 + 40
        right_ring_center_y = container_size // 2 + 20
        right_ring_radius = 55
        right_ring_thickness = 18
        
        # Draw right ring
        container_draw.ellipse([
            right_ring_center_x - right_ring_radius,
            right_ring_center_y - right_ring_radius,
            right_ring_center_x + right_ring_radius,
            right_ring_center_y + right_ring_radius
        ], fill=color_rgb, outline=(0, 0, 0, 30), width=1)
        
        # Inner circle for right ring
        container_draw.ellipse([
            right_ring_center_x - (right_ring_radius - right_ring_thickness),
            right_ring_center_y - (right_ring_radius - right_ring_thickness),
            right_ring_center_x + (right_ring_radius - right_ring_thickness),
            right_ring_center_y + (right_ring_radius - right_ring_thickness)
        ], fill=(255, 255, 255))
        
        # Add simple diamond on top of each ring
        # Left ring diamond
        diamond_size = 8
        diamond_y = left_ring_center_y - left_ring_radius + left_ring_thickness//2
        container_draw.polygon([
            (left_ring_center_x, diamond_y - diamond_size),
            (left_ring_center_x + diamond_size//2, diamond_y - diamond_size//2),
            (left_ring_center_x, diamond_y),
            (left_ring_center_x - diamond_size//2, diamond_y - diamond_size//2)
        ], fill=(255, 255, 255), outline=(180, 180, 180))
        
        # Right ring diamond
        small_diamond_size = 6
        small_diamond_y = right_ring_center_y - right_ring_radius + right_ring_thickness//2
        container_draw.polygon([
            (right_ring_center_x, small_diamond_y - small_diamond_size),
            (right_ring_center_x + small_diamond_size//2, small_diamond_y - small_diamond_size//2),
            (right_ring_center_x, small_diamond_y),
            (right_ring_center_x - small_diamond_size//2, small_diamond_y - small_diamond_size//2)
        ], fill=(255, 255, 255), outline=(180, 180, 180))
        
        # Add subtle shadow
        shadow_img = Image.new('RGBA', (container_size + 10, container_size + 10), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        shadow_draw.rectangle([5, 5, container_size + 5, container_size + 5], 
                            fill=(0, 0, 0, 20))
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Paste shadow first
        section_img.paste(shadow_img, (x - 5, y - 5), shadow_img)
        
        # Paste container
        section_img.paste(container, (x, y))
        
        # Draw label
        label_width, _ = get_text_dimensions(draw, name, label_font)
        draw.text((x + container_size//2 - label_width//2, y + container_size + 30), 
                 name, font=label_font, fill=(102, 102, 102))
    
    return section_img

def create_ai_generated_md_talk(claude_text, width=FIXED_WIDTH):
    """Create MD Talk text section from Claude-generated content"""
    section_height = 600
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    body_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 48)
                body_font = ImageFont.truetype(font_path, 24)
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
        cleaned_text = clean_claude_text(claude_text)
        
        # Force line breaks every 10-12 characters for Korean text
        words = cleaned_text.split()
        lines = []
        current_line = ""
        char_count = 0
        
        for word in words:
            # Check if adding this word would exceed character limit
            if char_count + len(word) > 12:  # 10-12 characters per line
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
                char_count = len(word) + 1
            else:
                current_line += word + " "
                char_count += len(word) + 1
        
        if current_line:
            lines.append(current_line.strip())
        
        # Limit to 3 lines for MD TALK
        lines = lines[:3]
    else:
        lines = [
            "고급스러운 텍스처와 균형 잡힌 디테일이",
            "감성의 깊이를 더하는 커플링입니다.",
            "섬세한 연결을 느끼고 싶은 커플에게 추천드립니다."
        ]
    
    y_pos = 250  # Start lower for better centering
    line_height = 50
    
    for line in lines:
        line_width, _ = get_text_dimensions(draw, line, body_font)
        draw.text((width//2 - line_width//2, y_pos), line, font=body_font, fill=(80, 80, 80))
        y_pos += line_height
    
    return section_img

def create_ai_generated_design_point(claude_text, width=FIXED_WIDTH):
    """Create Design Point text section from Claude-generated content"""
    section_height = 700
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    body_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 48)
                body_font = ImageFont.truetype(font_path, 20)
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
        cleaned_text = clean_claude_text(claude_text)
        
        # Force line breaks every 10-12 characters for Korean text
        words = cleaned_text.split()
        lines = []
        current_line = ""
        char_count = 0
        
        for word in words:
            # Check if adding this word would exceed character limit
            if char_count + len(word) > 12:  # 10-12 characters per line
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
                char_count = len(word) + 1
            else:
                current_line += word + " "
                char_count += len(word) + 1
        
        if current_line:
            lines.append(current_line.strip())
        
        # Limit to 4 lines for DESIGN POINT
        lines = lines[:4]
    else:
        lines = [
            "리프링 무광 텍스처와 유광 라인의 조화가",
            "견고한 감성을 전하고 여자 단품은",
            "파베 세팅과 섬세한 밀그레인의 디테일",
            "화려하면서도 고급스러운 반영을 표현합니다"
        ]
    
    y_pos = 250  # Start lower for better centering
    line_height = 55
    
    for line in lines:
        line_width, _ = get_text_dimensions(draw, line, body_font)
        draw.text((width//2 - line_width//2, y_pos), line, font=body_font, fill=(80, 80, 80))
        y_pos += line_height
    
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
    """Process single image (groups 1, 2)"""
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
    
    draw = ImageDraw.Draw(detail_page)
    actual_page_number = input_data.get('route_number', group_number)
    page_text = f"- {actual_page_number} -"
    
    small_font = None
    for font_path in ["/tmp/NanumMyeongjo.ttf", 
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if os.path.exists(font_path):
            try:
                small_font = ImageFont.truetype(font_path, 16)
                break
            except:
                continue
    
    if small_font is None:
        small_font = ImageFont.load_default()
    
    text_width, _ = get_text_dimensions(draw, page_text, small_font)
    draw.text((FIXED_WIDTH//2 - text_width//2, TOTAL_HEIGHT - 30), 
             page_text, fill=(200, 200, 200), font=small_font)
    
    return detail_page

def process_clean_combined_images(images_data, group_number, input_data=None):
    """Process combined images WITHOUT text sections (groups 3, 4, 5)"""
    print(f"Processing {len(images_data)} CLEAN images for group {group_number} (NO TEXT SECTIONS)")
    
    # CRITICAL FIX: For group 5, we need images 7 and 8
    if group_number == 5:
        print("=== GROUP 5 SPECIAL HANDLING ===")
        print(f"Need to get images 7 and 8 from input_data")
        
        # Check if we have the correct image keys
        if 'image7' in input_data and 'image8' in input_data:
            print("Found image7 and image8 keys")
            images_data = []
            for key in ['image7', 'image8']:
                images_data.append({'url': input_data[key]})
            print(f"Created images_data with 2 entries for group 5")
        elif len(images_data) > 2:
            print(f"WARNING: Group 5 has {len(images_data)} images, using first 2 only")
            images_data = images_data[:2]
    
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
    
    draw = ImageDraw.Draw(detail_page)
    actual_page_number = input_data.get('route_number', group_number) if input_data else group_number
    
    if group_number == 5:
        page_text = f"- Gallery 7-8 -"
    else:
        page_text = f"- {actual_page_number} -"
    
    small_font = None
    for font_path in ["/tmp/NanumMyeongjo.ttf", 
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if os.path.exists(font_path):
            try:
                small_font = ImageFont.truetype(font_path, 16)
                break
            except:
                continue
    
    if small_font is None:
        small_font = ImageFont.load_default()
    
    text_width, _ = get_text_dimensions(draw, page_text, small_font)
    draw.text((FIXED_WIDTH//2 - text_width//2, total_height - 50), 
             page_text, fill=(200, 200, 200), font=small_font)
    
    return detail_page

def process_color_section(input_data):
    """Process group 6 - COLOR section with ring image"""
    print("=== PROCESSING GROUP 6 COLOR SECTION ===")
    
    # Multiple ways to find the image for color section
    img = None
    
    # Method 1: Check for image9 key
    if 'image9' in input_data:
        print("Found image9 key for COLOR section")
        img_data = {'url': input_data['image9']}
        img = get_image_from_input(img_data)
    # Method 2: Check for group6 key
    elif 'group6' in input_data:
        print("Found group6 key for COLOR section")
        img_data = {'url': input_data['group6']}
        img = get_image_from_input(img_data)
    # Method 3: Check if it's a single image input
    else:
        print("Using standard image input for COLOR section")
        img = get_image_from_input(input_data)
    
    if img:
        print(f"Ring image for color section: {img.size}, mode: {img.mode}")
    else:
        print("WARNING: No ring image found, creating without ring image")
    
    # Create color section (with or without ring image)
    color_section = create_color_options_section(ring_image=img)
    
    if img:
        img.close()
    
    print("Color section created successfully")
    return color_section

def process_text_section(input_data, group_number):
    """Process text-only sections (groups 7, 8) with Claude-generated content"""
    print(f"Processing text section for group {group_number}")
    
    # Check for base64 encoded text first
    claude_text_base64 = input_data.get('claude_text_base64', '')
    if claude_text_base64:
        try:
            print("Found base64 encoded claude_text")
            # Add padding if needed
            missing_padding = len(claude_text_base64) % 4
            if missing_padding:
                claude_text_base64 += '=' * (4 - missing_padding)
            
            # Decode from base64
            claude_text = base64.b64decode(claude_text_base64).decode('utf-8')
            print("Successfully decoded base64 claude_text")
        except Exception as e:
            print(f"Error decoding base64: {e}")
            claude_text = ''
    else:
        # Fallback to regular text fields
        claude_text = (input_data.get('claude_text') or 
                      input_data.get('text_content') or 
                      input_data.get('ai_text') or 
                      input_data.get('generated_text') or '')
    
    if claude_text:
        claude_text = clean_claude_text(claude_text)
    
    text_type = (input_data.get('text_type') or 
                input_data.get('section_type') or '')
    
    print(f"Text type: {text_type}")
    print(f"Cleaned Claude text preview: {claude_text[:100] if claude_text else 'No text provided'}...")
    
    if group_number == 7 or text_type == 'md_talk':
        text_section = create_ai_generated_md_talk(claude_text)
        section_type = "md_talk"
    elif group_number == 8 or text_type == 'design_point':
        text_section = create_ai_generated_design_point(claude_text)
        section_type = "design_point"
    else:
        if group_number == 7:
            text_section = create_ai_generated_md_talk(claude_text)
            section_type = "md_talk"
        else:
            text_section = create_ai_generated_design_point(claude_text)
            section_type = "design_point"
    
    return text_section, section_type

def send_to_webhook(image_base64, handler_type, file_name, route_number=0, metadata={}):
    """Send results to Google Apps Script webhook"""
    try:
        if not WEBHOOK_URL:
            print("WARNING: Webhook URL not configured, skipping webhook send")
            return None
            
        webhook_data = {
            "handler_type": handler_type,
            "file_name": file_name,
            "route_number": route_number,
            "runpod_result": {
                "output": {
                    "output": metadata
                }
            }
        }
        
        webhook_data["runpod_result"]["output"]["output"]["enhanced_image"] = image_base64
        
        print(f"Sending to webhook: {handler_type} for {file_name}")
        
        response = requests.post(
            WEBHOOK_URL,
            json=webhook_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Webhook success: {result}")
            return result
        else:
            print(f"Webhook failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Webhook error: {str(e)}")
        return None

def detect_group_number_from_input(input_data):
    """Enhanced group number detection with better group differentiation"""
    print("=== GROUP NUMBER DETECTION ENHANCED V119 ===")
    
    # Method 1: Direct route_number - HIGHEST PRIORITY
    route_number = input_data.get('route_number', 0)
    if route_number > 0:
        print(f"Found route_number: {route_number} - USING THIS AS DEFINITIVE")
        return route_number
    
    # Method 2: group_number - SECOND PRIORITY
    group_number = input_data.get('group_number', 0)
    if group_number > 0:
        print(f"Found group_number: {group_number}")
        return group_number
    
    # Method 3: Check for specific image keys (image1-image9)
    for i in range(1, 10):  # Extended to 9 for color section
        key = f'image{i}'
        if key in input_data:
            print(f"Found {key} key")
            if i == 9:
                print("image9 indicates GROUP 6 (COLOR section)")
                return 6
            else:
                print(f"Assuming group {i}")
                return i
    
    # Method 4: Check for group6 key specifically
    if 'group6' in input_data:
        print("Found group6 key, returning group 6")
        return 6
    
    # Method 5: Check text_type for groups 7, 8
    text_type = input_data.get('text_type', '')
    if text_type == 'md_talk':
        print("Found md_talk text_type, assuming group 7")
        return 7
    elif text_type == 'design_point':
        print("Found design_point text_type, assuming group 8")
        return 8
    
    # Method 6: Check for Claude text presence (base64 or regular)
    has_claude_text = bool(input_data.get('claude_text') or input_data.get('claude_text_base64'))
    if has_claude_text:
        print("Found claude_text, checking text_type...")
        if text_type == 'md_talk':
            return 7
        elif text_type == 'design_point':
            return 8
        else:
            # Default to MD Talk if text type not specified
            print("Has claude_text but no text_type, defaulting to group 7 (MD Talk)")
            return 7
    
    # Method 7: Enhanced URL analysis
    image_data = input_data.get('image', '')
    if image_data:
        print(f"Analyzing image URLs: {image_data[:200]}...")
        
        if ';' in image_data:
            # Multiple URLs
            urls = image_data.split(';')
            url_count = len([url for url in urls if url.strip()])
            print(f"Found {url_count} URLs in image data")
            
            if url_count == 2:
                # Could be groups 3, 4, or 5
                print("2 URLs detected - likely group 3 or 5")
                return 3  # Default to group 3 for 2 images
            elif url_count == 3:
                print("3 URLs detected, assuming group 4")
                return 4  # Default to group 4 for 3 images
        else:
            # Single URL - could be groups 1, 2, 6, 7, 8
            print("Single URL detected")
            
            # Check if it's a color section request
            if 'color' in str(input_data).lower():
                print("Found 'color' keyword, assuming group 6")
                return 6
            
            # For single images without other indicators
            print("Single URL, no route_number - defaulting to group 1")
            return 1
    
    # Last resort
    print("WARNING: Could not reliably detect group number")
    print(f"Available keys in input: {list(input_data.keys())}")
    
    return 0

def handler(event):
    """Main handler for detail page creation - V119 FIXED TEXT & COLOR"""
    try:
        print(f"=== V119 Detail Page Handler - FIXED TEXT & COLOR ===")
        
        # Download Korean font if not exists
        if not os.path.exists('/tmp/NanumMyeongjo.ttf'):
            print("Korean font not found, downloading...")
            if not download_korean_font():
                print("WARNING: Failed to download Korean font, text may appear corrupted")
        else:
            print("Korean font already exists")
        
        # Get input data
        input_data = event.get('input', event)
        
        print(f"Input keys: {list(input_data.keys())}")
        
        # Enhanced group number detection
        group_number = detect_group_number_from_input(input_data)
        route_number = input_data.get('route_number', group_number)
        
        print(f"DETECTED: group_number={group_number}, route_number={route_number}")
        
        # CRITICAL: Always use route_number as the actual group number
        if route_number > 0 and route_number != group_number:
            print(f"OVERRIDE: Using route_number {route_number} instead of detected group {group_number}")
            group_number = route_number
        
        # Validate group number
        if group_number == 0:
            raise ValueError(f"Could not determine group number. Keys: {list(input_data.keys())}")
        
        if group_number < 1 or group_number > 8:
            raise ValueError(f"Invalid group number: {group_number}. Must be 1-8.")
        
        print(f"FINAL GROUP NUMBER: {group_number}")
        
        # Handle Make.com's 'image' key format
        if 'image' in input_data and input_data['image']:
            print(f"Found 'image' key with value: {input_data['image'][:100]}...")
            image_data = input_data['image']
            
            if ';' in image_data:
                urls = image_data.split(';')
                input_data['images'] = []
                for url in urls:
                    url = url.strip()
                    if url:
                        input_data['images'].append({'url': url})
                print(f"Converted 'image' to {len(input_data['images'])} images array")
            else:
                input_data['url'] = image_data
                print(f"Set single URL from 'image' key")
        
        # Handle combined_urls format
        if 'combined_urls' in input_data and input_data['combined_urls']:
            urls = input_data['combined_urls'].split(';')
            input_data['images'] = []
            for url in urls:
                url = url.strip()
                if url:
                    input_data['images'].append({'url': url})
            print(f"Converted combined_urls to {len(input_data['images'])} images")
        
        # Handle Make.com specific image keys
        elif f'image{group_number}' in input_data:
            image_url = input_data[f'image{group_number}']
            if ';' in image_url:
                urls = image_url.split(';')
                input_data['images'] = [{'url': url.strip()} for url in urls if url.strip()]
            else:
                input_data['url'] = image_url
        
        # Process based on group number
        if group_number == 6:
            print("=== GROUP 6 CONFIRMED: Processing COLOR section ===")
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif group_number in [7, 8]:
            print(f"=== GROUP {group_number} CONFIRMED: Processing Text-only section ===")
            detail_page, section_type = process_text_section(input_data, group_number)
            page_type = f"text_section_{section_type}"
            
        elif group_number in [1, 2]:
            print(f"=== GROUP {group_number} CONFIRMED: Processing Individual image ===")
            detail_page = process_single_image(input_data, group_number)
            page_type = "individual"
            
        elif group_number in [3, 4, 5]:
            print(f"=== GROUP {group_number} CONFIRMED: Processing CLEAN Combined images ===")
            if 'images' not in input_data or not isinstance(input_data['images'], list):
                input_data['images'] = [input_data]
            
            # CRITICAL FIX FOR GROUP 5
            if group_number == 5:
                # Always pass the full input_data to handle image7 and image8
                detail_page = process_clean_combined_images(input_data.get('images', []), group_number, input_data)
            else:
                detail_page = process_clean_combined_images(input_data['images'], group_number, input_data)
            
            page_type = "clean_combined"
        
        else:
            raise ValueError(f"Invalid group number: {group_number}")
        
        # Convert to base64
        buffered = BytesIO()
        detail_page.save(buffered, format="PNG", optimize=True)
        detail_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        detail_base64_no_padding = detail_base64.rstrip('=')
        
        print(f"Detail page created: {detail_page.size}")
        print(f"Base64 length: {len(detail_base64_no_padding)} chars")
        
        # Prepare metadata
        metadata = {
            "enhanced_image": detail_base64_no_padding,
            "status": "success",
            "page_type": page_type,
            "page_number": route_number if route_number > 0 else group_number,
            "route_number": route_number,
            "dimensions": {
                "width": detail_page.width,
                "height": detail_page.height
            },
            "version": "V119_FIXED_TEXT_COLOR",
            "image_count": len(input_data.get('images', [input_data])),
            "processing_time": "calculated_later",
            "detected_group_method": "route_number_priority",
            "font_status": "korean_font_available" if os.path.exists('/tmp/NanumMyeongjo.ttf') else "fallback_font"
        }
        
        # Send to webhook if configured
        file_name = f"detail_group_{route_number if route_number > 0 else group_number}.png"
        webhook_result = send_to_webhook(detail_base64_no_padding, "detail", file_name, route_number, metadata)
        
        # Return response (Make.com format)
        return {
            "output": metadata
        }
        
    except Exception as e:
        error_msg = f"Detail page creation failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "error",
                "traceback": traceback.format_exc(),
                "version": "V119_FIXED_TEXT_COLOR"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("Starting Detail Page Handler V119 - FIXED TEXT & COLOR...")
    runpod.serverless.start({"handler": handler})
