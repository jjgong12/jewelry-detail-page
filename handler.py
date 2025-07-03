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
    """Download Korean font for text rendering - Enhanced version"""
    try:
        font_path = '/tmp/NanumMyeongjo.ttf'
        
        # Check if already exists
        if os.path.exists(font_path):
            print("Korean font already exists")
            return True
        
        # Try to download using requests instead of wget
        font_url = 'https://github.com/naver/nanumfont/raw/master/fonts/NanumMyeongjo/NanumMyeongjo.ttf'
        print(f"Downloading Korean font from: {font_url}")
        
        response = requests.get(font_url, timeout=30)
        if response.status_code == 200:
            with open(font_path, 'wb') as f:
                f.write(response.content)
            print("Korean font downloaded successfully using requests")
            return True
        else:
            print(f"Failed to download Korean font: HTTP {response.status_code}")
            
            # Fallback: try alternative URL
            alt_url = 'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumMyeongjo/NanumMyeongjo.ttf'
            print(f"Trying alternative URL: {alt_url}")
            response = requests.get(alt_url, timeout=30)
            if response.status_code == 200:
                with open(font_path, 'wb') as f:
                    f.write(response.content)
                print("Korean font downloaded successfully from CDN")
                return True
                
        return False
        
    except Exception as e:
        print(f"Error downloading Korean font: {str(e)}")
        return False

def clean_claude_text(text):
    """Enhanced Claude text cleaning to prevent JSON errors and Korean issues"""
    if not text:
        return ""
    
    # Convert to string and handle None
    text = str(text) if text is not None else ""
    
    # CRITICAL FIX: DO NOT decode unicode_escape for Korean text
    # This was causing Korean characters to break
    
    # Replace newlines and tabs with spaces
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    
    # Clean quotes without breaking Korean
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    
    # Remove markdown formatting
    text = text.replace('#', '')
    text = text.replace('*', '')
    text = text.replace('_', '')
    text = text.replace('`', '')
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Trim to safe length
    if len(text) > 500:
        text = text[:497] + "..."
    
    # Keep all printable characters including Korean
    # Do NOT filter by ord(char) >= 32 as this breaks Korean
    
    print(f"Cleaned text (first 100 chars): {text[:100]}...")
    return text

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
    """Create COLOR section with CSS-style ring design inspired by HTML"""
    section_height = 1000
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    label_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 72)
                label_font = ImageFont.truetype(font_path, 32)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Title
    title = "COLOR"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 80), title, font=title_font, fill=(60, 60, 60))
    
    # Color definitions
    colors = [
        ("Yellow Gold", "#FFD700", (255, 215, 0)),
        ("Rose Gold", "#F4C2C2", (244, 194, 194)),
        ("White Gold", "#E5E4E2", (229, 228, 226)),
        ("White", "#FFFFFF", (240, 240, 240))  # Slightly gray for visibility
    ]
    
    # Layout settings
    ring_size = 400
    h_spacing = 100
    v_spacing = 450
    
    grid_width = 2 * ring_size + h_spacing
    start_x = (width - grid_width) // 2
    start_y = 200
    
    for i, (name, color_hex, color_rgb) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (ring_size + h_spacing)
        y = start_y + row * v_spacing
        
        # Create ring display area
        ring_area = Image.new('RGBA', (ring_size, ring_size), (255, 255, 255, 255))
        ring_draw = ImageDraw.Draw(ring_area)
        
        # Draw CSS-style rings (inspired by HTML example)
        # Large ring (left)
        large_ring_size = 180
        large_ring_x = ring_size // 2 - 50
        large_ring_y = ring_size // 2 - 30
        ring_thickness = 30
        
        # Outer circle
        ring_draw.ellipse([
            large_ring_x - large_ring_size//2,
            large_ring_y - large_ring_size//2,
            large_ring_x + large_ring_size//2,
            large_ring_y + large_ring_size//2
        ], fill=color_rgb, outline=None)
        
        # Inner circle (to create ring shape)
        inner_size = large_ring_size - 2 * ring_thickness
        ring_draw.ellipse([
            large_ring_x - inner_size//2,
            large_ring_y - inner_size//2,
            large_ring_x + inner_size//2,
            large_ring_y + inner_size//2
        ], fill=(255, 255, 255), outline=None)
        
        # Small ring (right)
        small_ring_size = 140
        small_ring_x = ring_size // 2 + 60
        small_ring_y = ring_size // 2 + 40
        small_ring_thickness = 24
        
        # Outer circle
        ring_draw.ellipse([
            small_ring_x - small_ring_size//2,
            small_ring_y - small_ring_size//2,
            small_ring_x + small_ring_size//2,
            small_ring_y + small_ring_size//2
        ], fill=color_rgb, outline=None)
        
        # Inner circle
        inner_size_small = small_ring_size - 2 * small_ring_thickness
        ring_draw.ellipse([
            small_ring_x - inner_size_small//2,
            small_ring_y - inner_size_small//2,
            small_ring_x + inner_size_small//2,
            small_ring_y + inner_size_small//2
        ], fill=(255, 255, 255), outline=None)
        
        # Add diamond accents
        # Large ring diamond
        diamond_size = 12
        diamond_x = large_ring_x
        diamond_y = large_ring_y - large_ring_size//2 + ring_thickness//2
        ring_draw.polygon([
            (diamond_x, diamond_y - diamond_size//2),
            (diamond_x + diamond_size//2, diamond_y),
            (diamond_x, diamond_y + diamond_size//2),
            (diamond_x - diamond_size//2, diamond_y)
        ], fill=(255, 255, 255), outline=(200, 200, 200))
        
        # Small ring diamond
        small_diamond_size = 8
        small_diamond_x = small_ring_x
        small_diamond_y = small_ring_y - small_ring_size//2 + small_ring_thickness//2
        ring_draw.polygon([
            (small_diamond_x, small_diamond_y - small_diamond_size//2),
            (small_diamond_x + small_diamond_size//2, small_diamond_y),
            (small_diamond_x, small_diamond_y + small_diamond_size//2),
            (small_diamond_x - small_diamond_size//2, small_diamond_y)
        ], fill=(255, 255, 255), outline=(200, 200, 200))
        
        # Add subtle shadow effect
        shadow = Image.new('RGBA', (ring_size, ring_size), (255, 255, 255, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_offset = 5
        
        # Shadow for large ring
        shadow_draw.ellipse([
            large_ring_x - large_ring_size//2 + shadow_offset,
            large_ring_y - large_ring_size//2 + shadow_offset,
            large_ring_x + large_ring_size//2 + shadow_offset,
            large_ring_y + large_ring_size//2 + shadow_offset
        ], fill=(200, 200, 200, 50))
        
        # Shadow for small ring
        shadow_draw.ellipse([
            small_ring_x - small_ring_size//2 + shadow_offset,
            small_ring_y - small_ring_size//2 + shadow_offset,
            small_ring_x + small_ring_size//2 + shadow_offset,
            small_ring_y + small_ring_size//2 + shadow_offset
        ], fill=(200, 200, 200, 50))
        
        # Combine shadow and rings
        combined = Image.alpha_composite(shadow, ring_area)
        
        # Draw border
        border_draw = ImageDraw.Draw(combined)
        border_draw.rectangle([0, 0, ring_size-1, ring_size-1], outline=(230, 230, 230), width=1)
        
        # Paste to main image
        section_img.paste(combined, (x, y), combined)
        
        # Draw label
        label_width, _ = get_text_dimensions(draw, name, label_font)
        draw.text((x + ring_size//2 - label_width//2, y + ring_size + 30), 
                 name, font=label_font, fill=(80, 80, 80))
    
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
        
        # Split text into lines more naturally
        words = cleaned_text.split()
        lines = []
        current_line = []
        max_width = width - 200  # Margin
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
    else:
        lines = [
            "고급스러운 텍스처와 균형 잡힌 디테일이",
            "감성의 깊이를 더하는 커플링입니다.",
            "섬세한 연결을 느끼고 싶은 커플에게 추천드립니다."
        ]
    
    y_pos = 200
    line_height = 45
    
    for line in lines[:8]:  # Limit to 8 lines
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
        
        # Split text into lines more naturally
        words = cleaned_text.split()
        lines = []
        current_line = []
        max_width = width - 200  # Margin
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
    else:
        lines = [
            "리프링 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고",
            "여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일",
            "화려하면서도 고급스러운 반영영을 표현합니다",
            "메인 스톤이 두 반지를 하나의 결로 이어주는 상징이 됩니다"
        ]
    
    y_pos = 200
    line_height = 50
    
    for line in lines[:10]:  # Limit to 10 lines
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
    """Process group 6 - COLOR section with ring image (image 9 only)"""
    print("=== PROCESSING GROUP 6 COLOR SECTION ===")
    
    # CRITICAL FIX: For group 6, we need image 9
    if 'image9' in input_data:
        print("Found image9 key for COLOR section")
        img_data = {'url': input_data['image9']}
        img = get_image_from_input(img_data)
    else:
        img = get_image_from_input(input_data)
    
    print(f"Ring image for color section: {img.size}, mode: {img.mode}")
    
    color_section = create_color_options_section(ring_image=img)
    
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
    """Enhanced group number detection with better group 1/2 differentiation"""
    print("=== GROUP NUMBER DETECTION ENHANCED V117 ===")
    
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
    
    # Method 3: Check for specific image keys (image1, image2, etc.)
    for i in range(1, 9):
        key = f'image{i}'
        if key in input_data:
            print(f"Found {key} key, assuming group {i}")
            return i
    
    # Method 4: Check text_type for groups 7, 8
    text_type = input_data.get('text_type', '')
    if text_type == 'md_talk':
        print("Found md_talk text_type, assuming group 7")
        return 7
    elif text_type == 'design_point':
        print("Found design_point text_type, assuming group 8")
        return 8
    
    # Method 5: Check for Claude text presence (base64 or regular)
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
    
    # Method 6: Enhanced URL analysis
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
            
            # For single images without other indicators, we can't reliably
            # distinguish between group 1 and 2 without route_number
            print("Single URL, no route_number - cannot distinguish group 1 vs 2")
            return 1  # Default to group 1
    
    # Last resort
    print("WARNING: Could not reliably detect group number")
    print(f"Available keys in input: {list(input_data.keys())}")
    
    return 0

def handler(event):
    """Main handler for detail page creation - V117 with Korean Text Fix"""
    try:
        print(f"=== V117 Detail Page Handler - KOREAN TEXT FIX ===")
        
        # Download Korean font if not exists - Enhanced version
        if not os.path.exists('/tmp/NanumMyeongjo.ttf'):
            print("Korean font not found, downloading...")
            if not download_korean_font():
                print("WARNING: Failed to download Korean font, text may appear corrupted")
        else:
            print("Korean font already exists")
        
        # Get input data
        input_data = event.get('input', event)
        
        # DON'T pre-clean text fields here - let process_text_section handle it
        # This prevents double-cleaning which can cause issues
        
        print(f"Input keys: {list(input_data.keys())}")
        
        # Enhanced group number detection
        group_number = detect_group_number_from_input(input_data)
        route_number = input_data.get('route_number', group_number)
        
        print(f"FINAL: group_number={group_number}, route_number={route_number}")
        
        # CRITICAL: Always use route_number as the actual group number
        # This fixes the group 2 being detected as group 1 issue
        if route_number > 0 and route_number != group_number:
            print(f"OVERRIDE: Using route_number {route_number} instead of detected group {group_number}")
            group_number = route_number
        
        # Validate group number
        if group_number == 0:
            raise ValueError(f"Could not determine group number. Keys: {list(input_data.keys())}")
        
        if group_number < 1 or group_number > 8:
            raise ValueError(f"Invalid group number: {group_number}. Must be 1-8.")
        
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
            print("=== Processing Group 6: COLOR section (image 9) ===")
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif group_number in [7, 8]:
            print(f"=== Processing Group {group_number}: Text-only section ===")
            detail_page, section_type = process_text_section(input_data, group_number)
            page_type = f"text_section_{section_type}"
            
        elif group_number in [1, 2]:
            print(f"=== Processing Group {group_number}: Individual image ===")
            detail_page = process_single_image(input_data, group_number)
            page_type = "individual"
            
        elif group_number in [3, 4, 5]:
            print(f"=== Processing Group {group_number}: CLEAN Combined images ===")
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
        
        # Prepare metadata - use route_number for accurate page numbering
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
            "version": "V117_KOREAN_TEXT_FIX",
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
                "version": "V117_KOREAN_TEXT_FIX"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("Starting Detail Page Handler V117 - KOREAN TEXT FIX...")
    runpod.serverless.start({"handler": handler})
