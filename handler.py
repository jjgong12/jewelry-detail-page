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

class FontDownloadError(Exception):
    """Custom exception for font download failures"""
    pass

def download_korean_font():
    """Download Korean font for text rendering with proper error handling"""
    font_path = '/tmp/NanumMyeongjo.ttf'
    
    # Check if already exists and valid
    if os.path.exists(font_path):
        try:
            # Try to load it to verify it's valid
            test_font = ImageFont.truetype(font_path, 20)
            print("Korean font already exists and is valid")
            return True
        except (IOError, OSError) as e:
            print(f"Korean font exists but is corrupted: {str(e)}")
            try:
                os.remove(font_path)
            except OSError as remove_error:
                print(f"Failed to remove corrupted font: {str(remove_error)}")
    
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
            response.raise_for_status()
            
            if len(response.content) < 100000:  # Font should be > 100KB
                print(f"Downloaded font too small ({len(response.content)} bytes)")
                continue
            
            with open(font_path, 'wb') as f:
                f.write(response.content)
            
            # Verify the font works
            test_font = ImageFont.truetype(font_path, 20)
            print(f"Korean font downloaded successfully from {url}")
            return True
            
        except requests.RequestException as e:
            print(f"Network error downloading from {url}: {str(e)}")
            continue
        except (IOError, OSError) as e:
            print(f"File system error with {url}: {str(e)}")
            continue
        except Exception as e:
            print(f"Unexpected error with {url}: {str(e)}")
            continue
    
    raise FontDownloadError("Failed to download Korean font from all sources")

def clean_claude_text(text):
    """Clean text for safe JSON encoding while preserving Korean characters"""
    if not text:
        return ""
    
    # Convert to string if needed
    text = str(text) if text is not None else ""
    
    # CRITICAL: Never use decode('unicode_escape') - it destroys Korean text!
    
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
    
    # NO LENGTH LIMIT - Trust Claude!
    # if len(text) > 500:
    #     text = text[:497] + "..."
    
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

class BackgroundRemovalError(Exception):
    """Custom exception for background removal failures"""
    pass

def extract_ring_with_replicate(img):
    """Extract ring from background using Replicate API with proper error handling"""
    if not REPLICATE_AVAILABLE:
        print("Replicate not available, using local fallback")
        return extract_ring_local_fallback(img)
        
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Replicate API token not found, using local fallback")
        return extract_ring_local_fallback(img)
        
    try:
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
        
        if not output:
            raise BackgroundRemovalError("Replicate API returned empty result")
        
        response = requests.get(output, timeout=30)
        response.raise_for_status()
        
        result_img = Image.open(BytesIO(response.content))
        
        if result_img.mode != 'RGBA':
            result_img = result_img.convert('RGBA')
        
        print("Replicate background removal completed successfully")
        return result_img
        
    except requests.RequestException as e:
        print(f"Network error with Replicate API: {str(e)}")
        return extract_ring_local_fallback(img)
    except Exception as e:
        print(f"Unexpected error with Replicate API: {str(e)}")
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
    section_height = 800  # Increased height for more text
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
        
        # Remove MD TALK or md talk from the beginning of content
        cleaned_text = re.sub(r'^(MD TALK|md talk|MD talk|엠디톡)\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        # NO CHARACTER LIMIT - Trust Claude! Break naturally at spaces
        words = cleaned_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Test if adding this word would exceed visual width (not character count)
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 100:  # Leave 50px margin on each side
                if current_line:
                    lines.append(current_line.strip())
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line.strip())
        
        # NO LINE LIMIT - Use all lines Claude provides
        # lines = lines[:3]  # REMOVED
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
    section_height = 900  # Increased height for more text
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
        
        # Remove DESIGN POINT or design point from the beginning of content
        cleaned_text = re.sub(r'^(DESIGN POINT|design point|Design Point|디자인포인트|디자인 포인트)\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        # NO CHARACTER LIMIT - Trust Claude! Break naturally at visual width
        words = cleaned_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            # Test if adding this word would exceed visual width
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 100:  # Leave 50px margin on each side
                if current_line:
                    lines.append(current_line.strip())
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line.strip())
        
        # NO LINE LIMIT - Use all lines Claude provides
        # lines = lines[:4]  # REMOVED
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

class GoogleDriveDownloadError(Exception):
    """Custom exception for Google Drive download failures"""
    pass

def download_image_from_google_drive(url):
    """Download image from Google Drive URL with proper error handling"""
    print(f"Processing Google Drive URL: {url[:100]}...")
    
    file_id = extract_file_id_from_url(url)
    if not file_id:
        raise GoogleDriveDownloadError(f"Could not extract file ID from URL: {url}")
    
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
    last_error = None
    
    for download_url in download_urls:
        try:
            print(f"Trying: {download_url}")
            response = session.get(download_url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            content_length = len(response.content)
            
            if 'image' in content_type and content_length > 1000:
                img = Image.open(BytesIO(response.content))
                print(f"Successfully downloaded image: {img.size}")
                return img
            else:
                print(f"Invalid response: content-type={content_type}, size={content_length}")
                continue
                
        except requests.RequestException as e:
            print(f"Network error with URL: {download_url}, Error: {str(e)}")
            last_error = e
            continue
        except (IOError, OSError) as e:
            print(f"Image processing error with URL: {download_url}, Error: {str(e)}")
            last_error = e
            continue
        except Exception as e:
            print(f"Unexpected error with URL: {download_url}, Error: {str(e)}")
            last_error = e
            continue
    
    raise GoogleDriveDownloadError(f"Failed to download from all URLs. Last error: {str(last_error)}")

def validate_input_data(input_data):
    """Validate input data structure and required fields"""
    if not isinstance(input_data, dict):
        raise ValueError("Input data must be a dictionary")
    
    # Check for required fields based on operation type
    has_image_url = any(key in input_data for key in ['image_url', 'imageUrl', 'url', 'webContentLink', 'image'])
    has_image_base64 = any(key in input_data for key in ['image_base64', 'base64', 'image_data', 'enhanced_image'])
    has_image_keys = any(f'image{i}' in input_data for i in range(1, 10))
    
    if not (has_image_url or has_image_base64 or has_image_keys):
        raise ValueError("No valid image source found in input data")
    
    return True

def validate_url(url):
    """Validate URL format and allowed domains"""
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")
    
    # Basic URL validation
    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")
    
    # Check for allowed domains
    allowed_domains = ['drive.google.com', 'docs.google.com', 'googleapis.com']
    if not any(domain in url for domain in allowed_domains):
        # For non-Google domains, add basic validation
        if len(url) > 2048:  # Maximum URL length
            raise ValueError("URL too long")
    
    return True

def validate_base64_data(base64_data):
    """Validate base64 image data"""
    if not base64_data or not isinstance(base64_data, str):
        raise ValueError("Base64 data must be a non-empty string")
    
    # Remove data URL prefix if present
    if base64_data.startswith('data:'):
        base64_data = base64_data.split(',')[1]
    
    # Check base64 format
    try:
        decoded = base64.b64decode(base64_data + '==')  # Add padding for test
        if len(decoded) < 100:  # Minimum viable image size
            raise ValueError("Base64 data too small to be a valid image")
    except Exception:
        raise ValueError("Invalid base64 format")
    
    return base64_data

def get_image_from_input(input_data):
    """Get image from URL or base64 with proper validation"""
    # Validate input data structure
    validate_input_data(input_data)
    
    # Try to get image URL first
    image_url = None
    for key in ['image_url', 'imageUrl', 'url', 'webContentLink', 'image']:
        if key in input_data and input_data[key]:
            image_url = str(input_data[key]).strip()
            break
    
    if image_url:
        print(f"Found image URL: {image_url[:100]}...")
        validate_url(image_url)
        
        try:
            if 'drive.google.com' in image_url or 'docs.google.com' in image_url:
                return download_image_from_google_drive(image_url)
            else:
                response = requests.get(image_url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; ImageProcessor/1.0)'
                })
                response.raise_for_status()
                
                # Validate response content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise ValueError(f"Invalid content type: {content_type}")
                
                return Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            raise ValueError(f"Failed to download image from URL: {str(e)}")
    
    # Try to get base64 data
    image_base64 = None
    for key in ['image_base64', 'base64', 'image_data', 'enhanced_image']:
        if key in input_data and input_data[key]:
            image_base64 = str(input_data[key]).strip()
            break
    
    if image_base64:
        print(f"Using base64 data, length: {len(image_base64)}")
        validated_base64 = validate_base64_data(image_base64)
        
        # Add padding if needed
        missing_padding = len(validated_base64) % 4
        if missing_padding:
            validated_base64 += '=' * (4 - missing_padding)
        
        try:
            image_data = base64.b64decode(validated_base64)
            return Image.open(BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image data: {str(e)}")
    
    raise ValueError("No valid image URL or base64 data found in input")

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
    # FIXED: Use group_number directly, not route_number
    page_text = f"- {group_number} -"
    
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
        
        # Check if images_data already has 2 images (from 'images' array)
        if len(images_data) >= 2:
            print(f"Using first 2 images from images_data (total: {len(images_data)})")
            images_data = images_data[:2]
        # Otherwise, try to find image7 and image8
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
    
    draw = ImageDraw.Draw(detail_page)
    
    # FIXED: Use group_number directly for page text
    if group_number == 5:
        page_text = f"- Gallery 7-8 -"
    else:
        page_text = f"- {group_number} -"
    
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
    
    print(f"Text type from input: {text_type}")
    print(f"Group number: {group_number}")
    print(f"Cleaned Claude text preview: {claude_text[:100] if claude_text else 'No text provided'}...")
    
    # CRITICAL FIX: Force correct text section based on route_number/group_number
    # Route 7 = MD TALK, Route 8 = DESIGN POINT
    if group_number == 7:
        print("GROUP 7 CONFIRMED - Creating MD TALK section")
        # Check if the text content looks like DESIGN POINT content
        if any(keyword in claude_text.lower() for keyword in ['텍스처', '파베', '밀그레인', '리프링']):
            print("WARNING: Text content seems like DESIGN POINT but group is 7 (MD TALK)")
            print("Forcing MD TALK creation anyway based on group number")
        text_section = create_ai_generated_md_talk(claude_text)
        section_type = "md_talk"
    elif group_number == 8:
        print("GROUP 8 CONFIRMED - Creating DESIGN POINT section")
        # Check if the text content looks like MD TALK content
        if any(keyword in claude_text.lower() for keyword in ['엔그레이빙', '감성', '커플링', '세련미']):
            print("WARNING: Text content seems like MD TALK but group is 8 (DESIGN POINT)")
            print("Forcing DESIGN POINT creation anyway based on group number")
        text_section = create_ai_generated_design_point(claude_text)
        section_type = "design_point"
    else:
        # This shouldn't happen if routing is correct
        print(f"WARNING: Unexpected group number {group_number} for text section")
        print("Falling back to text_type detection")
        if text_type == 'md_talk' or 'md' in text_type.lower():
            text_section = create_ai_generated_md_talk(claude_text)
            section_type = "md_talk"
        elif text_type == 'design_point' or 'design' in text_type.lower():
            text_section = create_ai_generated_design_point(claude_text)
            section_type = "design_point"
        else:
            # Ultimate fallback based on content analysis
            if any(keyword in claude_text.lower() for keyword in ['텍스처', '파베', '밀그레인', '디테일']):
                print("Content analysis suggests DESIGN POINT")
                text_section = create_ai_generated_design_point(claude_text)
                section_type = "design_point"
            else:
                print("Content analysis suggests MD TALK")
                text_section = create_ai_generated_md_talk(claude_text)
                section_type = "md_talk"
    
    return text_section, section_type

class WebhookError(Exception):
    """Custom exception for webhook failures"""
    pass

def send_to_webhook(image_base64, handler_type, file_name, route_number=0, metadata={}):
    """Send results to Google Apps Script webhook with proper error handling"""
    if not WEBHOOK_URL:
        print("WARNING: Webhook URL not configured, skipping webhook send")
        return None
    
    if not image_base64:
        raise WebhookError("No image data provided for webhook")
    
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
    
    try:
        print(f"Sending to webhook: {handler_type} for {file_name}")
        
        response = requests.post(
            WEBHOOK_URL,
            json=webhook_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        response.raise_for_status()
        
        try:
            result = response.json()
            print(f"Webhook success: {result}")
            return result
        except ValueError as e:
            print(f"Webhook returned invalid JSON: {str(e)}")
            return {"status": "success", "raw_response": response.text}
        
    except requests.RequestException as e:
        print(f"Network error sending webhook: {str(e)}")
        raise WebhookError(f"Failed to send webhook: {str(e)}")
    except Exception as e:
        print(f"Unexpected webhook error: {str(e)}")
        raise WebhookError(f"Unexpected webhook error: {str(e)}")

def safe_int_conversion(value, default=0):
    """Safely convert value to integer with default fallback"""
    if value is None:
        return default
    
    try:
        # Handle string numbers
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
            # Remove any non-numeric characters except minus
            cleaned = ''.join(c for c in value if c.isdigit() or c == '-')
            if not cleaned or cleaned == '-':
                return default
            return int(cleaned)
        
        # Handle numeric types
        if isinstance(value, (int, float)):
            return int(value)
        
        # Handle other types by string conversion
        return int(str(value))
        
    except (ValueError, TypeError):
        return default

def validate_group_number(group_number):
    """Validate group number is within acceptable range"""
    if not isinstance(group_number, int):
        raise ValueError(f"Group number must be an integer, got {type(group_number)}")
    
    if group_number < 1 or group_number > 8:
        raise ValueError(f"Group number must be between 1 and 8, got {group_number}")
    
    return group_number

def detect_group_number_from_input(input_data):
    """Detect group number with clear priority hierarchy and validation"""
    if not isinstance(input_data, dict):
        raise ValueError("Input data must be a dictionary")
    
    print("=== GROUP NUMBER DETECTION V129 - IMPROVED VERSION ===")
    print(f"Available keys: {sorted(input_data.keys())}")
    
    # PRIORITY 1: Explicit route_number (highest priority)
    route_number = safe_int_conversion(input_data.get('route_number'))
    if route_number > 0:
        print(f"Found explicit route_number: {route_number}")
        try:
            return validate_group_number(route_number)
        except ValueError as e:
            print(f"Invalid route_number: {e}")
            # Continue to other detection methods
    
    # PRIORITY 2: Explicit group_number
    group_number = safe_int_conversion(input_data.get('group_number'))
    if group_number > 0:
        print(f"Found explicit group_number: {group_number}")
        try:
            return validate_group_number(group_number)
        except ValueError as e:
            print(f"Invalid group_number: {e}")
            # Continue to other detection methods
    
    # PRIORITY 3: Special group indicators
    # Group 6 (COLOR) indicators
    if 'image9' in input_data:
        print("Found image9 key - GROUP 6 (COLOR)")
        return 6
    
    if any(key in input_data for key in ['group6', 'group_6', 'color_section']):
        print("Found group6/color_section key - GROUP 6")
        return 6
    
    # PRIORITY 4: Text type indicators (Groups 7, 8)
    text_type = str(input_data.get('text_type', '')).lower().strip()
    if text_type:
        if text_type in ['md_talk', 'mdtalk']:
            print(f"Found text_type '{text_type}' - GROUP 7")
            return 7
        elif text_type in ['design_point', 'designpoint']:
            print(f"Found text_type '{text_type}' - GROUP 8")
            return 8
    
    # PRIORITY 5: Content analysis for COLOR section
    color_keywords = ['color', 'colour', '컬러', '색상', '컬러섹션']
    input_text = str(input_data).lower()
    if any(keyword in input_text for keyword in color_keywords):
        print("Found COLOR keywords in input data - GROUP 6")
        return 6
    
    # PRIORITY 6: Image count analysis
    # Check for combined image field
    if 'image' in input_data and ';' in str(input_data['image']):
        urls = [u.strip() for u in str(input_data['image']).split(';') if u.strip()]
        url_count = len(urls)
        print(f"Found {url_count} URLs in 'image' field")
        
        if url_count == 1:
            return 1
        elif url_count == 2:
            return 3
        elif url_count >= 3:
            return 4
    
    # PRIORITY 7: Check for images array
    if 'images' in input_data and isinstance(input_data['images'], list):
        image_count = len(input_data['images'])
        print(f"Found {image_count} images in array")
        
        if image_count == 1:
            return 1
        elif image_count == 2:
            return 3
        elif image_count >= 3:
            return 4
    
    # PRIORITY 8: Individual image keys (fallback)
    image_keys = [key for key in input_data.keys() if key.startswith('image') and key[5:].isdigit()]
    if image_keys:
        # Sort by number and take the highest
        image_numbers = [int(key[5:]) for key in image_keys]
        max_image_num = max(image_numbers)
        print(f"Found image keys up to image{max_image_num}")
        
        # Special handling for specific patterns
        if max_image_num <= 8:
            return max_image_num
    
    # PRIORITY 9: Single image indicators
    single_image_keys = ['image_url', 'imageUrl', 'url', 'webContentLink', 'image_base64', 'base64']
    if any(key in input_data for key in single_image_keys):
        print("Found single image indicators - GROUP 1")
        return 1
    
    # If no group can be determined, raise an error
    raise ValueError("Could not determine group number from input data. Please provide route_number or group_number explicitly.")

def handler(event):
    """Main handler for detail page creation with improved error handling"""
    start_time = datetime.now()
    
    try:
        print(f"=== V129 Detail Page Handler - IMPROVED VERSION ===")
        
        # Download Korean font if not exists
        try:
            if not os.path.exists('/tmp/NanumMyeongjo.ttf'):
                print("Korean font not found, downloading...")
                download_korean_font()
            else:
                print("Korean font already exists")
        except FontDownloadError as e:
            print(f"Font download failed: {str(e)}")
            print("WARNING: Will use fallback font, text may appear corrupted")
        
        # Get input data
        input_data = event.get('input', event)
        
        print(f"=== INCOMING DATA ===")
        print(f"Keys: {sorted(input_data.keys())}")
        print(f"route_number: {input_data.get('route_number', 'NOT FOUND')}")
        
        # Group detection with improved error handling
        try:
            group_number = detect_group_number_from_input(input_data)
        except ValueError as e:
            print(f"Group detection failed: {str(e)}")
            raise ValueError(f"Could not determine processing group: {str(e)}")
        
        print(f"\n=== DETECTION RESULT ===")
        print(f"Detected group_number: {group_number}")
        print(f"Input route_number: {input_data.get('route_number', 'NONE')}")
        
        # Handle image input formats with validation
        if 'image' in input_data and input_data['image']:
            print(f"Found 'image' key with value: {str(input_data['image'])[:100]}...")
            image_data = input_data['image']
            
            if ';' in image_data:
                urls = [url.strip() for url in image_data.split(';') if url.strip()]
                input_data['images'] = [{'url': url} for url in urls]
                print(f"Converted 'image' to {len(input_data['images'])} images array")
            else:
                input_data['url'] = image_data
                print(f"Set single URL from 'image' key")
        
        if 'combined_urls' in input_data and input_data['combined_urls']:
            urls = [url.strip() for url in input_data['combined_urls'].split(';') if url.strip()]
            input_data['images'] = [{'url': url} for url in urls]
            print(f"Converted combined_urls to {len(input_data['images'])} images")
        
        # Process based on group number
        try:
            if group_number == 6:
                print("=== Processing GROUP 6: COLOR section ===")
                detail_page = process_color_section(input_data)
                page_type = "color_section"
                
            elif group_number == 7:
                print("=== Processing GROUP 7: MD TALK text section ===")
                detail_page, section_type = process_text_section(input_data, 7)
                page_type = f"text_section_{section_type}"
                
            elif group_number == 8:
                print("=== Processing GROUP 8: DESIGN POINT text section ===")
                detail_page, section_type = process_text_section(input_data, 8)
                page_type = f"text_section_{section_type}"
                
            elif group_number in [1, 2]:
                print(f"=== Processing GROUP {group_number}: Individual image ===")
                detail_page = process_single_image(input_data, group_number)
                page_type = "individual"
                
            elif group_number in [3, 4, 5]:
                print(f"=== Processing GROUP {group_number}: Combined images ===")
                if 'images' not in input_data or not isinstance(input_data['images'], list):
                    input_data['images'] = [input_data]
                
                if group_number == 5:
                    detail_page = process_clean_combined_images(input_data.get('images', []), group_number, input_data)
                else:
                    detail_page = process_clean_combined_images(input_data['images'], group_number, input_data)
                
                page_type = "clean_combined"
            
            else:
                raise ValueError(f"Invalid group number: {group_number}. Must be 1-8.")
                
        except Exception as e:
            print(f"Image processing failed: {str(e)}")
            raise ValueError(f"Failed to process group {group_number}: {str(e)}")
        
        # Convert to base64
        try:
            buffered = BytesIO()
            detail_page.save(buffered, format="PNG", optimize=True)
            detail_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            detail_base64_no_padding = detail_base64.rstrip('=')
            
            print(f"Detail page created: {detail_page.size}")
            print(f"Base64 length: {len(detail_base64_no_padding)} chars")
            
        except Exception as e:
            raise ValueError(f"Failed to encode result image: {str(e)}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create metadata
        metadata = {
            "enhanced_image": detail_base64_no_padding,
            "status": "success",
            "page_type": page_type,
            "page_number": group_number,
            "route_number": group_number,
            "actual_group": group_number,
            "dimensions": {
                "width": detail_page.width,
                "height": detail_page.height
            },
            "version": "V129_IMPROVED",
            "image_count": len(input_data.get('images', [input_data])),
            "processing_time": f"{processing_time:.2f}s",
            "font_status": "korean_font_available" if os.path.exists('/tmp/NanumMyeongjo.ttf') else "fallback_font"
        }
        
        # Send to webhook if configured
        try:
            file_name = f"detail_group_{group_number}.png"
            webhook_result = send_to_webhook(detail_base64_no_padding, "detail", file_name, group_number, metadata)
        except WebhookError as e:
            print(f"Webhook failed: {str(e)}")
            # Continue execution, webhook failure is not critical
        
        return {
            "output": metadata
        }
        
    except ValueError as e:
        # These are expected validation errors
        error_msg = str(e)
        print(f"Validation Error: {error_msg}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "validation_error",
                "version": "V129_IMPROVED",
                "processing_time": f"{(datetime.now() - start_time).total_seconds():.2f}s"
            }
        }
        
    except Exception as e:
        # Unexpected errors
        error_msg = f"Unexpected error: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "system_error",
                "traceback": traceback.format_exc(),
                "version": "V129_IMPROVED",
                "processing_time": f"{(datetime.now() - start_time).total_seconds():.2f}s"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("Starting Detail Page Handler V129 - FORCE FIX VERSION...")
    runpod.serverless.start({"handler": handler})
