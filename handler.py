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
    """V134 ULTRA FIX: 모든 가능한 인코딩 오류를 방지"""
    if not text:
        return ""
    
    try:
        # 1단계: 문자열로 변환
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        else:
            text = str(text)
        
        # 2단계: 문제가 되는 문자들을 안전한 대체 문자로 변경
        # Latin-1에서 문제가 되는 한글 문자들을 처리
        safe_replacements = {
            '가': 'ga', '나': 'na', '다': 'da', '라': 'ra', '마': 'ma',
            '바': 'ba', '사': 'sa', '아': 'a', '자': 'ja', '차': 'cha',
            '카': 'ka', '타': 'ta', '파': 'pa', '하': 'ha',
            '고': 'go', '노': 'no', '도': 'do', '로': 'ro', '모': 'mo',
            '보': 'bo', '소': 'so', '오': 'o', '조': 'jo', '초': 'cho',
            '코': 'ko', '토': 'to', '포': 'po', '호': 'ho',
            '구': 'gu', '누': 'nu', '두': 'du', '루': 'ru', '무': 'mu',
            '부': 'bu', '수': 'su', '우': 'u', '주': 'ju', '추': 'chu',
            '쿠': 'ku', '투': 'tu', '푸': 'pu', '후': 'hu',
            '급': 'geup', '런': 'leon', '스': 'seu', '러': 'leo', '운': 'un',
            '텍': 'tek', '스': 'seu', '처': 'cheo', '와': 'wa', '균': 'gyun',
            '형': 'hyeong', '잡': 'jab', '힌': 'hin', '디': 'di', '테': 'te',
            '일': 'il', '이': 'i', '감': 'gam', '성': 'seong', '의': 'ui',
            '깊': 'gip', '더': 'deo', '하': 'ha', '는': 'neun', '커': 'keo',
            '플': 'peul', '링': 'ring', '입': 'ip', '니': 'ni', '섬': 'seom',
            '세': 'se', '한': 'han', '연': 'yeon', '결': 'gyeol', '을': 'eul',
            '느': 'neu', '끼': 'kki', '고': 'go', '싶': 'sip', '은': 'eun',
            '에': 'e', '게': 'ge', '추': 'chu', '천': 'cheon', '드': 'deu',
            '립': 'rip', '다': 'da'
        }
        
        # 실제로는 한글을 보존하되, JSON 안전하게 처리
        # 3단계: UTF-8로 강제 인코딩/디코딩
        try:
            text_bytes = text.encode('utf-8')
            text = text_bytes.decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            # 문제가 있는 문자들을 ASCII로 변환
            text = text.encode('ascii', errors='ignore').decode('ascii')
        
        # 4단계: JSON 안전성 테스트
        try:
            json.dumps(text, ensure_ascii=False)
        except (TypeError, ValueError):
            # JSON 직렬화가 실패하면 ASCII만 사용
            text = ''.join(c for c in text if ord(c) < 128)
        
        return text.strip()
        
    except Exception as e:
        print(f"V134 WARNING: Text encoding failed: {e}")
        return "text_encoding_failed"

def clean_claude_text(text):
    """V134 FIXED: Clean text for safe JSON encoding while preserving Korean characters"""
    if not text:
        return ""
    
    # V134 CRITICAL: Use ultra safe encoding first
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
    
    print(f"V134 Cleaned text preview: {text[:100]}...")
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
            
        print("V134: Starting Replicate background removal...")
        
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
        
        print("V134: Replicate background removal completed successfully")
        return result_img
        
    except Exception as e:
        print(f"V134: Error with Replicate API: {e}")
        print("V134: Falling back to local method...")
        return extract_ring_local_fallback(img)

def extract_ring_local_fallback(img):
    """Local fallback method for background removal"""
    print("V134: Using local fallback for background removal")
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
    """V134 MAJOR FIX: Create COLOR section with ACTUAL ring image - FORCE USE REAL IMAGE"""
    print("V134: === CREATING COLOR SECTION WITH REAL WEDDING RING ===")
    
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
                label_font = ImageFont.truetype(font_path, 24)
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
    
    # Color definitions
    colors = [
        ("yellow", (255, 215, 0)),      # #FFD700
        ("rose", (255, 192, 203)),      # #FFC0CB  
        ("white", (229, 229, 229)),     # #E5E5E5
        ("antique", (212, 175, 55))     # #D4AF37
    ]
    
    # Layout settings
    container_size = 300
    h_spacing = 100
    v_spacing = 400
    
    grid_width = 2 * container_size + h_spacing
    start_x = (width - grid_width) // 2
    start_y = 200
    
    # V134 CRITICAL FIX: FORCE PROCESS ACTUAL RING IMAGE
    processed_ring = None
    if ring_image:
        try:
            print("V134 CRITICAL: Processing actual ring image with background removal...")
            print(f"V134: Input ring image size: {ring_image.size}, mode: {ring_image.mode}")
            
            # FORCE background removal
            processed_ring = extract_ring_with_replicate(ring_image)
            print(f"V134 SUCCESS: Ring extraction completed, size: {processed_ring.size}")
            
            # Additional verification that we have a valid ring image
            if processed_ring and processed_ring.size[0] > 0 and processed_ring.size[1] > 0:
                print("V134 VERIFIED: Ring image is valid and ready for color application")
            else:
                print("V134 ERROR: Ring extraction returned invalid image")
                processed_ring = None
                
        except Exception as e:
            print(f"V134 ERROR: Ring processing failed completely: {e}")
            print(f"V134 TRACEBACK: {traceback.format_exc()}")
            processed_ring = None
    else:
        print("V134 WARNING: No ring image provided to color section")
    
    # Create color variants
    for i, (name, color_rgb) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (container_size + h_spacing)
        y = start_y + row * v_spacing
        
        # White container background
        container = Image.new('RGBA', (container_size, container_size), (255, 255, 255, 255))
        container_draw = ImageDraw.Draw(container)
        
        # V134 FORCE: Use actual ring image if available
        if processed_ring and processed_ring.size[0] > 0:
            try:
                print(f"V134: Applying {name} color to actual ring image...")
                
                # Resize the ring to fit in container with proper aspect ratio
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
                
                # Resize with high quality
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                ring_resized = processed_ring.resize((new_width, new_height), resample_filter)
                
                # Apply color filter to the ring
                color_multipliers = [
                    color_rgb[0] / 255.0,
                    color_rgb[1] / 255.0,
                    color_rgb[2] / 255.0
                ]
                colored_ring = apply_metal_color_filter(ring_resized, color_multipliers)
                
                # Center the ring in the container
                ring_x = (container_size - new_width) // 2
                ring_y = (container_size - new_height) // 2
                
                # Paste the colored ring with alpha compositing
                if colored_ring.mode == 'RGBA':
                    container.paste(colored_ring, (ring_x, ring_y), colored_ring)
                else:
                    container.paste(colored_ring, (ring_x, ring_y))
                
                print(f"V134 SUCCESS: Applied {name} color to actual ring image")
                
            except Exception as e:
                print(f"V134 ERROR: Failed to apply {name} color to ring: {e}")
                print(f"V134 TRACEBACK: {traceback.format_exc()}")
                # Fallback to drawing circles
                draw_fallback_rings(container_draw, container_size, color_rgb)
        else:
            print(f"V134 FALLBACK: Using circle graphics for {name} (no ring image)")
            # Fallback: Draw ring graphics
            draw_fallback_rings(container_draw, container_size, color_rgb)
        
        # Add shadow
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
    
    print("V134: Color section creation completed")
    return section_img

def draw_fallback_rings(container_draw, container_size, color_rgb):
    """Draw fallback ring graphics if actual ring image processing fails"""
    print("V134: Drawing fallback ring graphics")
    
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
    
    # Add diamonds
    diamond_size = 8
    diamond_y = left_ring_center_y - left_ring_radius + left_ring_thickness//2
    container_draw.polygon([
        (left_ring_center_x, diamond_y - diamond_size),
        (left_ring_center_x + diamond_size//2, diamond_y - diamond_size//2),
        (left_ring_center_x, diamond_y),
        (left_ring_center_x - diamond_size//2, diamond_y - diamond_size//2)
    ], fill=(255, 255, 255), outline=(180, 180, 180))
    
    small_diamond_size = 6
    small_diamond_y = right_ring_center_y - right_ring_radius + right_ring_thickness//2
    container_draw.polygon([
        (right_ring_center_x, small_diamond_y - small_diamond_size),
        (right_ring_center_x + small_diamond_size//2, small_diamond_y - small_diamond_size//2),
        (right_ring_center_x, small_diamond_y),
        (right_ring_center_x - small_diamond_size//2, small_diamond_y - small_diamond_size//2)
    ], fill=(255, 255, 255), outline=(180, 180, 180))

def create_ai_generated_md_talk(claude_text, width=FIXED_WIDTH):
    """V134: Create MD Talk text section - ULTRA ENCODING SAFE"""
    print("V134: Creating MD TALK text section")
    
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
        # V134 ULTRA SAFE: Clean text with maximum safety
        cleaned_text = clean_claude_text(claude_text)
        
        # Remove title prefixes
        cleaned_text = re.sub(r'^(MD TALK|md talk|MD talk|엠디톡)\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        # Break text into lines
        words = cleaned_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 100:
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
    
    y_pos = 250
    line_height = 50
    
    for line in lines:
        # V134 ULTRA SAFE: Ensure each line is encoding-safe
        safe_line = ultra_safe_string_encode(line)
        line_width, _ = get_text_dimensions(draw, safe_line, body_font)
        draw.text((width//2 - line_width//2, y_pos), safe_line, font=body_font, fill=(80, 80, 80))
        y_pos += line_height
    
    print("V134: MD TALK section completed")
    return section_img

def create_ai_generated_design_point(claude_text, width=FIXED_WIDTH):
    """V134: Create Design Point text section - ULTRA ENCODING SAFE"""
    print("V134: Creating DESIGN POINT text section")
    
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
        # V134 ULTRA SAFE: Clean text with maximum safety
        cleaned_text = clean_claude_text(claude_text)
        
        # Remove title prefixes
        cleaned_text = re.sub(r'^(DESIGN POINT|design point|Design Point|디자인포인트|디자인 포인트)\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        # Break text into lines
        words = cleaned_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 100:
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
    
    y_pos = 250
    line_height = 55
    
    for line in lines:
        # V134 ULTRA SAFE: Ensure each line is encoding-safe
        safe_line = ultra_safe_string_encode(line)
        line_width, _ = get_text_dimensions(draw, safe_line, body_font)
        draw.text((width//2 - line_width//2, y_pos), safe_line, font=body_font, fill=(80, 80, 80))
        y_pos += line_height
    
    print("V134: DESIGN POINT section completed")
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
    """Process single image (groups 1, 2) - V134 NO PAGE NUMBERING"""
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
    
    # V134: NO PAGE NUMBERING TEXT
    
    return detail_page

def process_clean_combined_images(images_data, group_number, input_data=None):
    """Process combined images WITHOUT text sections (groups 3, 4, 5) - V134 NO PAGE NUMBERING"""
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
    
    # V134: NO PAGE NUMBERING TEXT
    
    return detail_page

def process_color_section(input_data):
    """V134 MAJOR FIX: Process group 6 - COLOR section with ring image - FORCE REAL IMAGE"""
    print("V134: === PROCESSING GROUP 6 COLOR SECTION ===")
    
    # Multiple ways to find the image for color section
    img = None
    
    # Method 1: Check for image9 key
    if 'image9' in input_data:
        print("V134: Found image9 key for COLOR section")
        img_data = {'url': input_data['image9']}
        img = get_image_from_input(img_data)
    # Method 2: Check for group6 key
    elif 'group6' in input_data:
        print("V134: Found group6 key for COLOR section")
        img_data = {'url': input_data['group6']}
        img = get_image_from_input(img_data)
    # Method 3: Check for image6 key
    elif 'image6' in input_data:
        print("V134: Found image6 key for COLOR section")
        img_data = {'url': input_data['image6']}
        img = get_image_from_input(img_data)
    # Method 4: Check standard image input
    else:
        print("V134: Using standard image input for COLOR section")
        try:
            img = get_image_from_input(input_data)
        except:
            print("V134: No image found for COLOR section")
            img = None
    
    if img:
        print(f"V134 SUCCESS: Ring image for color section: {img.size}, mode: {img.mode}")
    else:
        print("V134 WARNING: No ring image found, creating without ring image")
    
    # V134 CRITICAL: FORCE pass the actual ring image
    color_section = create_color_options_section(ring_image=img)
    
    if img:
        img.close()
    
    print("V134: Color section created successfully")
    return color_section

def process_text_section(input_data, group_number):
    """V134 ULTRA FIX: Process text sections with MAXIMUM encoding safety"""
    print(f"V134: Processing text section for group {group_number}")
    
    # Check for base64 encoded text first
    claude_text_base64 = input_data.get('claude_text_base64', '')
    claude_text = ""
    
    if claude_text_base64:
        try:
            print("V134: Found base64 encoded claude_text")
            # Add padding if needed
            missing_padding = len(claude_text_base64) % 4
            if missing_padding:
                claude_text_base64 += '=' * (4 - missing_padding)
            
            # V134 ULTRA SAFE: Multiple decoding attempts with ultimate fallback
            try:
                # First: Try direct UTF-8 decode
                decoded_bytes = base64.b64decode(claude_text_base64)
                claude_text = decoded_bytes.decode('utf-8')
                print("V134 SUCCESS: Direct UTF-8 decode successful")
            except UnicodeDecodeError:
                try:
                    # Second: Try UTF-8 with error replacement
                    claude_text = decoded_bytes.decode('utf-8', errors='replace')
                    print("V134 SUCCESS: UTF-8 with replacement successful")
                except:
                    try:
                        # Third: Try latin-1 then convert
                        temp_text = decoded_bytes.decode('latin-1')
                        # Convert problematic characters to safe equivalents
                        claude_text = temp_text.encode('ascii', errors='ignore').decode('ascii')
                        print("V134 SUCCESS: Latin-1 to ASCII conversion successful")
                    except:
                        # Ultimate fallback: Use placeholder
                        claude_text = "Text encoding error - using fallback content"
                        print("V134 FALLBACK: Using placeholder text due to encoding failure")
        except Exception as e:
            print(f"V134 ERROR: Base64 decoding failed: {e}")
            claude_text = ""
    else:
        # Fallback to regular text fields
        claude_text = (input_data.get('claude_text') or 
                      input_data.get('text_content') or 
                      input_data.get('ai_text') or 
                      input_data.get('generated_text') or '')
    
    # V134 ULTRA SAFE: Clean the text with maximum safety
    if claude_text:
        claude_text = ultra_safe_string_encode(claude_text)
        claude_text = clean_claude_text(claude_text)
    
    text_type = (input_data.get('text_type') or 
                input_data.get('section_type') or '')
    
    print(f"V134: Text type: {text_type}")
    print(f"V134: Group number: {group_number}")
    print(f"V134: Cleaned text preview: {claude_text[:100] if claude_text else 'No text'}...")
    
    # Create text section based on group number
    if group_number == 7:
        print("V134: GROUP 7 CONFIRMED - Creating MD TALK section")
        text_section = create_ai_generated_md_talk(claude_text)
        section_type = "md_talk"
    elif group_number == 8:
        print("V134: GROUP 8 CONFIRMED - Creating DESIGN POINT section")
        text_section = create_ai_generated_design_point(claude_text)
        section_type = "design_point"
    else:
        print(f"V134 WARNING: Unexpected group number {group_number} for text section")
        if 'md' in text_type.lower():
            text_section = create_ai_generated_md_talk(claude_text)
            section_type = "md_talk"
        else:
            text_section = create_ai_generated_design_point(claude_text)
            section_type = "design_point"
    
    print(f"V134: Text section created successfully: {section_type}")
    return text_section, section_type

def send_to_webhook(image_base64, handler_type, file_name, route_number=0, metadata={}):
    """V134 ULTRA SAFE: Send results to webhook with maximum encoding safety"""
    try:
        if not WEBHOOK_URL:
            print("WARNING: Webhook URL not configured, skipping webhook send")
            return None
        
        # V134 ULTRA SAFE: Ensure all metadata is encoding-safe
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
        
        print(f"V134: Sending to webhook: {handler_type} for {file_name}")
        
        # V134 ULTRA SAFE: Test JSON serialization before sending
        try:
            test_json = json.dumps(webhook_data, ensure_ascii=True)  # Force ASCII to avoid issues
            print("V134: JSON serialization test passed (ASCII mode)")
        except Exception as json_err:
            print(f"V134: JSON serialization failed: {json_err}")
            # Don't send webhook if JSON fails
            return None
        
        response = requests.post(
            WEBHOOK_URL,
            json=webhook_data,
            headers={'Content-Type': 'application/json; charset=utf-8'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"V134: Webhook success: {result}")
            return result
        else:
            print(f"V134: Webhook failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"V134: Webhook error: {str(e)}")
        return None

def detect_group_number_from_input(input_data):
    """CORRECT group number detection - V134 FIXED"""
    print("=== GROUP NUMBER DETECTION V134 - FIXED VERSION ===")
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
    """V134 ULTRA FIXED: Main handler with maximum safety and real ring images"""
    try:
        print(f"=== V134 Detail Page Handler - ULTRA FIXED VERSION ===")
        
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
                # V134 CRITICAL: Ensure GROUP 6 gets the ring image
                if 'image9' in input_data:
                    input_data['url'] = input_data['image9']
                    print("V134 GROUP 6: Using image9")
                elif 'image6' in input_data:
                    input_data['url'] = input_data['image6']
                    print("V134 GROUP 6: Using image6")
                elif 'group6' in input_data:
                    input_data['url'] = input_data['group6']
                    print("V134 GROUP 6: Using group6")
                else:
                    print("V134 WARNING: GROUP 6 but no specific image found")
            
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
            print("V134: === Processing GROUP 6: COLOR section ===")
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif group_number == 7:
            print("V134: === Processing GROUP 7: MD TALK text section ===")
            detail_page, section_type = process_text_section(input_data, 7)
            page_type = f"text_section_{section_type}"
            
        elif group_number == 8:
            print("V134: === Processing GROUP 8: DESIGN POINT text section ===")
            detail_page, section_type = process_text_section(input_data, 8)
            page_type = f"text_section_{section_type}"
            
        elif group_number in [1, 2]:
            print(f"V134: === Processing GROUP {group_number}: Individual image ===")
            detail_page = process_single_image(input_data, group_number)
            page_type = "individual"
            
        elif group_number in [3, 4, 5]:
            print(f"V134: === Processing GROUP {group_number}: Combined images ===")
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
        
        print(f"V134: Detail page created: {detail_page.size}")
        print(f"V134: Base64 length: {len(detail_base64_no_padding)} chars")
        
        # V134 ULTRA SAFE: Metadata with encoding safety
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
            "version": "V134_ULTRA_FIXED",
            "image_count": len(input_data.get('images', [input_data])),
            "processing_time": "calculated_later",
            "font_status": "korean_font_available" if os.path.exists('/tmp/NanumMyeongjo.ttf') else "fallback_font"
        }
        
        # Send to webhook
        file_name = f"detail_group_{group_number}.png"
        webhook_result = send_to_webhook(detail_base64_no_padding, "detail", file_name, group_number, metadata)
        
        # Return response
        return {
            "output": metadata
        }
        
    except Exception as e:
        # V134 ULTRA SAFE: Error handling
        error_msg = ultra_safe_string_encode(f"Detail page creation failed: {str(e)}")
        print(f"V134 ERROR: {error_msg}")
        traceback_str = ultra_safe_string_encode(traceback.format_exc())
        print(f"V134 TRACEBACK: {traceback_str}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "error",
                "traceback": traceback_str,
                "version": "V134_ULTRA_FIXED"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("Starting Detail Page Handler V134 - ULTRA FIXED VERSION...")
    runpod.serverless.start({"handler": handler})
