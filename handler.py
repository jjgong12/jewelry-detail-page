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

# Webhook URL
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzOQ7SaTtIXRubvSNXNY53pphacVmJg_XKV5sIyOgxjpDykiWsAHN7ecKFHcygGFrYi/exec"

# FIXED WIDTH FOR ALL IMAGES
FIXED_WIDTH = 1200

def download_korean_font():
    """Download Korean font for text rendering"""
    try:
        font_path = '/tmp/NanumMyeongjo.ttf'
        
        if os.path.exists(font_path):
            try:
                test_font = ImageFont.truetype(font_path, 20)
                print("Korean font already exists and is valid")
                return True
            except:
                print("Korean font exists but is corrupted, re-downloading...")
                os.remove(font_path)
        
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
                
                if response.status_code == 200 and len(response.content) > 100000:
                    with open(font_path, 'wb') as f:
                        f.write(response.content)
                    
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

def korean_safe_string_encode(text):
    """V136 KOREAN SAFE: 100% 한국어 안전 문자열 처리 - NO LATIN-1 EVER!"""
    if not text:
        return ""
    
    try:
        # 1단계: 문자열 변환 (한국어 보존)
        if isinstance(text, bytes):
            try:
                # 오직 UTF-8만 사용! latin-1 절대 금지!
                text = text.decode('utf-8', errors='replace')
                print(f"V136: Bytes to UTF-8 successful: {text[:30]}...")
            except Exception as e:
                print(f"V136: Bytes decode failed: {e}")
                return "한국어 바이트 디코딩 실패"
        else:
            text = str(text)
        
        # 2단계: UTF-8 유효성 검사 (한국어 보존)
        try:
            # UTF-8 인코딩/디코딩 테스트
            text_bytes = text.encode('utf-8')
            text = text_bytes.decode('utf-8')
            
            # JSON 직렬화 테스트 (한국어 허용)
            json.dumps(text, ensure_ascii=False)
            
            print(f"V136 SUCCESS: 한국어 텍스트 완전 보존: {text[:30]}...")
            return text.strip()
            
        except (UnicodeEncodeError, UnicodeDecodeError, TypeError) as utf_error:
            print(f"V136: UTF-8 처리 중 오류: {utf_error}")
            
            # 3단계: 한국어 안전 복구 시도
            try:
                # replace로 복구 시도 (한국어 최대한 보존)
                safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
                
                # JSON 테스트
                json.dumps(safe_text, ensure_ascii=False)
                
                print(f"V136 REPAIRED: 한국어 텍스트 복구됨: {safe_text[:30]}...")
                return safe_text.strip()
                
            except Exception as repair_error:
                print(f"V136 WARNING: 복구 실패: {repair_error}")
                return "한국어 텍스트 복구 실패"
        
    except Exception as e:
        print(f"V136 ERROR: 완전 실패: {e}")
        return "한국어 처리 완전 실패"

def clean_korean_text(text):
    """V136 KOREAN SAFE: 한국어 텍스트 안전 정리"""
    if not text:
        return ""
    
    # V136: 한국어 안전 인코딩 먼저
    text = korean_safe_string_encode(text)
    
    # 이스케이프 시퀀스 정리
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\\\', '\\')
    
    # 실제 제어 문자 정리
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    
    # 따옴표 정리
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    
    # 마크다운 기호 제거
    for char in ['#', '*', '_', '`', '[', ']', '(', ')']:
        text = text.replace(char, '')
    
    # 공백 정리
    text = ' '.join(text.split())
    
    print(f"V136: 한국어 텍스트 정리 완료: {text[:50]}...")
    return text

def get_text_dimensions(draw, text, font):
    """Get text dimensions compatible with all PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def extract_ring_with_replicate(img):
    """V136: Enhanced background removal"""
    try:
        if not REPLICATE_AVAILABLE:
            print("Replicate not available, using local fallback")
            return extract_ring_local_fallback(img)
            
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print("Replicate API token not found, using local fallback")
            return extract_ring_local_fallback(img)
            
        print("V136: Starting Replicate background removal...")
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        output = replicate.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            input={
                "image": f"data:image/png;base64,{img_base64}",
                "model": "u2net",
                "alpha_matting": True,
                "alpha_matting_foreground_threshold": 270,
                "alpha_matting_background_threshold": 10,
                "alpha_matting_erode_size": 10
            }
        )
        
        response = requests.get(output)
        result_img = Image.open(BytesIO(response.content))
        
        if result_img.mode != 'RGBA':
            result_img = result_img.convert('RGBA')
        
        print("V136: Replicate background removal completed")
        return result_img
        
    except Exception as e:
        print(f"V136: Error with Replicate API: {e}")
        return extract_ring_local_fallback(img)

def extract_ring_local_fallback(img):
    """V136: Local fallback background removal"""
    print("V136: Using local fallback for background removal")
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    width, height = img.size
    img_array = np.array(img)
    
    corner_size = 15
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
    
    threshold = np.percentile(color_distance, 25)
    mask = color_distance > threshold
    
    mask = mask.astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, 'L')
    
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(5))
    mask_img = mask_img.filter(ImageFilter.SMOOTH_MORE)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
    
    result = img.copy()
    result.putalpha(mask_img)
    
    return result

def apply_metal_color_filter(img, color_multipliers):
    """V136: Apply metal color filter"""
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
    else:
        img = img.convert('RGBA')
        r, g, b, a = img.split()
    
    r = r.point(lambda x: min(255, max(0, int(x * color_multipliers[0] * 1.1))))
    g = g.point(lambda x: min(255, max(0, int(x * color_multipliers[1] * 1.1))))
    b = b.point(lambda x: min(255, max(0, int(x * color_multipliers[2] * 1.1))))
    
    return Image.merge('RGBA', (r, g, b, a))

def create_color_options_section(width=FIXED_WIDTH, ring_image=None):
    """V136: Create COLOR section with no inner filling"""
    print("V136: Creating COLOR section with real wedding ring")
    
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
                label_font = ImageFont.truetype(font_path, 30)
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
    
    # Colors
    colors = [
        ("yellow", (255, 215, 0)),
        ("rose", (232, 180, 184)),
        ("white", (245, 245, 245)),
        ("antique", (212, 175, 55))
    ]
    
    # Layout
    container_size = 300
    h_spacing = 100
    v_spacing = 400
    
    grid_width = 2 * container_size + h_spacing
    start_x = (width - grid_width) // 2
    start_y = 200
    
    # Process ring image
    processed_ring = None
    if ring_image:
        try:
            print("V136: Processing ring image...")
            processed_ring = extract_ring_with_replicate(ring_image)
            print(f"V136: Ring extraction completed, size: {processed_ring.size}")
        except Exception as e:
            print(f"V136: Ring processing failed: {e}")
            processed_ring = None
    
    # Create color variants
    for i, (name, color_rgb) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (container_size + h_spacing)
        y = start_y + row * v_spacing
        
        # Container
        container = Image.new('RGBA', (container_size, container_size), (255, 255, 255, 255))
        container_draw = ImageDraw.Draw(container)
        
        # Use real ring or fallback
        if processed_ring and processed_ring.size[0] > 0:
            try:
                print(f"V136: Applying {name} color to ring...")
                
                # Resize ring
                ring_width, ring_height = processed_ring.size
                max_size = container_size - 60
                
                aspect_ratio = ring_width / ring_height
                if aspect_ratio > 1:
                    new_width = max_size
                    new_height = int(max_size / aspect_ratio)
                else:
                    new_height = max_size
                    new_width = int(max_size * aspect_ratio)
                
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                ring_resized = processed_ring.resize((new_width, new_height), resample_filter)
                
                # Apply color
                color_multipliers = [
                    color_rgb[0] / 255.0,
                    color_rgb[1] / 255.0,
                    color_rgb[2] / 255.0
                ]
                colored_ring = apply_metal_color_filter(ring_resized, color_multipliers)
                
                # Center and paste
                ring_x = (container_size - new_width) // 2
                ring_y = (container_size - new_height) // 2
                
                if colored_ring.mode == 'RGBA':
                    container.paste(colored_ring, (ring_x, ring_y), colored_ring)
                else:
                    container.paste(colored_ring, (ring_x, ring_y))
                
                print(f"V136: {name} color applied successfully")
                
            except Exception as e:
                print(f"V136: Failed to apply {name} color: {e}")
                draw_fallback_rings(container_draw, container_size, color_rgb)
        else:
            draw_fallback_rings(container_draw, container_size, color_rgb)
        
        # Shadow
        shadow_img = Image.new('RGBA', (container_size + 15, container_size + 15), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        shadow_draw.rectangle([8, 8, container_size + 8, container_size + 8], fill=(0, 0, 0, 25))
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=6))
        
        section_img.paste(shadow_img, (x - 8, y - 8), shadow_img)
        section_img.paste(container, (x, y))
        
        # Label
        label_width, _ = get_text_dimensions(draw, name, label_font)
        draw.text((x + container_size//2 - label_width//2, y + container_size + 35), 
                 name, font=label_font, fill=(80, 80, 80))
    
    print("V136: Color section completed")
    return section_img

def draw_fallback_rings(container_draw, container_size, color_rgb):
    """V136: Draw fallback rings with NO inner filling - outline only"""
    print("V136: Drawing fallback rings - outline only")
    
    # Left ring - OUTLINE ONLY
    left_center_x = container_size // 2 - 35
    left_center_y = container_size // 2 - 10
    left_radius = 75
    left_thickness = 12
    
    container_draw.ellipse([
        left_center_x - left_radius,
        left_center_y - left_radius,
        left_center_x + left_radius,
        left_center_y + left_radius
    ], outline=color_rgb, width=left_thickness)
    
    # Inner ring outline
    inner_radius = left_radius - left_thickness - 5
    container_draw.ellipse([
        left_center_x - inner_radius,
        left_center_y - inner_radius,
        left_center_x + inner_radius,
        left_center_y + inner_radius
    ], outline=color_rgb, width=8)
    
    # Right ring - OUTLINE ONLY
    right_center_x = container_size // 2 + 45
    right_center_y = container_size // 2 + 25
    right_radius = 60
    right_thickness = 10
    
    container_draw.ellipse([
        right_center_x - right_radius,
        right_center_y - right_radius,
        right_center_x + right_radius,
        right_center_y + right_radius
    ], outline=color_rgb, width=right_thickness)
    
    inner_radius = right_radius - right_thickness - 5
    container_draw.ellipse([
        right_center_x - inner_radius,
        right_center_y - inner_radius,
        right_center_x + inner_radius,
        right_center_y + inner_radius
    ], outline=color_rgb, width=6)
    
    # Diamonds
    diamond_size = 10
    diamond_y = left_center_y - left_radius + left_thickness//2
    container_draw.polygon([
        (left_center_x, diamond_y - diamond_size),
        (left_center_x + diamond_size//2, diamond_y - diamond_size//2),
        (left_center_x, diamond_y),
        (left_center_x - diamond_size//2, diamond_y - diamond_size//2)
    ], fill=(255, 255, 255), outline=(200, 200, 200), width=2)
    
    small_diamond_size = 8
    small_diamond_y = right_center_y - right_radius + right_thickness//2
    container_draw.polygon([
        (right_center_x, small_diamond_y - small_diamond_size),
        (right_center_x + small_diamond_size//2, small_diamond_y - small_diamond_size//2),
        (right_center_x, small_diamond_y),
        (right_center_x - small_diamond_size//2, small_diamond_y - small_diamond_size//2)
    ], fill=(255, 255, 255), outline=(200, 200, 200), width=2)

def create_ai_generated_md_talk(claude_text, width=FIXED_WIDTH):
    """V136: Create MD Talk section with perfect Korean support"""
    print("V136: Creating MD TALK section with Korean support")
    
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
                body_font = ImageFont.truetype(font_path, 32)  # Larger font
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
        cleaned_text = clean_korean_text(claude_text)
        cleaned_text = re.sub(r'^(MD TALK|md talk|MD talk|엠디톡)\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        # Text wrapping
        words = cleaned_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 140:
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
    line_height = 60
    
    for line in lines:
        safe_line = korean_safe_string_encode(line)
        line_width, _ = get_text_dimensions(draw, safe_line, body_font)
        
        # Text shadow
        shadow_x = width//2 - line_width//2 + 2
        shadow_y = y_pos + 2
        draw.text((shadow_x, shadow_y), safe_line, font=body_font, fill=(220, 220, 220))
        
        # Main text
        draw.text((width//2 - line_width//2, y_pos), safe_line, font=body_font, fill=(50, 50, 50))
        y_pos += line_height
    
    print("V136: MD TALK section completed")
    return section_img

def create_ai_generated_design_point(claude_text, width=FIXED_WIDTH):
    """V136: Create Design Point section with perfect Korean support"""
    print("V136: Creating DESIGN POINT section with Korean support")
    
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
                body_font = ImageFont.truetype(font_path, 28)
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
        cleaned_text = clean_korean_text(claude_text)
        cleaned_text = re.sub(r'^(DESIGN POINT|design point|Design Point|디자인포인트|디자인 포인트)\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        # Text wrapping
        words = cleaned_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 140:
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
    line_height = 65
    
    for line in lines:
        safe_line = korean_safe_string_encode(line)
        line_width, _ = get_text_dimensions(draw, safe_line, body_font)
        
        # Text shadow
        shadow_x = width//2 - line_width//2 + 2
        shadow_y = y_pos + 2
        draw.text((shadow_x, shadow_y), safe_line, font=body_font, fill=(220, 220, 220))
        
        # Main text
        draw.text((width//2 - line_width//2, y_pos), safe_line, font=body_font, fill=(50, 50, 50))
        y_pos += line_height
    
    print("V136: DESIGN POINT section completed")
    return section_img

def extract_file_id_from_url(url):
    """Extract Google Drive file ID"""
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
    """Download image from Google Drive"""
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
    """Calculate proportional height"""
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
    
    return detail_page

def process_clean_combined_images(images_data, group_number, input_data=None):
    """Process combined images (groups 3, 4, 5)"""
    print(f"Processing {len(images_data)} images for group {group_number}")
    
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
    
    print(f"Creating combined canvas: {FIXED_WIDTH}x{total_height}")
    
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
    
    return detail_page

def process_color_section(input_data):
    """V136: Process group 6 - COLOR section"""
    print("V136: Processing GROUP 6 COLOR section")
    
    img = None
    
    if 'image9' in input_data:
        print("V136: Found image9 key")
        img_data = {'url': input_data['image9']}
        img = get_image_from_input(img_data)
    elif 'group6' in input_data:
        print("V136: Found group6 key")
        img_data = {'url': input_data['group6']}
        img = get_image_from_input(img_data)
    elif 'image6' in input_data:
        print("V136: Found image6 key")
        img_data = {'url': input_data['image6']}
        img = get_image_from_input(img_data)
    else:
        print("V136: Using standard image input")
        try:
            img = get_image_from_input(input_data)
        except:
            print("V136: No image found for COLOR section")
            img = None
    
    if img:
        print(f"V136: Ring image found: {img.size}, mode: {img.mode}")
    else:
        print("V136: No ring image, creating without")
    
    color_section = create_color_options_section(ring_image=img)
    
    if img:
        img.close()
    
    print("V136: Color section created")
    return color_section

def process_korean_text_section(input_data, group_number):
    """V136 KOREAN SAFE: 100% 한국어 안전 텍스트 처리 - NO LATIN-1 EVER!"""
    print(f"V136: 한국어 안전 텍스트 처리 시작 - 그룹 {group_number}")
    
    # V136: 한국어 안전 base64 처리
    claude_text_base64 = input_data.get('claude_text_base64', '')
    claude_text = ""
    
    if claude_text_base64:
        try:
            print("V136: base64 한국어 텍스트 발견")
            # 패딩 추가
            missing_padding = len(claude_text_base64) % 4
            if missing_padding:
                claude_text_base64 += '=' * (4 - missing_padding)
            
            # V136 한국어 안전: ONLY UTF-8, NO LATIN-1 EVER!
            try:
                # 1차: UTF-8 직접 디코딩
                decoded_bytes = base64.b64decode(claude_text_base64)
                claude_text = decoded_bytes.decode('utf-8')
                print("V136 성공: UTF-8 직접 디코딩 성공")
            except UnicodeDecodeError as e1:
                print(f"V136: UTF-8 직접 실패 ({e1}), 오류 처리 시도")
                try:
                    # 2차: UTF-8 with replace (한국어 보존)
                    claude_text = decoded_bytes.decode('utf-8', errors='replace')
                    print("V136 성공: UTF-8 replace 디코딩 성공")
                except Exception as e2:
                    print(f"V136: UTF-8 replace 실패 ({e2}), 안전 fallback 사용")
                    # 3차: 한국어 안전 fallback (NO LATIN-1!)
                    claude_text = "한국어 텍스트 디코딩 오류 발생"
                    print("V136: 한국어 안전 fallback 사용")
        except Exception as e:
            print(f"V136 오류: Base64 디코딩 실패: {e}")
            claude_text = ""
    else:
        # 일반 텍스트 필드 fallback
        claude_text = (input_data.get('claude_text') or 
                      input_data.get('text_content') or 
                      input_data.get('ai_text') or 
                      input_data.get('generated_text') or '')
    
    # V136: 한국어 안전 텍스트 정리
    if claude_text:
        claude_text = korean_safe_string_encode(claude_text)
        claude_text = clean_korean_text(claude_text)
    
    text_type = (input_data.get('text_type') or 
                input_data.get('section_type') or '')
    
    print(f"V136: 텍스트 타입: {text_type}")
    print(f"V136: 그룹 번호: {group_number}")
    print(f"V136: 한국어 텍스트 미리보기: {claude_text[:100] if claude_text else '텍스트 없음'}...")
    
    # 그룹별 텍스트 섹션 생성
    if group_number == 7:
        print("V136: 그룹 7 확인 - MD TALK 섹션 생성")
        text_section = create_ai_generated_md_talk(claude_text)
        section_type = "md_talk"
    elif group_number == 8:
        print("V136: 그룹 8 확인 - DESIGN POINT 섹션 생성")
        text_section = create_ai_generated_design_point(claude_text)
        section_type = "design_point"
    else:
        print(f"V136 경고: 예상치 못한 그룹 번호 {group_number}")
        if 'md' in text_type.lower():
            text_section = create_ai_generated_md_talk(claude_text)
            section_type = "md_talk"
        else:
            text_section = create_ai_generated_design_point(claude_text)
            section_type = "design_point"
    
    print(f"V136: 한국어 텍스트 섹션 생성 완료: {section_type}")
    return text_section, section_type

def send_to_webhook(image_base64, handler_type, file_name, route_number=0, metadata={}):
    """V136: Korean-safe webhook sending"""
    try:
        if not WEBHOOK_URL:
            print("WARNING: Webhook URL not configured")
            return None
        
        # V136: Korean-safe metadata
        safe_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                safe_metadata[key] = korean_safe_string_encode(value)
            elif isinstance(value, dict):
                safe_dict = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        safe_dict[sub_key] = korean_safe_string_encode(sub_value)
                    else:
                        safe_dict[sub_key] = sub_value
                safe_metadata[key] = safe_dict
            else:
                safe_metadata[key] = value
                
        webhook_data = {
            "handler_type": korean_safe_string_encode(handler_type),
            "file_name": korean_safe_string_encode(file_name),
            "route_number": route_number,
            "runpod_result": {
                "output": {
                    "output": safe_metadata
                }
            }
        }
        
        webhook_data["runpod_result"]["output"]["output"]["enhanced_image"] = image_base64
        
        print(f"V136: 웹훅 전송: {handler_type} for {file_name}")
        
        # V136: Korean-safe JSON serialization
        try:
            test_json = json.dumps(webhook_data, ensure_ascii=False)
            print("V136: JSON 직렬화 성공 (한국어 지원)")
        except Exception as json_err:
            print(f"V136: JSON 직렬화 실패: {json_err}")
            try:
                test_json = json.dumps(webhook_data, ensure_ascii=True)
                print("V136: ASCII fallback 성공")
            except Exception as ascii_err:
                print(f"V136: ASCII 직렬화도 실패: {ascii_err}")
                return None
        
        response = requests.post(
            WEBHOOK_URL,
            json=webhook_data,
            headers={'Content-Type': 'application/json; charset=utf-8'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"V136: 웹훅 성공: {result}")
            return result
        else:
            print(f"V136: 웹훅 실패: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"V136: 웹훅 오류: {str(e)}")
        return None

def detect_group_number_from_input(input_data):
    """V136: Group number detection"""
    print("=== GROUP NUMBER DETECTION V136 ===")
    print(f"Input keys: {sorted(input_data.keys())}")
    
    # Priority 1: route_number
    route_number = input_data.get('route_number', 0)
    try:
        route_number = int(str(route_number)) if str(route_number).isdigit() else 0
    except:
        route_number = 0
        
    if route_number > 0:
        print(f"Found route_number: {route_number}")
        return route_number
    
    # Priority 2: image9 (Group 6)
    if 'image9' in input_data:
        print("Found image9 - GROUP 6")
        return 6
    
    # Priority 3: group keys
    if 'group6' in input_data or 'group_6' in input_data:
        print("Found group6 key - GROUP 6")
        return 6
    
    # Priority 4: group_number field
    group_number = input_data.get('group_number', 0)
    try:
        group_number = int(str(group_number)) if str(group_number).isdigit() else 0
    except:
        group_number = 0
        
    if group_number > 0:
        print(f"Found group_number: {group_number}")
        return group_number
    
    # Priority 5: text_type
    text_type = str(input_data.get('text_type', ''))
    if text_type == 'md_talk':
        print("Found md_talk - GROUP 7")
        return 7
    elif text_type == 'design_point':
        print("Found design_point - GROUP 8")
        return 8
    
    # Priority 6: COLOR indicators
    all_text = str(input_data).lower()
    if any(word in all_text for word in ['color', 'colour', '컬러', '색상']):
        print("Found COLOR keywords - GROUP 6")
        return 6
    
    # Priority 7: image field analysis
    if 'image' in input_data and not any(f'image{i}' in input_data for i in range(1, 10)):
        image_value = str(input_data.get('image', ''))
        if ';' in image_value:
            urls = [u.strip() for u in image_value.split(';') if u.strip()]
            url_count = len(urls)
            print(f"Found {url_count} URLs in image field")
            
            if url_count == 2:
                return 3
            elif url_count >= 3:
                return 4
        else:
            return 1
    
    # Priority 8: specific image keys
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
    """V136 KOREAN SAFE: Main handler with 100% Korean safety - NO LATIN-1 EVER!"""
    try:
        print("=== V136 한국어 안전 핸들러 시작 ===")
        
        # 한국어 폰트 다운로드
        if not os.path.exists('/tmp/NanumMyeongjo.ttf'):
            print("한국어 폰트 다운로드 중...")
            if not download_korean_font():
                print("경고: 한국어 폰트 다운로드 실패")
        else:
            print("한국어 폰트 존재함")
        
        # 입력 데이터 가져오기
        input_data = event.get('input', event)
        
        print("=== 입력 데이터 ===")
        print(f"Keys: {sorted(input_data.keys())}")
        print(f"route_number: {input_data.get('route_number', '없음')}")
        
        # 그룹 감지
        group_number = detect_group_number_from_input(input_data)
        
        print(f"\n=== 감지 결과 ===")
        print(f"감지된 그룹: {group_number}")
        
        # route_number 오버라이드
        route_str = str(input_data.get('route_number', '0'))
        try:
            route_int = int(route_str) if route_str.isdigit() else 0
        except:
            route_int = 0
            
        if route_int > 0:
            original_group = group_number
            group_number = route_int
            if original_group != group_number:
                print(f"!!! 오버라이드: 그룹 {original_group} → {group_number}")
        
        print(f"\n=== 최종 그룹: {group_number} ===")
        
        # 그룹 유효성 검사
        if group_number == 0:
            raise ValueError(f"그룹 번호를 결정할 수 없습니다. Keys: {list(input_data.keys())}")
        
        if group_number < 1 or group_number > 8:
            raise ValueError(f"잘못된 그룹 번호: {group_number}. 1-8이어야 합니다.")
        
        # 이미지 입력 형식 처리
        if 'image' in input_data and input_data['image']:
            print(f"'image' 키 발견: {input_data['image'][:100]}...")
            image_data = input_data['image']
            
            if ';' in str(image_data):
                urls = str(image_data).split(';')
                input_data['images'] = []
                for url in urls:
                    url = url.strip()
                    if url:
                        input_data['images'].append({'url': url})
                print(f"'image'를 {len(input_data['images'])}개 이미지 배열로 변환")
            else:
                input_data['url'] = image_data
                print("'image' 키에서 단일 URL 설정")
        
        if 'combined_urls' in input_data and input_data['combined_urls']:
            urls = str(input_data['combined_urls']).split(';')
            input_data['images'] = []
            for url in urls:
                url = url.strip()
                if url:
                    input_data['images'].append({'url': url})
            print(f"combined_urls를 {len(input_data['images'])}개 이미지로 변환")
        
        # 특정 이미지 키 처리
        if not input_data.get('images') and not input_data.get('url'):
            if group_number == 6:
                if 'image9' in input_data:
                    input_data['url'] = input_data['image9']
                    print("V136 그룹 6: image9 사용")
                elif 'image6' in input_data:
                    input_data['url'] = input_data['image6']
                    print("V136 그룹 6: image6 사용")
                elif 'group6' in input_data:
                    input_data['url'] = input_data['group6']
                    print("V136 그룹 6: group6 사용")
                else:
                    print("V136 경고: 그룹 6이지만 특정 이미지를 찾을 수 없음")
            
            elif f'image{group_number}' in input_data:
                image_url = input_data[f'image{group_number}']
                if ';' in str(image_url):
                    urls = str(image_url).split(';')
                    input_data['images'] = [{'url': url.strip()} for url in urls if url.strip()]
                else:
                    input_data['url'] = image_url
                print(f"image{group_number} 처리됨")
            
            elif group_number == 3:
                images_to_add = []
                if 'image3' in input_data:
                    images_to_add.append({'url': input_data['image3']})
                if 'image4' in input_data:
                    images_to_add.append({'url': input_data['image4']})
                if images_to_add:
                    input_data['images'] = images_to_add
                    print(f"그룹 3: {len(images_to_add)}개 이미지 발견")
            
            elif group_number == 4:
                images_to_add = []
                if 'image5' in input_data:
                    images_to_add.append({'url': input_data['image5']})
                if 'image6' in input_data and 'image5' in input_data:
                    images_to_add.append({'url': input_data['image6']})
                if images_to_add:
                    input_data['images'] = images_to_add
                    print(f"그룹 4: {len(images_to_add)}개 이미지 발견")
            
            elif group_number == 5:
                images_to_add = []
                if 'image7' in input_data:
                    images_to_add.append({'url': input_data['image7']})
                if 'image8' in input_data:
                    images_to_add.append({'url': input_data['image8']})
                if images_to_add:
                    input_data['images'] = images_to_add
                    print(f"그룹 5: {len(images_to_add)}개 이미지 발견")
        
        # 그룹별 처리
        if group_number == 6:
            print("V136: 그룹 6 COLOR 섹션 처리")
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif group_number == 7:
            print("V136: 그룹 7 MD TALK 텍스트 섹션 처리")
            detail_page, section_type = process_korean_text_section(input_data, 7)
            page_type = f"text_section_{section_type}"
            
        elif group_number == 8:
            print("V136: 그룹 8 DESIGN POINT 텍스트 섹션 처리")
            detail_page, section_type = process_korean_text_section(input_data, 8)
            page_type = f"text_section_{section_type}"
            
        elif group_number in [1, 2]:
            print(f"V136: 그룹 {group_number} 개별 이미지 처리")
            detail_page = process_single_image(input_data, group_number)
            page_type = "individual"
            
        elif group_number in [3, 4, 5]:
            print(f"V136: 그룹 {group_number} 결합 이미지 처리")
            if 'images' not in input_data or not isinstance(input_data['images'], list):
                input_data['images'] = [input_data]
            
            if group_number == 5:
                detail_page = process_clean_combined_images(input_data.get('images', []), group_number, input_data)
            else:
                detail_page = process_clean_combined_images(input_data['images'], group_number, input_data)
            
            page_type = "clean_combined"
        
        else:
            raise ValueError(f"잘못된 그룹 번호: {group_number}. 1-8이어야 합니다.")
        
        # Base64 변환
        buffered = BytesIO()
        detail_page.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue())
        
        detail_base64 = img_str.decode('utf-8')
        detail_base64_no_padding = detail_base64.rstrip('=')
        
        print(f"V136: 상세 페이지 생성됨: {detail_page.size}")
        print(f"V136: Base64 길이: {len(detail_base64_no_padding)} 문자")
        
        # V136: 한국어 안전 메타데이터
        metadata = {
            "enhanced_image": detail_base64_no_padding,
            "status": "success",
            "page_type": korean_safe_string_encode(page_type),
            "page_number": group_number,
            "route_number": group_number,
            "actual_group": group_number,
            "dimensions": {
                "width": detail_page.width,
                "height": detail_page.height
            },
            "version": "V136_KOREAN_SAFE",
            "image_count": len(input_data.get('images', [input_data])),
            "processing_time": "calculated_later",
            "font_status": "korean_font_available" if os.path.exists('/tmp/NanumMyeongjo.ttf') else "fallback_font",
            "korean_support": "100_percent_safe",
            "latin1_usage": "completely_eliminated"
        }
        
        # 웹훅 전송
        file_name = f"detail_group_{group_number}.png"
        webhook_result = send_to_webhook(detail_base64_no_padding, "detail", file_name, group_number, metadata)
        
        # 응답 반환
        return {
            "output": metadata
        }
        
    except Exception as e:
        # V136: 한국어 안전 오류 처리
        error_msg = korean_safe_string_encode(f"상세 페이지 생성 실패: {str(e)}")
        print(f"V136 오류: {error_msg}")
        traceback_str = korean_safe_string_encode(traceback.format_exc())
        print(f"V136 추적: {traceback_str}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "error",
                "traceback": traceback_str,
                "version": "V136_KOREAN_SAFE"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("V136 한국어 안전 핸들러 시작 - LATIN-1 완전 제거!")
    runpod.serverless.start({"handler": handler})
