import runpod
import os
import requests
import base64
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
from io import BytesIO
import json
import re
import numpy as np
from scipy import ndimage
import cv2
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import time

# Claude API configuration
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# REPLICATE API for background removal
try:
    import replicate
    REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
    if REPLICATE_API_TOKEN:
        REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_API_TOKEN)
        REPLICATE_AVAILABLE = True
    else:
        print("WARNING: REPLICATE_API_TOKEN not set. Background removal will use local method.")
        REPLICATE_AVAILABLE = False
except ImportError:
    print("WARNING: replicate package not installed. Background removal will use local method.")
    REPLICATE_AVAILABLE = False

# Try to import rembg for local background removal
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
    print("rembg available for local background removal")
except ImportError:
    REMBG_AVAILABLE = False
    print("rembg not available, will use Replicate if available")

# Webhook URL - Google Apps Script Web App URL
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzOQ7SaTtIXRubvSNXNY53pphacVmJg_XKV5sIyOgxjpDykiWsAHN7ecKFHcygGFrYi/exec"

# FIXED WIDTH FOR ALL IMAGES
FIXED_WIDTH = 1200

# Global cache for performance
SAMPLE_CACHE = {}
MASK_CACHE = {}

def download_korean_font():
    """Download Korean font for text rendering with better error handling"""
    try:
        font_path = '/tmp/NanumGothic.ttf'
        
        if os.path.exists(font_path):
            try:
                test_font = ImageFont.truetype(font_path, 20)
                img_test = Image.new('RGB', (100, 100), 'white')
                draw_test = ImageDraw.Draw(img_test)
                draw_test.text((10, 10), "테스트", font=test_font, fill='black')
                print("Korean font already exists and works properly")
                return font_path
            except Exception as e:
                print(f"Korean font exists but has issues: {e}")
                os.remove(font_path)
        
        font_urls = [
            'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
            'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
            'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf'
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
                    img_test = Image.new('RGB', (100, 100), 'white')
                    draw_test = ImageDraw.Draw(img_test)
                    draw_test.text((10, 10), "한글테스트", font=test_font, fill='black')
                    print(f"Korean font downloaded and verified from {url}")
                    return font_path
            except Exception as e:
                print(f"Failed to download from {url}: {str(e)}")
                continue
        
        print("Failed to download Korean font, using fallback")
        return None
        
    except Exception as e:
        print(f"Error in font download process: {str(e)}")
        return None

def get_font(size, korean_font_path=None):
    """Get font with proper fallback handling"""
    fonts_to_try = []
    
    if korean_font_path and os.path.exists(korean_font_path):
        fonts_to_try.append(korean_font_path)
    
    fonts_to_try.extend([
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf"
    ])
    
    for font_path in fonts_to_try:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, size)
                img_test = Image.new('RGB', (100, 100), 'white')
                draw_test = ImageDraw.Draw(img_test)
                draw_test.text((10, 10), "한글", font=font, fill='black')
                return font
            except Exception as e:
                print(f"Failed to load font {font_path}: {e}")
                continue
    
    print("Using default font as last resort")
    return ImageFont.load_default()

def safe_draw_text(draw, position, text, font, fill):
    """Safely draw text with proper encoding"""
    try:
        if text:
            text = str(text)
            draw.text(position, text, font=font, fill=fill)
    except Exception as e:
        print(f"Error drawing text '{text}': {str(e)}")
        try:
            safe_text = ''.join(c if ord(c) < 128 or 0xAC00 <= ord(c) <= 0xD7A3 else '?' for c in text)
            draw.text(position, safe_text or "[Error]", font=font, fill=fill)
        except:
            draw.text(position, "[Error]", font=font, fill=fill)

def get_text_size(draw, text, font):
    """Get text size compatible with different PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def call_claude_api(image_base64, prompt):
    """Call Claude API to generate text based on image"""
    if not CLAUDE_API_KEY:
        print("WARNING: CLAUDE_API_KEY not set")
        return None
    
    try:
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 500,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }]
        }
        
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('content', [{}])[0].get('text', '')
        else:
            print(f"Claude API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Claude API: {str(e)}")
        return None

def generate_product_name_and_description(image):
    """Generate product name and beautiful description using Claude"""
    try:
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prompt for Claude
        prompt = """이 주얼리 이미지를 보고 다음 두 가지를 생성해주세요:

1. 제품명: 이 주얼리에 어울리는 아름답고 우아한 이름 (한글 또는 영어, 2-3단어)
2. 아름다운 해석: 제품명에 담긴 의미를 시적이고 감성적으로 설명 (한국어, 1-2문장)

다음 형식으로 응답해주세요:
제품명: [여기에 제품명]
해석: [여기에 아름다운 해석]

예시:
제품명: Eternal Bloom
해석: 영원히 피어나는 꽃처럼, 변하지 않는 사랑과 아름다움을 담은 주얼리입니다."""
        
        # Call Claude API
        response = call_claude_api(img_base64, prompt)
        
        if response:
            # Parse response
            lines = response.strip().split('\n')
            product_name = ""
            description = ""
            
            for line in lines:
                if line.startswith("제품명:"):
                    product_name = line.replace("제품명:", "").strip()
                elif line.startswith("해석:"):
                    description = line.replace("해석:", "").strip()
            
            return product_name, description
        else:
            # Fallback values
            return "Signature Ring", "당신만의 특별한 순간을 영원히 간직하는 시그니처 링"
            
    except Exception as e:
        print(f"Error generating product name: {str(e)}")
        return "Signature Ring", "당신만의 특별한 순간을 영원히 간직하는 시그니처 링"

def clean_claude_text(text):
    """Clean text for safe rendering"""
    if not text:
        return ""
    
    text = str(text) if text is not None else ""
    text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'[#*_`]', '', text)
    text = ' '.join(text.split())
    
    return text.strip()

def extract_file_id_from_url(url):
    """Extract Google Drive file ID from URL"""
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
        print(f"Processing Google Drive URL: {url[:80]}...")
        
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
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }
        
        session = requests.Session()
        
        for download_url in download_urls:
            try:
                response = session.get(download_url, headers=headers, stream=True, timeout=30)
                
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    print(f"Successfully downloaded image: {img.size}")
                    return img
                    
            except Exception as e:
                print(f"Failed with URL: {download_url}, Error: {str(e)}")
                continue
        
        raise Exception(f"Failed to download from all URLs for file ID: {file_id}")
        
    except Exception as e:
        print(f"Error downloading from Google Drive: {str(e)}")
        raise

def get_image_from_input(input_data):
    """Get image from various input formats"""
    try:
        for key in ['image', 'url', 'enhanced_image', 'image1', 'image2', 'image3', 
                   'image4', 'image5', 'image6', 'image7', 'image8', 'image9']:
            if key in input_data and input_data[key]:
                image_data = input_data[key]
                
                if isinstance(image_data, str):
                    if image_data.startswith('http'):
                        return download_image_from_google_drive(image_data)
                    elif image_data.startswith('data:'):
                        header, data = image_data.split(',', 1)
                        return Image.open(BytesIO(base64.b64decode(data)))
                    else:
                        return Image.open(BytesIO(base64.b64decode(image_data)))
        
        raise ValueError("No valid image data found in input")
        
    except Exception as e:
        print(f"Error getting image: {str(e)}")
        raise

def parse_figma_style_info(input_data):
    """Parse Figma style information from input data"""
    style_info = {
        'text_position': {'x': None, 'y': None},
        'font_size': 48,
        'text_color': (40, 40, 40),
        'background_color': None,
        'text_align': 'center',
        'has_background': False,
        'background_padding': 20,
        'background_opacity': 0.9
    }
    
    # Check for Figma style data
    figma_style = input_data.get('figma_style', {})
    if isinstance(figma_style, str):
        try:
            figma_style = json.loads(figma_style)
        except:
            figma_style = {}
    
    # Parse position
    if 'position' in figma_style:
        style_info['text_position']['x'] = figma_style['position'].get('x')
        style_info['text_position']['y'] = figma_style['position'].get('y')
    
    # Parse text style
    if 'fontSize' in figma_style:
        style_info['font_size'] = int(figma_style['fontSize'])
    
    if 'color' in figma_style:
        # Convert hex to RGB
        hex_color = figma_style['color'].lstrip('#')
        if len(hex_color) == 6:
            style_info['text_color'] = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Parse background
    if 'background' in figma_style:
        style_info['has_background'] = True
        bg_color = figma_style['background'].lstrip('#')
        if len(bg_color) == 6:
            style_info['background_color'] = tuple(int(bg_color[i:i+2], 16) for i in (0, 2, 4))
    
    return style_info

def add_figma_style_background(draw, text_bbox, style_info):
    """Add Figma-style background to text"""
    if not style_info['has_background']:
        return
    
    x1, y1, x2, y2 = text_bbox
    padding = style_info['background_padding']
    
    # Expand bbox with padding
    bg_bbox = [
        x1 - padding,
        y1 - padding,
        x2 + padding,
        y2 + padding
    ]
    
    # Draw background
    bg_color = style_info['background_color'] or (255, 255, 255)
    if style_info['background_opacity'] < 1.0:
        # Create semi-transparent background
        overlay = Image.new('RGBA', draw.im.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        opacity = int(255 * style_info['background_opacity'])
        overlay_draw.rectangle(bg_bbox, fill=(*bg_color, opacity))
        draw.im.paste(overlay, (0, 0), overlay)
    else:
        draw.rectangle(bg_bbox, fill=bg_color)

def add_text_overlay_with_figma_style(image, text, figma_style_info, is_group1=True):
    """Add text overlay using Figma style information"""
    img_copy = image.copy()
    
    # Convert to RGBA for transparency support
    if img_copy.mode != 'RGBA':
        img_copy = img_copy.convert('RGBA')
    
    draw = ImageDraw.Draw(img_copy)
    
    korean_font_path = download_korean_font()
    font = get_font(figma_style_info['font_size'], korean_font_path)
    
    # Calculate text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Determine position
    if figma_style_info['text_position']['x'] is not None:
        x_position = figma_style_info['text_position']['x']
    else:
        # Default center alignment
        x_position = (img_copy.width - text_width) // 2
    
    if figma_style_info['text_position']['y'] is not None:
        y_position = figma_style_info['text_position']['y']
    else:
        # Default position based on group
        y_position = 100 if is_group1 else 300
    
    # Create text bbox
    text_bbox = [
        x_position,
        y_position,
        x_position + text_width,
        y_position + text_height
    ]
    
    # Add background if specified
    add_figma_style_background(draw, text_bbox, figma_style_info)
    
    # Draw text with outline for better visibility
    outline_width = 2
    outline_color = (255, 255, 255)  # White outline
    
    for adj_x in range(-outline_width, outline_width + 1):
        for adj_y in range(-outline_width, outline_width + 1):
            if adj_x != 0 or adj_y != 0:
                safe_draw_text(draw, (x_position + adj_x, y_position + adj_y), text, font, outline_color)
    
    # Draw main text
    safe_draw_text(draw, (x_position, y_position), text, font, figma_style_info['text_color'])
    
    # Convert back to RGB if needed
    if image.mode == 'RGB':
        img_copy = img_copy.convert('RGB')
    
    return img_copy

def add_text_overlay_group1(image, figma_info=None):
    """Add 'twinkring' text overlay for Group 1 with Figma style"""
    # Default text
    text = "twinkring"
    
    # Check if custom text from Figma
    if figma_info and figma_info.get('figma_text_content'):
        custom_text = figma_info['figma_text_content']
        if 'text_001' in figma_info and figma_info['text_001'] == 'TRUE':
            text = custom_text.split(',')[0].strip() if ',' in custom_text else custom_text
    
    # Parse Figma style
    style_info = parse_figma_style_info(figma_info or {})
    
    return add_text_overlay_with_figma_style(image, text, style_info, is_group1=True)

def add_text_overlay_group2(image, product_name, description, figma_info=None):
    """Add product name and description text overlay for Group 2 with Figma style"""
    img_copy = image.copy()
    
    # Convert to RGBA for transparency
    if img_copy.mode != 'RGBA':
        img_copy = img_copy.convert('RGBA')
    
    draw = ImageDraw.Draw(img_copy)
    
    # Parse Figma style
    style_info = parse_figma_style_info(figma_info or {})
    
    korean_font_path = download_korean_font()
    
    # Font settings from Figma or defaults
    title_font_size = style_info['font_size']
    desc_font_size = int(title_font_size * 0.7)  # Description font is 70% of title
    
    title_font = get_font(title_font_size, korean_font_path)
    desc_font = get_font(desc_font_size, korean_font_path)
    
    # Starting position
    if style_info['text_position']['y'] is not None:
        y_position = style_info['text_position']['y']
    else:
        y_position = 300  # Default for group 2
    
    # Draw product name with background
    if product_name:
        # Calculate text dimensions
        bbox = draw.textbbox((0, 0), product_name, font=title_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # X position
        if style_info['text_position']['x'] is not None:
            x_position = style_info['text_position']['x']
        else:
            x_position = (img_copy.width - text_width) // 2
        
        # Text bbox for background
        text_bbox = [
            x_position,
            y_position,
            x_position + text_width,
            y_position + text_height
        ]
        
        # Add background
        add_figma_style_background(draw, text_bbox, style_info)
        
        # Draw text with outline
        outline_width = 2
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                if adj_x != 0 or adj_y != 0:
                    safe_draw_text(draw, (x_position + adj_x, y_position + adj_y), product_name, title_font, (255, 255, 255))
        
        safe_draw_text(draw, (x_position, y_position), product_name, title_font, style_info['text_color'])
        y_position += text_height + 40
    
    # Draw description with word wrap
    if description:
        # Word wrap
        words = description.split()
        lines = []
        current_line = ""
        max_line_width = img_copy.width - 200
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width = draw.textbbox((0, 0), test_line, font=desc_font)[2]
            
            if test_width > max_line_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        # Create background for all lines
        if style_info['has_background'] and lines:
            line_height = desc_font_size + 20
            total_height = len(lines) * line_height
            
            # Find max width
            max_width = 0
            for line in lines:
                line_width = draw.textbbox((0, 0), line, font=desc_font)[2]
                max_width = max(max_width, line_width)
            
            # Background box
            bg_x = (img_copy.width - max_width) // 2 - style_info['background_padding']
            bg_bbox = [
                bg_x,
                y_position - style_info['background_padding'],
                bg_x + max_width + 2 * style_info['background_padding'],
                y_position + total_height
            ]
            
            add_figma_style_background(draw, bg_bbox, style_info)
        
        # Draw each line
        for line in lines:
            if line:
                bbox = draw.textbbox((0, 0), line, font=desc_font)
                text_width = bbox[2] - bbox[0]
                x_position = (img_copy.width - text_width) // 2
                
                # Outline
                for adj_x in range(-1, 2):
                    for adj_y in range(-1, 2):
                        if adj_x != 0 or adj_y != 0:
                            safe_draw_text(draw, (x_position + adj_x, y_position + adj_y), line, desc_font, (255, 255, 255))
                
                safe_draw_text(draw, (x_position, y_position), line, desc_font, style_info['text_color'])
                y_position += desc_font_size + 20
    
    # Convert back to RGB if needed
    if image.mode == 'RGB':
        img_copy = img_copy.convert('RGB')
    
    return img_copy

def process_single_image(input_data, group_number):
    """Process single image with text overlay for groups 1 and 2"""
    print(f"Processing single image for group {group_number}")
    
    # Get the image based on group number
    if group_number == 1:
        img = get_image_from_input({'image1': input_data.get('image1', input_data.get('image'))})
    elif group_number == 2:
        img = get_image_from_input({'image2': input_data.get('image2', input_data.get('image'))})
    else:
        raise ValueError(f"Invalid group number for single image: {group_number}")
    
    # Get Figma information from input
    figma_info = {
        'text_001': input_data.get('text_001', 'FALSE'),
        'text_002': input_data.get('text_002', 'FALSE'),
        'text_003': input_data.get('text_003', 'FALSE'),
        'figma_text_content': input_data.get('figma_text_content', ''),
        'figma_node_id': input_data.get('figma_node_id') or input_data.get('NodeID', ''),
        'figma_style': input_data.get('figma_style', {})
    }
    
    # Calculate dimensions
    target_width = FIXED_WIDTH
    image_height = int(target_width * 1.3)
    
    # Layout parameters
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    
    total_height = TOP_MARGIN + image_height + BOTTOM_MARGIN
    
    # Create page
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    
    # Resize image
    try:
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        img_resized = img.resize((target_width, image_height), resample_filter)
        
        # Add text overlay based on group
        if group_number == 1:
            # Add 'twinkring' text with Figma style
            img_with_text = add_text_overlay_group1(img_resized, figma_info)
        elif group_number == 2:
            # Check if we should use Figma text or generate with Claude
            if figma_info.get('text_002') == 'TRUE' and figma_info.get('figma_text_content'):
                # Use Figma text content
                figma_texts = figma_info['figma_text_content'].split(',')
                if len(figma_texts) >= 2:
                    product_name = figma_texts[0].strip()
                    description = figma_texts[1].strip() if len(figma_texts) > 1 else ""
                else:
                    product_name = figma_texts[0].strip()
                    description = ""
            else:
                # Generate product name and description using Claude
                product_name, description = generate_product_name_and_description(img)
            
            img_with_text = add_text_overlay_group2(img_resized, product_name, description, figma_info)
        else:
            img_with_text = img_resized
        
        # Paste image
        detail_page.paste(img_with_text, (0, TOP_MARGIN))
        
        img.close()
        print(f"Placed single image with text overlay at y={TOP_MARGIN}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
    page_texts = {
        1: "- twinkring -",
        2: "- product name -"
    }
    page_text = page_texts.get(group_number, f"- Image {group_number} -")
    
    korean_font_path = download_korean_font()
    small_font = get_font(16, korean_font_path)
    
    text_width, _ = get_text_size(draw, page_text, small_font)
    safe_draw_text(draw, (FIXED_WIDTH//2 - text_width//2, total_height - 30), 
                 page_text, small_font, (200, 200, 200))
    
    return detail_page

def parse_semicolon_separated_urls(url_string):
    """Parse semicolon-separated URLs from Google Script"""
    if not url_string:
        return []
    
    url_string = url_string.strip()
    urls = []
    for url in url_string.split(';'):
        url = url.strip()
        if url and url.startswith('http'):
            urls.append(url)
    
    print(f"Parsed {len(urls)} URLs from semicolon-separated string")
    for i, url in enumerate(urls):
        print(f"  URL {i+1}: {url[:60]}...")
    
    return urls

def process_combined_images(input_data, group_number):
    """Process combined images (groups 3, 4, 5) - unchanged"""
    print(f"Processing combined images for group {group_number}")
    print(f"Available input keys: {list(input_data.keys())}")
    
    images = []
    main_keys = {
        3: ['image3', 'image'],
        4: ['image4', 'image'],
        5: ['image5', 'image']
    }
    
    urls_found = False
    
    for key in main_keys.get(group_number, []):
        if key in input_data and input_data[key]:
            value = input_data[key]
            print(f"Checking key '{key}' with value type: {type(value)}")
            
            if isinstance(value, str):
                value = value.strip()
                
                if ';' in value:
                    print(f"Found semicolon-separated URLs in {key}")
                    urls = parse_semicolon_separated_urls(value)
                    
                    if len(urls) >= 2:
                        for i, url in enumerate(urls[:2]):
                            try:
                                print(f"Downloading image {i+1} from URL...")
                                img = download_image_from_google_drive(url)
                                images.append(img)
                                print(f"Successfully downloaded image {i+1}")
                            except Exception as e:
                                print(f"Failed to download image {i+1}: {e}")
                        
                        if len(images) == 2:
                            urls_found = True
                            break
                    else:
                        print(f"WARNING: Expected 2 URLs but found {len(urls)}")
                else:
                    print(f"Found single URL in {key}, looking for second image...")
                    try:
                        img = download_image_from_google_drive(value)
                        images.append(img)
                    except Exception as e:
                        print(f"Failed to download single URL: {e}")
    
    if not urls_found and len(images) < 2:
        print("No semicolon-separated URLs found, trying individual keys...")
        
        key_pairs = {
            3: ['image3', 'image4'],
            4: ['image5', 'image6'],
            5: ['image7', 'image8']
        }
        
        for key in key_pairs.get(group_number, []):
            if key in input_data and input_data[key] and len(images) < 2:
                try:
                    print(f"Trying to get image from key: {key}")
                    img = get_image_from_input({key: input_data[key]})
                    images.append(img)
                    print(f"Successfully got image from {key}")
                except Exception as e:
                    print(f"Failed to get image from {key}: {e}")
    
    if len(images) != 2:
        print(f"ERROR: Group {group_number} requires exactly 2 images, but {len(images)} found")
        raise ValueError(f"Group {group_number} requires exactly 2 images, but {len(images)} found")
    
    print(f"Successfully loaded 2 images for group {group_number}")
    
    target_width = FIXED_WIDTH
    image_height = int(target_width * 1.3)
    
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    IMAGE_SPACING = 200
    
    total_height = TOP_MARGIN + (2 * image_height) + IMAGE_SPACING + BOTTOM_MARGIN
    
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    
    current_y = TOP_MARGIN
    
    for idx, img in enumerate(images):
        try:
            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            img_resized = img.resize((target_width, image_height), resample_filter)
            
            detail_page.paste(img_resized, (0, current_y))
            
            current_y += image_height
            if idx == 0:
                current_y += IMAGE_SPACING
            
            img.close()
            print(f"Placed image {idx + 1} at y={current_y - image_height - (IMAGE_SPACING if idx == 1 else 0)}")
            
        except Exception as e:
            print(f"Error processing image {idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    draw = ImageDraw.Draw(detail_page)
    page_texts = {
        3: "- Images 3-4 -",
        4: "- Images 5-6 -", 
        5: "- Images 7-8 -"
    }
    page_text = page_texts.get(group_number, f"- Details {group_number} -")
    
    korean_font_path = download_korean_font()
    small_font = get_font(16, korean_font_path)
    
    text_width, _ = get_text_size(draw, page_text, small_font)
    safe_draw_text(draw, (FIXED_WIDTH//2 - text_width//2, total_height - 30), 
                 page_text, small_font, (200, 200, 200))
    
    return detail_page

# ... (나머지 함수들은 동일하게 유지)

def check_if_already_transparent(image):
    """Check if image already has transparency"""
    if image.mode != 'RGBA':
        return False
    
    alpha = np.array(image.split()[3])
    transparent_pixels = np.sum(alpha < 250)
    total_pixels = alpha.size
    transparency_ratio = transparent_pixels / total_pixels
    
    print(f"Transparency check: {transparency_ratio:.2%} of pixels are transparent")
    return transparency_ratio > 0.1

def remove_background_from_image(image, skip_if_transparent=False):
    """Remove background from image"""
    try:
        if skip_if_transparent and check_if_already_transparent(image):
            print("Image already has transparency, skipping background removal")
            return image
        
        if REMBG_AVAILABLE:
            try:
                print("Removing background using local rembg...")
                
                if not hasattr(remove_background_from_image, 'session'):
                    remove_background_from_image.session = new_session('u2netp')
                
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                
                output = remove(
                    buffered.getvalue(),
                    session=remove_background_from_image.session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=270,
                    alpha_matting_background_threshold=50,
                    alpha_matting_erode_size=1,
                    only_mask=False
                )
                
                result_image = Image.open(BytesIO(output))
                result_image = moderate_ring_transparency(result_image)
                
                print("Background removed successfully")
                return result_image
                
            except Exception as e:
                print(f"Local rembg failed: {e}")
        
        if REPLICATE_AVAILABLE and REPLICATE_CLIENT:
            try:
                print("Removing background using Replicate API...")
                
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                img_data_url = f"data:image/png;base64,{img_base64}"
                
                output = REPLICATE_CLIENT.run(
                    "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
                    input={
                        "image": img_data_url,
                        "model": "u2netp",
                        "alpha_matting": True,
                        "alpha_matting_foreground_threshold": 270,
                        "alpha_matting_background_threshold": 50,
                        "alpha_matting_erode_size": 1
                    }
                )
                
                if output:
                    if isinstance(output, str):
                        response = requests.get(output)
                        result_image = Image.open(BytesIO(response.content))
                    else:
                        result_image = Image.open(BytesIO(base64.b64decode(output)))
                    
                    result_image = moderate_ring_transparency(result_image)
                    
                    print("Background removed successfully with Replicate")
                    return result_image
                    
            except Exception as e:
                print(f"Replicate background removal failed: {e}")
        
        print("Using manual background removal")
        result = manual_remove_background(image)
        return moderate_ring_transparency(result)
        
    except Exception as e:
        print(f"All background removal methods failed: {e}")
        return image

def moderate_ring_transparency(image):
    """Moderate post-process for ring center hole detection"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    data = np.array(image)
    alpha_channel = data[:,:,3]
    
    height, width = data.shape[:2]
    center_y, center_x = height // 2, width // 2
    
    gray = np.mean(data[:,:,:3], axis=2)
    
    bright_threshold = 245
    bright_mask = gray > bright_threshold
    opaque_bright = bright_mask & (alpha_channel > 200)
    
    labeled, num_features = ndimage.label(opaque_bright)
    
    for i in range(1, num_features + 1):
        region = labeled == i
        region_coords = np.where(region)
        
        if len(region_coords[0]) > 10:
            region_center_y = np.mean(region_coords[0])
            region_center_x = np.mean(region_coords[1])
            region_size = len(region_coords[0])
            
            dist_from_center = np.sqrt((region_center_y - center_y)**2 + (region_center_x - center_x)**2)
            
            is_centered = dist_from_center < min(height, width) * 0.3
            is_reasonable_size = region_size < (height * width * 0.1)
            
            dilated = ndimage.binary_dilation(region, iterations=5)
            touches_edge = (dilated[0,:].any() or dilated[-1,:].any() or 
                           dilated[:,0].any() or dilated[:,-1].any())
            
            if is_centered and is_reasonable_size and not touches_edge:
                region_colors = data[region][:,:3]
                color_std = np.std(region_colors)
                
                if color_std < 10:
                    data[region] = [255, 255, 255, 0]
                    print(f"Removed uniform bright region (size: {region_size})")
    
    return Image.fromarray(data, 'RGBA')

def manual_remove_background(image):
    """Manual background removal for jewelry"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    data = np.array(image, dtype=np.float32)
    
    white_mask = (data[:,:,0] > 250) & (data[:,:,1] > 250) & (data[:,:,2] > 250)
    near_white = (data[:,:,0] > 240) & (data[:,:,1] > 240) & (data[:,:,2] > 240)
    
    max_diff = 15
    color_diff = np.abs(data[:,:,0] - data[:,:,1]) + np.abs(data[:,:,1] - data[:,:,2])
    gray_mask = color_diff < max_diff
    
    background_mask = white_mask | (near_white & gray_mask)
    data[background_mask] = [255, 255, 255, 0]
    
    return Image.fromarray(data.astype(np.uint8), 'RGBA')

def auto_crop_transparent(image):
    """Auto-crop transparent borders from image with padding"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    data = np.array(image)
    alpha = data[:,:,3]
    
    non_transparent = np.where(alpha > 10)
    
    if len(non_transparent[0]) == 0:
        return image
    
    min_y = non_transparent[0].min()
    max_y = non_transparent[0].max()
    min_x = non_transparent[1].min()
    max_x = non_transparent[1].max()
    
    padding = 10
    min_y = max(0, min_y - padding)
    max_y = min(data.shape[0] - 1, max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(data.shape[1] - 1, max_x + padding)
    
    cropped = image.crop((min_x, min_y, max_x + 1, max_y + 1))
    return cropped

def create_ai_generated_md_talk(claude_text, width=FIXED_WIDTH):
    """Create MD Talk section with dynamic height based on content"""
    korean_font_path = download_korean_font()
    title_font = get_font(48, korean_font_path)
    body_font = get_font(28, korean_font_path)
    
    temp_img = Image.new('RGB', (width, 1000), '#FFFFFF')
    draw = ImageDraw.Draw(temp_img)
    
    title = "MD TALK"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    if claude_text:
        text = clean_claude_text(claude_text)
        text = text.replace('MD TALK', '').replace('MD Talk', '').strip()
        
        words = text.split()
        lines = []
        current_line = ""
        max_line_width = width - 120
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_size(draw, test_line, body_font)
            
            if test_width > max_line_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
    else:
        lines = [
            "이 제품은 일상에서도 부담없이",
            "착용할 수 있는 편안한 디자인으로",
            "매일의 스타일링에 포인트를 더해줍니다.",
            "",
            "특별한 날은 물론 평범한 일상까지",
            "모든 순간을 빛나게 만들어주는",
            "당신만의 특별한 주얼리입니다."
        ]
    
    top_margin = 60
    title_bottom_margin = 140
    line_height = 50
    bottom_margin = 80
    
    content_height = len(lines) * line_height
    total_height = top_margin + title_height + title_bottom_margin + content_height + bottom_margin
    
    section_img = Image.new('RGB', (width, total_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    safe_draw_text(draw, (width//2 - title_width//2, top_margin), title, title_font, (40, 40, 40))
    
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    return section_img

def create_ai_generated_design_point(claude_text, width=FIXED_WIDTH):
    """Create Design Point section with dynamic height based on content"""
    korean_font_path = download_korean_font()
    title_font = get_font(48, korean_font_path)
    body_font = get_font(24, korean_font_path)
    
    temp_img = Image.new('RGB', (width, 1000), '#FFFFFF')
    draw = ImageDraw.Draw(temp_img)
    
    title = "DESIGN POINT"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    if claude_text:
        text = clean_claude_text(claude_text)
        text = text.replace('DESIGN POINT', '').replace('Design Point', '').strip()
        
        words = text.split()
        lines = []
        current_line = ""
        max_line_width = width - 100
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_size(draw, test_line, body_font)
            
            if test_width > max_line_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
    else:
        lines = [
            "남성 단품은 무광 텍스처와 유광 라인의 조화가",
            "견고한 감성을 전하고 여자 단품은",
            "파베 세팅과 섬세한 밀그레인의 디테일",
            "화려하면서도 고급스러운 반영을 표현합니다"
        ]
    
    top_margin = 60
    title_bottom_margin = 160
    line_height = 55
    bottom_margin = 100
    
    content_height = len(lines) * line_height
    total_height = top_margin + title_height + title_bottom_margin + content_height + bottom_margin
    
    section_img = Image.new('RGB', (width, total_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    safe_draw_text(draw, (width//2 - title_width//2, top_margin), title, title_font, (40, 40, 40))
    
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    draw.rectangle([100, y_pos + 30, width - 100, y_pos + 32], fill=(220, 220, 220))
    
    return section_img

def create_color_options_section(ring_image=None):
    """Create COLOR section with English labels and enhanced colors"""
    width = FIXED_WIDTH
    height = 850
    
    section_img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    korean_font_path = download_korean_font()
    title_font = get_font(56, korean_font_path)
    label_font = get_font(24, korean_font_path)
    
    title = "COLOR"
    title_width, _ = get_text_size(draw, title, title_font)
    safe_draw_text(draw, (width//2 - title_width//2, 60), title, title_font, (40, 40, 40))
    
    ring_no_bg = None
    if ring_image:
        try:
            print("Processing ring for color section")
            ring_no_bg = remove_background_from_image(ring_image, skip_if_transparent=True)
            
            if ring_no_bg.mode != 'RGBA':
                ring_no_bg = ring_no_bg.convert('RGBA')
            
            ring_no_bg = auto_crop_transparent(ring_no_bg)
                
        except Exception as e:
            print(f"Failed to process ring: {e}")
            ring_no_bg = ring_image.convert('RGBA') if ring_image else None
    
    colors = [
        ("yellow", "YELLOW", (255, 200, 50), 0.3),
        ("rose", "ROSE", (255, 160, 120), 0.35),
        ("white", "WHITE", (255, 255, 255), 0.0),
        ("antique", "ANTIQUE", (245, 235, 225), 0.1)
    ]
    
    grid_size = 260
    padding = 60
    start_x = (width - (grid_size * 2 + padding)) // 2
    start_y = 160
    
    for i, (color_id, label, color_rgb, strength) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (grid_size + padding)
        y = start_y + row * (grid_size + 100)
        
        container = Image.new('RGBA', (grid_size, grid_size), (252, 252, 252, 255))
        container_draw = ImageDraw.Draw(container)
        
        container_draw.rectangle([0, 0, grid_size-1, grid_size-1], 
                                fill=None, outline=(240, 240, 240), width=1)
        
        if ring_no_bg:
            try:
                ring_copy = ring_no_bg.copy()
                max_size = int(grid_size * 0.7)
                ring_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                ring_tinted = apply_enhanced_metal_color(ring_copy, color_rgb, strength, color_id)
                
                paste_x = (grid_size - ring_tinted.width) // 2
                paste_y = (grid_size - ring_tinted.height) // 2
                container.paste(ring_tinted, (paste_x, paste_y), ring_tinted)
                
            except Exception as e:
                print(f"Error applying color {color_id}: {e}")
        
        section_img.paste(container, (x, y))
        
        label_width, _ = get_text_size(draw, label, label_font)
        safe_draw_text(draw, (x + grid_size//2 - label_width//2, y + grid_size + 20), 
                     label, label_font, (80, 80, 80))
    
    return section_img

def apply_enhanced_metal_color(image, metal_color, strength=0.3, color_id=""):
    """Apply enhanced metal color effect with special handling for white and rose"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    
    r_array = np.array(r, dtype=np.float32)
    g_array = np.array(g, dtype=np.float32)
    b_array = np.array(b, dtype=np.float32)
    a_array = np.array(a)
    
    mask = a_array > 0
    
    if mask.any():
        luminance = (0.299 * r_array + 0.587 * g_array + 0.114 * b_array) / 255.0
        
        metal_r, metal_g, metal_b = [c/255.0 for c in metal_color]
        
        if color_id == "white":
            brightness_boost = 1.05
            r_array[mask] = np.clip(r_array[mask] * brightness_boost, 0, 255)
            g_array[mask] = np.clip(g_array[mask] * brightness_boost, 0, 255)
            b_array[mask] = np.clip(b_array[mask] * brightness_boost, 0, 255)
        
        elif color_id == "rose":
            highlight_mask = luminance > 0.85
            shadow_mask = luminance < 0.15
            midtone_mask = ~highlight_mask & ~shadow_mask & mask
            
            if midtone_mask.any():
                blend_factor = 0.5
                r_array[midtone_mask] = r_array[midtone_mask] * (1 - blend_factor) + (255 * luminance[midtone_mask]) * blend_factor
                g_array[midtone_mask] = g_array[midtone_mask] * (1 - blend_factor) + (160 * luminance[midtone_mask]) * blend_factor
                b_array[midtone_mask] = b_array[midtone_mask] * (1 - blend_factor) + (120 * luminance[midtone_mask]) * blend_factor
            
            if highlight_mask.any():
                r_array[highlight_mask] = np.clip(r_array[highlight_mask] * 0.5 + 255 * 0.5, 0, 255)
                g_array[highlight_mask] = np.clip(g_array[highlight_mask] * 0.5 + 160 * 0.5, 0, 255)
                b_array[highlight_mask] = np.clip(b_array[highlight_mask] * 0.5 + 120 * 0.5, 0, 255)
            
            if shadow_mask.any():
                r_array[shadow_mask] = r_array[shadow_mask] * 0.8 + 50 * 0.2
                g_array[shadow_mask] = g_array[shadow_mask] * 0.8 + 30 * 0.2
                b_array[shadow_mask] = b_array[shadow_mask] * 0.8 + 20 * 0.2
        
        else:
            highlight_mask = luminance > 0.85
            shadow_mask = luminance < 0.15
            midtone_mask = ~highlight_mask & ~shadow_mask & mask
            
            if midtone_mask.any():
                blend_factor = strength * 2.0
                r_array[midtone_mask] = r_array[midtone_mask] * (1 - blend_factor) + (metal_r * 255 * luminance[midtone_mask]) * blend_factor
                g_array[midtone_mask] = g_array[midtone_mask] * (1 - blend_factor) + (metal_g * 255 * luminance[midtone_mask]) * blend_factor
                b_array[midtone_mask] = b_array[midtone_mask] * (1 - blend_factor) + (metal_b * 255 * luminance[midtone_mask]) * blend_factor
            
            if highlight_mask.any():
                tint_factor = strength * 0.5
                r_array[highlight_mask] = r_array[highlight_mask] * (1 - tint_factor) + (metal_r * 255) * tint_factor
                g_array[highlight_mask] = g_array[highlight_mask] * (1 - tint_factor) + (metal_g * 255) * tint_factor
                b_array[highlight_mask] = b_array[highlight_mask] * (1 - tint_factor) + (metal_b * 255) * tint_factor
            
            if shadow_mask.any():
                shadow_tint = strength * 0.2
                r_array[shadow_mask] = r_array[shadow_mask] * (1 - shadow_tint) + (metal_r * r_array[shadow_mask]) * shadow_tint
                g_array[shadow_mask] = g_array[shadow_mask] * (1 - shadow_tint) + (metal_g * g_array[shadow_mask]) * shadow_tint
                b_array[shadow_mask] = b_array[shadow_mask] * (1 - shadow_tint) + (metal_b * b_array[shadow_mask]) * shadow_tint
    
    r_array = np.clip(r_array, 0, 255)
    g_array = np.clip(g_array, 0, 255)
    b_array = np.clip(b_array, 0, 255)
    
    r_new = Image.fromarray(r_array.astype(np.uint8))
    g_new = Image.fromarray(g_array.astype(np.uint8))
    b_new = Image.fromarray(b_array.astype(np.uint8))
    
    return Image.merge('RGBA', (r_new, g_new, b_new, a))

def process_color_section(input_data):
    """Process group 6 - COLOR section"""
    print("Processing COLOR section")
    
    img = get_image_from_input(input_data)
    color_section = create_color_options_section(ring_image=img)
    img.close()
    
    return color_section

def process_text_section(input_data, group_number):
    """Process text-only sections (groups 7, 8)"""
    print(f"Processing text section for group {group_number}")
    
    claude_text = (input_data.get('claude_text') or 
                  input_data.get('text_content') or 
                  input_data.get('ai_text') or 
                  input_data.get('generated_text') or '')
    
    if claude_text:
        claude_text = clean_claude_text(claude_text)
    
    text_type = (input_data.get('text_type') or 
                input_data.get('section_type') or '')
    
    print(f"Text type: {text_type}")
    print(f"Cleaned text preview: {claude_text[:100] if claude_text else 'No text'}...")
    
    if group_number == 7 or 'md_talk' in text_type.lower():
        text_section = create_ai_generated_md_talk(claude_text)
        section_type = "md_talk"
    else:
        text_section = create_ai_generated_design_point(claude_text)
        section_type = "design_point"
    
    return text_section, section_type

def detect_group_number_from_input(input_data):
    """Detect group number from input data"""
    route_number = input_data.get('route_number', 0)
    if route_number and str(route_number).isdigit():
        group_num = int(route_number)
        print(f"Found explicit route_number: {group_num}")
        return group_num
    
    group_number = input_data.get('group_number', 0)
    if group_number and str(group_number).isdigit():
        group_num = int(group_number)
        print(f"Found explicit group_number: {group_num}")
        return group_num
    
    text_type = input_data.get('text_type', '').lower()
    if 'md_talk' in text_type:
        return 7
    elif 'design_point' in text_type:
        return 8
    
    for key, group in [('image3', 3), ('image4', 4), ('image5', 5)]:
        if key in input_data and input_data[key]:
            value = input_data[key]
            if isinstance(value, str) and ';' in value:
                print(f"Detected group {group} from semicolon-separated URLs in {key}")
                return group
    
    if 'image6' in input_data or 'image9' in input_data:
        return 6
    
    if 'image1' in input_data:
        return 1
    elif 'image2' in input_data:
        return 2
    elif 'image3' in input_data:
        return 3
    elif 'image4' in input_data:
        return 4
    elif 'image5' in input_data:
        return 5
    
    if any(key in str(input_data).lower() for key in ['color', 'colour', 'gold']):
        return 6
    
    print("No clear group indicators found, defaulting to group 1")
    return 1

def send_to_webhook(image_base64, handler_type, file_name, route_number=0, metadata={}):
    """Send results to Google Apps Script webhook"""
    try:
        if not WEBHOOK_URL:
            print("WARNING: Webhook URL not configured")
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
        
        print(f"Sending to webhook: {handler_type} for {file_name} (route {route_number})")
        
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

def handler(event):
    """Main handler for detail page creation - Updated for text overlay on groups 1 and 2"""
    try:
        print(f"=== V124 Detail Page Handler - With Claude Text Generation ===")
        
        input_data = event.get('input', event)
        print(f"Input keys: {list(input_data.keys())}")
        
        group_number = detect_group_number_from_input(input_data)
        print(f"Detected group number: {group_number}")
        
        if group_number < 1 or group_number > 8:
            raise ValueError(f"Invalid group number: {group_number}")
        
        if group_number == 1:
            print("=== Processing Group 1: Single image 1 with 'twinkring' text ===")
            detail_page = process_single_image(input_data, group_number)
            page_type = "single_image_1_with_text"
            
        elif group_number == 2:
            print("=== Processing Group 2: Single image 2 with Claude-generated text ===")
            detail_page = process_single_image(input_data, group_number)
            page_type = "single_image_2_with_claude_text"
            
        elif group_number in [3, 4, 5]:
            print(f"=== Processing Group {group_number}: Combined images ===")
            detail_page = process_combined_images(input_data, group_number)
            page_type = "combined"
            
        elif group_number == 6:
            print("=== Processing Group 6: COLOR section ===")
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif group_number in [7, 8]:
            print(f"=== Processing Group {group_number}: Text section ===")
            detail_page, section_type = process_text_section(input_data, group_number)
            page_type = f"text_section_{section_type}"
        
        else:
            raise ValueError(f"Unknown group number: {group_number}")
        
        buffered = BytesIO()
        detail_page.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue())
        
        detail_base64 = img_str.decode('utf-8')
        detail_base64_no_padding = detail_base64.rstrip('=')
        
        print(f"Detail page created: {detail_page.size}")
        print(f"Base64 length: {len(detail_base64_no_padding)} chars")
        
        metadata = {
            "enhanced_image": detail_base64_no_padding,
            "status": "success",
            "page_type": page_type,
            "page_number": group_number,
            "route_number": group_number,
            "dimensions": {
                "width": detail_page.width,
                "height": detail_page.height
            },
            "has_text_overlay": group_number in [1, 2, 7, 8],
            "has_background_removal": group_number == 6,
            "has_claude_generation": group_number == 2,
            "format": "base64_no_padding",
            "version": "V124_WITH_CLAUDE"
        }
        
        file_name = f"detail_group_{group_number}_{page_type}.png"
        webhook_result = send_to_webhook(detail_base64_no_padding, "detail", file_name, group_number, metadata)
        
        return {
            "output": metadata
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "status": "error",
                "version": "V124_WITH_CLAUDE"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("V124 Detail Handler - With Claude Text Generation Started!")
    runpod.serverless.start({"handler": handler})
