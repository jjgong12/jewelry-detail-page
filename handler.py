import runpod
import os
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import re

# REPLICATE API (Optional for background removal)
try:
    import replicate
    REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
    if REPLICATE_API_TOKEN:
        REPLICATE_AVAILABLE = True
    else:
        print("WARNING: REPLICATE_API_TOKEN not set. Background removal will use local method.")
        REPLICATE_AVAILABLE = False
except ImportError:
    print("WARNING: replicate package not installed. Background removal will use local method.")
    REPLICATE_AVAILABLE = False

# Webhook URL - Google Apps Script Web App URL
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzOQ7SaTtIXRubvSNXNY53pphacVmJg_XKV5sIyOgxjpDykiWsAHN7ecKFHcygGFrYi/exec"

# FIXED WIDTH FOR ALL IMAGES
FIXED_WIDTH = 1200

def get_text_dimensions(draw, text, font):
    """Get text dimensions compatible with all PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

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

def clean_claude_text(text):
    """Clean text for safe JSON encoding while preserving Korean characters"""
    if not text:
        return ""
    
    # Convert to string if needed
    text = str(text) if text is not None else ""
    
    # Clean escape sequences while preserving Korean
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    
    # Remove any actual control characters
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    
    # Remove markdown formatting
    text = text.replace('#', '')
    text = text.replace('*', '')
    text = text.replace('_', '')
    text = text.replace('`', '')
    
    # Clean multiple spaces
    text = ' '.join(text.split())
    
    return text.strip()

def create_ai_generated_md_talk(claude_text, width=FIXED_WIDTH):
    """Create AI-generated MD Talk section"""
    section_height = 600
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Download Korean font if needed
    download_korean_font()
    
    # Get fonts
    title_font = None
    body_font = None
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    
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
    
    # Title
    title = "MD TALK"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 60), title, font=title_font, fill=(40, 40, 40))
    
    # Process Claude text
    if claude_text:
        # Remove any MD TALK prefix if exists
        text = claude_text.replace('MD TALK', '').replace('MD Talk', '').strip()
        
        # Split text into lines
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 120:  # 60px margin on each side
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
    else:
        # Default text
        lines = [
            "이 제품은 일상에서도 부담없이",
            "착용할 수 있는 편안한 디자인으로",
            "매일의 스타일링에 포인트를 더해줍니다.",
            "",
            "특별한 날은 물론 평범한 일상까지",
            "모든 순간을 빛나게 만들어주는",
            "당신만의 특별한 주얼리입니다."
        ]
    
    # Draw content lines
    y_pos = 180
    line_height = 40
    
    for line in lines:
        if line:  # Skip empty lines
            line_width, _ = get_text_dimensions(draw, line, body_font)
            draw.text((width//2 - line_width//2, y_pos), line, font=body_font, fill=(80, 80, 80))
        y_pos += line_height
    
    return section_img

def create_ai_generated_design_point(claude_text, width=FIXED_WIDTH):
    """Create AI-generated Design Point section"""
    section_height = 700
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Get fonts
    title_font = None
    body_font = None
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    
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
    
    # Title
    title = "DESIGN POINT"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 60), title, font=title_font, fill=(40, 40, 40))
    
    # Process Claude text
    if claude_text:
        # Remove any DESIGN POINT prefix if exists
        text = claude_text.replace('DESIGN POINT', '').replace('Design Point', '').strip()
        
        # Split text into lines with better handling
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_width, _ = get_text_dimensions(draw, test_line, body_font)
            
            if test_width > width - 100:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
    else:
        # Default text
        lines = [
            "남성 단품은 무광 텍스처와 유광 라인의 조화가",
            "견고한 감성을 전하고 여자 단품은",
            "파베 세팅과 섬세한 밀그레인의 디테일",
            "화려하면서도 고급스러운 반영을 표현합니다"
        ]
    
    # Draw content lines
    y_pos = 200
    line_height = 45
    
    for line in lines:
        if line:
            line_width, _ = get_text_dimensions(draw, line, body_font)
            draw.text((width//2 - line_width//2, y_pos), line, font=body_font, fill=(80, 80, 80))
        y_pos += line_height
    
    # Add decorative line at bottom
    draw.rectangle([100, y_pos + 30, width - 100, y_pos + 32], fill=(220, 220, 220))
    
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
        
        raise Exception(f"Failed to download from all URLs for file ID: {file_id}")
        
    except Exception as e:
        print(f"Error downloading from Google Drive: {str(e)}")
        raise

def get_image_from_input(input_data):
    """Get image from various input formats"""
    try:
        # Check for direct image data
        if 'image' in input_data and input_data['image']:
            image_data = input_data['image']
            if image_data.startswith('http'):
                return download_image_from_google_drive(image_data)
            elif image_data.startswith('data:'):
                header, data = image_data.split(',', 1)
                return Image.open(BytesIO(base64.b64decode(data)))
            else:
                return Image.open(BytesIO(base64.b64decode(image_data)))
        
        # Check for url field
        if 'url' in input_data and input_data['url']:
            return download_image_from_google_drive(input_data['url'])
        
        # Check for enhanced_image (base64)
        if 'enhanced_image' in input_data and input_data['enhanced_image']:
            img_data = input_data['enhanced_image']
            return Image.open(BytesIO(base64.b64decode(img_data)))
        
        raise ValueError("No valid image data found in input")
        
    except Exception as e:
        print(f"Error getting image: {str(e)}")
        raise

def detect_group_number_from_input(input_data):
    """Detect which group this is from input data"""
    # First check for explicit route_number
    route_number = input_data.get('route_number', 0)
    if route_number:
        print(f"Found route_number: {route_number}")
        return int(route_number)
    
    # Check for specific image keys
    if 'image1' in input_data:
        return 1
    elif 'image2' in input_data:
        return 2
    elif 'image3' in input_data and 'image4' in input_data:
        return 3
    elif 'image5' in input_data and 'image6' in input_data:
        return 4
    elif 'image7' in input_data and 'image8' in input_data:
        return 5
    elif 'image9' in input_data:
        return 6
    
    # Check for text type hints
    text_type = input_data.get('text_type', '')
    if 'md_talk' in text_type.lower():
        return 7
    elif 'design_point' in text_type.lower():
        return 8
    
    # Check images array
    if 'images' in input_data:
        images_count = len(input_data['images'])
        if images_count == 1:
            # Could be 1, 2, or 6
            if any(key in str(input_data) for key in ['color', 'COLOR', 'image9']):
                return 6
            return 1  # Default to 1 for single image
        elif images_count == 2:
            # Could be 3, 4, or 5
            if any(key in str(input_data) for key in ['image3', 'image4']):
                return 3
            elif any(key in str(input_data) for key in ['image5', 'image6']):
                return 4
            elif any(key in str(input_data) for key in ['image7', 'image8']):
                return 5
            return 3  # Default to 3 for 2 images
        elif images_count == 3:
            return 5  # 3 images is group 5
    
    return 1  # Default

def create_color_options_section(ring_image=None):
    """Create COLOR section with 4 color options"""
    width = FIXED_WIDTH
    height = 800
    
    section_img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Title
    title_font = None
    label_font = None
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 56)
                label_font = ImageFont.truetype(font_path, 24)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    title = "COLOR"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 60), title, font=title_font, fill=(40, 40, 40))
    
    # Color options
    colors = [
        ("yellow", "옐로우골드", (255, 215, 0)),
        ("rose", "로즈골드", (183, 110, 121)),
        ("white", "화이트골드", (220, 220, 220)),
        ("antique", "무도금화이트", (245, 245, 220))
    ]
    
    # Grid layout - 2x2
    grid_size = 220
    start_x = (width - (grid_size * 2 + 40)) // 2
    start_y = 180
    
    for i, (color_id, label, color_rgb) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (grid_size + 40)
        y = start_y + row * (grid_size + 80)
        
        # Color container
        container_rect = [x, y, x + grid_size, y + grid_size]
        
        # Fill with color
        draw.rectangle(container_rect, fill=color_rgb, outline=(200, 200, 200), width=2)
        
        # If ring image provided, overlay it
        if ring_image:
            try:
                # Create a copy and resize
                ring_copy = ring_image.copy()
                ring_copy.thumbnail((180, 180), Image.Resampling.LANCZOS)
                
                # Center in container
                paste_x = x + (grid_size - ring_copy.width) // 2
                paste_y = y + (grid_size - ring_copy.height) // 2
                
                # Paste with transparency if available
                if ring_copy.mode == 'RGBA':
                    section_img.paste(ring_copy, (paste_x, paste_y), ring_copy)
                else:
                    section_img.paste(ring_copy, (paste_x, paste_y))
            except Exception as e:
                print(f"Error overlaying ring on color {color_id}: {e}")
        
        # Label
        label_width, _ = get_text_dimensions(draw, label, label_font)
        draw.text((x + grid_size//2 - label_width//2, y + grid_size + 15), 
                 label, font=label_font, fill=(80, 80, 80))
    
    return section_img

def process_single_image(input_data, group_number):
    """Process individual images (groups 1, 2)"""
    print(f"Processing single image for group {group_number}")
    
    # Get image
    img = get_image_from_input(input_data)
    
    # Calculate dimensions
    if group_number == 1:
        # Main hero image
        new_height = int(FIXED_WIDTH * 1.375)  # 1200 x 1650
    else:
        # Sub hero image
        new_height = int(FIXED_WIDTH * 1.15)   # 1200 x 1383
    
    # Resize
    resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    img_resized = img.resize((FIXED_WIDTH, new_height), resample_filter)
    
    # Create page with margins
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    total_height = new_height + TOP_MARGIN + BOTTOM_MARGIN
    
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    detail_page.paste(img_resized, (0, TOP_MARGIN))
    
    img.close()
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
    page_text = f"- Image {group_number} -"
    
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
    draw.text((FIXED_WIDTH//2 - text_width//2, total_height - 30), 
             page_text, fill=(200, 200, 200), font=small_font)
    
    return detail_page

def process_combined_images(input_data, group_number):
    """Process combined images WITHOUT text sections (groups 3, 4, 5)"""
    print(f"Processing CLEAN combined images for group {group_number} (NO TEXT SECTIONS)")
    
    # Get images data
    images_data = input_data.get('images', [input_data])
    
    # CRITICAL: Group 5 should only have 2 images (7, 8), NOT 3!
    if group_number == 5 and len(images_data) > 2:
        print(f"WARNING: Group 5 has {len(images_data)} images, using first 2 only")
        images_data = images_data[:2]
    
    # Calculate dimensions for each image
    image_heights = []
    if group_number in [3, 4]:
        # Groups 3, 4: 860px width
        target_width = 860
        for _ in images_data:
            image_heights.append(int(target_width * 1.46))  # 860 x 1256
    else:
        # Group 5: 860px width
        target_width = 860
        for _ in images_data:
            image_heights.append(int(target_width * 1.46))  # 860 x 1256
    
    # Calculate total height WITHOUT text sections
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    IMAGE_SPACING = 200  # 200px between images
    
    total_height = TOP_MARGIN + sum(image_heights) + (len(images_data) - 1) * IMAGE_SPACING + BOTTOM_MARGIN
    
    print(f"Creating CLEAN combined page: {FIXED_WIDTH}x{total_height} (NO TEXT)")
    
    # Create combined page
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    
    current_y = TOP_MARGIN
    
    # Process each image WITHOUT adding text sections
    for idx, (img_data, img_height) in enumerate(zip(images_data, image_heights)):
        if idx > 0:
            current_y += IMAGE_SPACING
        
        # Get image
        img = get_image_from_input(img_data)
        
        # Center crop to target dimensions
        img_width, img_height_orig = img.size
        
        # Calculate crop
        if img_width / img_height_orig > target_width / img_height:
            # Image is wider - crop width
            new_width = int(img_height_orig * target_width / img_height)
            left = (img_width - new_width) // 2
            img_cropped = img.crop((left, 0, left + new_width, img_height_orig))
        else:
            # Image is taller - crop height
            new_height = int(img_width * img_height / target_width)
            top = (img_height_orig - new_height) // 2
            img_cropped = img.crop((0, top, img_width, top + new_height))
        
        # Resize to exact dimensions
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        img_resized = img_cropped.resize((target_width, img_height), resample_filter)
        
        # Paste centered on page
        x_offset = (FIXED_WIDTH - target_width) // 2
        detail_page.paste(img_resized, (x_offset, current_y))
        current_y += img_height
        
        img.close()
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
    if group_number == 5:
        page_text = f"- Gallery 7-8 -"  # 명확하게 7-8만 표시
    else:
        page_text = f"- Details {group_number} -"
    
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
    
    # Get the ring image
    img = get_image_from_input(input_data)
    print(f"Ring image for color section: {img.size}, mode: {img.mode}")
    
    # Create color section with the ring image
    color_section = create_color_options_section(ring_image=img)
    
    img.close()
    
    print("Color section created successfully")
    return color_section

def process_text_section(input_data, group_number):
    """Process text-only sections (groups 7, 8) with Claude-generated content"""
    print(f"Processing text section for group {group_number}")
    
    # Get Claude-generated text and CLEAN IT
    claude_text = (input_data.get('claude_text') or 
                  input_data.get('text_content') or 
                  input_data.get('ai_text') or 
                  input_data.get('generated_text') or '')
    
    # CRITICAL: Clean the text first to prevent JSON errors
    if claude_text:
        claude_text = clean_claude_text(claude_text)
    
    text_type = (input_data.get('text_type') or 
                input_data.get('section_type') or '')
    
    print(f"Text type: {text_type}")
    print(f"Cleaned Claude text preview: {claude_text[:100] if claude_text else 'No text provided'}...")
    
    if group_number == 7 or text_type == 'md_talk':
        # Group 7: MD Talk text section
        text_section = create_ai_generated_md_talk(claude_text)
        section_type = "md_talk"
    elif group_number == 8 or text_type == 'design_point':
        # Group 8: Design Point text section
        text_section = create_ai_generated_design_point(claude_text)
        section_type = "design_point"
    else:
        # Fallback - determine by group number
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

def handler(event):
    """Main handler for detail page creation"""
    try:
        print(f"=== V110 Detail Page Handler - 8 Groups System ===")
        
        # Find input data
        input_data = event.get('input', event)
        print(f"Input keys: {list(input_data.keys())}")
        
        # Detect group number
        group_number = detect_group_number_from_input(input_data)
        print(f"Detected group number: {group_number}")
        
        if group_number < 1 or group_number > 8:
            raise ValueError(f"Invalid group number: {group_number}. Must be 1-8.")
        
        # Process based on group number
        if group_number == 6:
            # Group 6: COLOR section ONLY (using image 9)
            print("=== Processing Group 6: COLOR section (image 9) ===")
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif group_number in [7, 8]:
            # Groups 7, 8: Text-only sections with CLEANED TEXT
            print(f"=== Processing Group {group_number}: Text-only section ===")
            detail_page, section_type = process_text_section(input_data, group_number)
            page_type = f"text_section_{section_type}"
            
        elif group_number in [1, 2]:
            # Groups 1, 2: Individual images
            print(f"=== Processing Group {group_number}: Individual image ===")
            detail_page = process_single_image(input_data, group_number)
            page_type = "individual"
            
        elif group_number in [3, 4, 5]:
            # Groups 3, 4, 5: CLEAN Combined images (NO TEXT SECTIONS!)
            print(f"=== Processing Group {group_number}: CLEAN Combined images (NO TEXT) ===")
            if 'images' not in input_data or not isinstance(input_data['images'], list):
                # Convert single image to images array
                input_data['images'] = [input_data]
            
            # CRITICAL: Group 5 should only have 2 images (7, 8), NOT 3!
            if group_number == 5 and len(input_data['images']) > 2:
                print(f"ERROR: Group 5 has {len(input_data['images'])} images, should have 2. Using first 2 only.")
                input_data['images'] = input_data['images'][:2]
            
            detail_page = process_combined_images(input_data, group_number)
            page_type = "combined_clean"
        
        else:
            raise ValueError(f"Unknown group number: {group_number}")
        
        # Convert to base64
        buffered = BytesIO()
        detail_page.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue())
        
        detail_base64 = img_str.decode('utf-8')
        detail_base64_no_padding = detail_base64.rstrip('=')
        
        print(f"Detail page created: {detail_page.size}")
        print(f"Base64 length: {len(detail_base64_no_padding)} chars")
        
        # Prepare metadata
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
            "has_text_overlay": group_number in [7, 8],
            "format": "base64_no_padding",
            "version": "V110_8_GROUPS"
        }
        
        # Send to webhook
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
                "version": "V110_8_GROUPS"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("V110 Detail Handler Started - 8 Groups System!")
    runpod.serverless.start({"handler": handler})
