import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io
import json
import os
import re
from datetime import datetime

def get_text_dimensions(draw, text, font):
    """Get text dimensions compatible with all PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def create_logo_text(text="twinkring", width=1200, height=150):
    """Create elegant logo text for image 1 ONLY"""
    logo_img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(logo_img)
    
    font = None
    font_size = 72
    font_paths = [
        "/tmp/Playfair_Display.ttf",
        "/tmp/Cormorant_Garamond.ttf",
        "/tmp/EB_Garamond.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
    
    if font is None:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    text_width, text_height = get_text_dimensions(draw, text, font)
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Shadow
    draw.text((x+2, y+2), text, font=font, fill=(200, 200, 200))
    # Main text
    draw.text((x, y), text, font=font, fill=(40, 40, 40))
    
    print(f"Logo created: {text} at position ({x}, {y})")
    
    return logo_img

def create_text_block(text, width=760):
    """Create elegant text block with Korean font support"""
    if not text:
        return Image.new('RGBA', (1, 1), (255, 255, 255, 0))
    
    temp_img = Image.new('RGBA', (width, 500), (255, 255, 255, 0))
    draw = ImageDraw.Draw(temp_img)
    
    font = None
    font_size = 28
    font_paths = [
        "/tmp/NanumMyeongjo.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
    
    if font is None:
        font = ImageFont.load_default()
    
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        text_width, _ = get_text_dimensions(draw, test_line, font)
            
        if text_width > width - 40:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
        else:
            current_line.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    line_height = 40
    text_height = max(len(lines) * line_height + 40, 100)
    
    text_img = Image.new('RGBA', (width, text_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_img)
    
    y = 20
    for line in lines:
        text_width, _ = get_text_dimensions(draw, line, font)
        x = (width - text_width) // 2
        draw.text((x, y), line, font=font, fill=(80, 80, 80))
        y += line_height
    
    return text_img

def create_html_section(html_content="", width=860, height=400):
    """Create HTML-like section (MD TALK) - ONLY for combined 3-4"""
    section_img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    if not html_content:
        html_content = """MD TALK

신부의 부케처럼 풍성하고,
드레스처럼 우아한 분위기를 담은 커플링이에요.
결혼이라는 가장 빛나는 순간을
손끝에 남기고 싶은 분들께 추천드립니다:)"""
    
    font_paths = [
        "/tmp/NanumMyeongjo.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    ]
    
    title_font = None
    body_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 42)
                body_font = ImageFont.truetype(font_path, 24)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    lines = html_content.strip().split('\n')
    y_position = 80
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            y_position += 20
            continue
            
        if i == 0:
            text_width, text_height = get_text_dimensions(draw, line, title_font)
            x = (width - text_width) // 2
            draw.text((x, y_position), line, font=title_font, fill=(40, 40, 40))
            y_position += text_height + 40
        else:
            text_width, text_height = get_text_dimensions(draw, line, body_font)
            x = (width - text_width) // 2
            draw.text((x, y_position), line, font=body_font, fill=(60, 60, 60))
            y_position += text_height + 15
    
    return section_img

def create_design_point_section(design_content="", width=860, height=500):
    """Create DESIGN POINT section - for between 005 and 006"""
    section_img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    if not design_content:
        design_content = """DESIGN POINT

중앙의 꼬임 텍스처가 따뜻한 연결감을 표현하고,
여자 단품은 파베 세팅과 메인 스톤의 화려한 반짝임을,
남자 단품은 하나의 메인 스톤으로 단단한 중심을 상징합니다."""
    
    font_paths = [
        "/tmp/NanumMyeongjo.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
    ]
    
    title_font = None
    body_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 48)  # Larger title
                body_font = ImageFont.truetype(font_path, 26)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    lines = design_content.strip().split('\n')
    y_position = 120  # Start lower for design point
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            y_position += 30
            continue
            
        if i == 0:  # DESIGN POINT title
            text_width, text_height = get_text_dimensions(draw, line, title_font)
            x = (width - text_width) // 2
            draw.text((x, y_position), line, font=title_font, fill=(40, 40, 40))
            y_position += text_height + 60  # More space after title
        else:
            # Body text with line wrapping
            words = line.split()
            current_line = []
            wrapped_lines = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                text_width, _ = get_text_dimensions(draw, test_line, body_font)
                
                if text_width > width - 100:  # Leave margins
                    if current_line:
                        wrapped_lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        wrapped_lines.append(word)
                else:
                    current_line.append(word)
            
            if current_line:
                wrapped_lines.append(' '.join(current_line))
            
            # Draw wrapped lines
            for wrapped_line in wrapped_lines:
                text_width, text_height = get_text_dimensions(draw, wrapped_line, body_font)
                x = (width - text_width) // 2
                draw.text((x, y_position), wrapped_line, font=body_font, fill=(60, 60, 60))
                y_position += text_height + 20
    
    return section_img

def create_color_options_section(width=860, thumbnail_images=None):
    """Create color options section with 2x2 layout - for combined 7-8-9"""
    section_height = 400  # Increased height for 2x2 layout
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    label_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 36)
                label_font = ImageFont.truetype(font_path, 18)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw title
    title = "COLOR"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 40), title, font=title_font, fill=(60, 60, 60))
    
    # Color information
    colors = [
        ("yellow", "#FFD700", "yellow gold"),
        ("rose", "#FFC0CB", "rose gold"),
        ("white", "#E8E8E8", "white gold"),
        ("antique", "#D2B48C", "antique white")
    ]
    
    # Draw color boxes in 2x2 grid
    box_size = 120
    h_spacing = 100  # Horizontal spacing
    v_spacing = 180  # Vertical spacing (includes label)
    
    # Calculate starting positions for centered 2x2 grid
    grid_width = 2 * box_size + h_spacing
    grid_height = 2 * v_spacing
    start_x = (width - grid_width) // 2
    start_y = 100
    
    for i, (name, color, label) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (box_size + h_spacing)
        y = start_y + row * v_spacing
        
        # Draw ring placeholder
        draw.ellipse([x+10, y+10, x+box_size-10, y+box_size-10], 
                    fill=color, outline=(180, 180, 180), width=2)
        draw.ellipse([x+30, y+30, x+box_size-30, y+box_size-30], 
                    fill=(255, 255, 255), outline=None)
        
        # Draw label
        label_width, _ = get_text_dimensions(draw, label, label_font)
        draw.text((x + box_size//2 - label_width//2, y + box_size + 15), 
                 label, font=label_font, fill=(80, 80, 80))
    
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
        # Check for URL first
        image_url = (input_data.get('image_url') or 
                    input_data.get('imageUrl') or 
                    input_data.get('url') or 
                    input_data.get('webContentLink') or '')
        
        if image_url:
            print(f"Found image URL: {image_url}")
            if 'drive.google.com' in image_url or 'docs.google.com' in image_url:
                return download_image_from_google_drive(image_url)
            else:
                response = requests.get(image_url, timeout=30)
                return Image.open(BytesIO(response.content))
        
        # Check for base64
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

def process_combined_images(images_data, html_section_content="", include_color_options=False, 
                          include_md_talk=True, include_design_point=False, design_content="",
                          PAGE_WIDTH=860, IMAGE_HEIGHT=1147, CONTENT_WIDTH=760):
    """Process images combined - MD TALK for 3-4, DESIGN POINT for 5-6, COLOR for 7-8-9"""
    print(f"Processing {len(images_data)} images for combined layout")
    print(f"MD TALK: {include_md_talk}, COLOR: {include_color_options}, DESIGN POINT: {include_design_point}")
    
    # Debug: Print all image data
    for i, img_data in enumerate(images_data):
        print(f"Image {i+1}: {img_data.get('file_name', 'unknown')}")
    
    # Calculate total height WITH SPACING between images
    TOP_MARGIN = 100
    BOTTOM_MARGIN = 100
    IMAGE_SPACING = 80  # IMPORTANT: Space between images
    MD_TALK_HEIGHT = 400 if include_md_talk else 0
    MD_TALK_SPACING = 50 if include_md_talk else 0
    COLOR_SECTION_HEIGHT = 400 if include_color_options else 0
    COLOR_SECTION_SPACING = 50 if include_color_options else 0
    
    total_height = TOP_MARGIN + BOTTOM_MARGIN
    total_height += len(images_data) * IMAGE_HEIGHT
    total_height += (len(images_data) - 1) * IMAGE_SPACING  # Spacing between images
    total_height += MD_TALK_HEIGHT + MD_TALK_SPACING
    total_height += DESIGN_POINT_HEIGHT + DESIGN_POINT_SPACING
    total_height += COLOR_SECTION_HEIGHT + COLOR_SECTION_SPACING
    
    # Add height for Claude advice texts
    for img_data in images_data:
        if img_data.get('claude_advice'):
            total_height += 200
    
    print(f"Creating combined canvas: {PAGE_WIDTH}x{total_height}")
    
    # Create canvas
    detail_page = Image.new('RGB', (PAGE_WIDTH, total_height), '#FFFFFF')
    
    current_y = TOP_MARGIN
    
    # Add MD TALK section at the beginning (only for images 3-4)
    if include_md_talk:
        print("Adding MD TALK section at the top")
        md_talk_section = create_html_section(html_section_content, PAGE_WIDTH, MD_TALK_HEIGHT)
        detail_page.paste(md_talk_section, (0, current_y))
        current_y += MD_TALK_HEIGHT + MD_TALK_SPACING
        print(f"MD TALK added, current_y: {current_y}")
    
    # Process each image
    for idx, img_data in enumerate(images_data):
        print(f"Processing image {idx + 1}/{len(images_data)}: {img_data.get('file_name', 'unknown')}")
        
        # Add spacing between images (except before first image)
        if idx > 0:
            current_y += IMAGE_SPACING
            print(f"Added {IMAGE_SPACING}px spacing, current_y: {current_y}")
        
        # Check if we need to add DESIGN POINT between 005 and 006
        file_name = img_data.get('file_name', '')
        if include_design_point and idx == 1 and '_006' in file_name:
            # Add DESIGN POINT before image 006
            print("Adding DESIGN POINT section between 005 and 006")
            current_y += DESIGN_POINT_SPACING // 2  # Half spacing before
            design_section = create_design_point_section(design_content, PAGE_WIDTH, DESIGN_POINT_HEIGHT)
            detail_page.paste(design_section, (0, current_y))
            current_y += DESIGN_POINT_HEIGHT + DESIGN_POINT_SPACING // 2  # Half spacing after
            print(f"DESIGN POINT added, current_y: {current_y}")
        
        # Get image
        img = get_image_from_input(img_data)
        print(f"Original image size: {img.size}")
        
        # Resize image
        height_ratio = IMAGE_HEIGHT / img.height
        temp_width = int(img.width * height_ratio)
        
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
            
        img_resized = img.resize((temp_width, IMAGE_HEIGHT), resample_filter)
        
        # Center crop if needed
        if temp_width > CONTENT_WIDTH:
            left = (temp_width - CONTENT_WIDTH) // 2
            img_cropped = img_resized.crop((left, 0, left + CONTENT_WIDTH, IMAGE_HEIGHT))
        else:
            img_cropped = img_resized
        
        # Apply enhancement
        claude_advice = img_data.get('claude_advice', '')
        if claude_advice and ('luxury' in claude_advice.lower() or 'premium' in claude_advice.lower() or 
                             '프리미엄' in claude_advice or '럭셔리' in claude_advice):
            enhancer = ImageEnhance.Brightness(img_cropped)
            img_cropped = enhancer.enhance(1.05)
            enhancer = ImageEnhance.Contrast(img_cropped)
            img_cropped = enhancer.enhance(1.1)
        
        # Paste image
        x_position = (PAGE_WIDTH - img_cropped.width) // 2
        detail_page.paste(img_cropped, (x_position, current_y))
        print(f"Pasted image at ({x_position}, {current_y})")
        current_y += IMAGE_HEIGHT
        
        # Add Claude's advice text if exists
        if claude_advice and claude_advice.strip():
            text_img = create_text_block(claude_advice, CONTENT_WIDTH)
            if text_img.width > 1 and text_img.height > 1:
                text_x = (PAGE_WIDTH - text_img.width) // 2
                text_y = current_y + 30
                
                if text_img.mode == 'RGBA':
                    detail_page.paste(text_img, (text_x, text_y), text_img)
                else:
                    detail_page.paste(text_img, (text_x, text_y))
                
                current_y = text_y + text_img.height + 30
    
    # Add color options section if requested (only for images 5-6)
    if include_color_options:
        print("Adding COLOR section at the bottom")
        current_y += COLOR_SECTION_SPACING
        color_section = create_color_options_section(PAGE_WIDTH)
        detail_page.paste(color_section, (0, current_y))
        print(f"COLOR section added at y={current_y}")
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
    if include_md_talk:
        page_text = "- Details 3-4 -"
    elif include_design_point:
        page_text = "- Details 5-6 -"
    elif include_color_options:
        page_text = "- Details 7-8-9 -"
    else:
        page_text = "- Details -"
    
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
    draw.text((PAGE_WIDTH//2 - text_width//2, total_height - 50), 
             page_text, fill=(200, 200, 200), font=small_font)
    
    return detail_page

def handler(event):
    """Create jewelry detail page - individual for 1,2 and combined for 3-6"""
    try:
        print(f"=== Detail Page Handler Started ===")
        
        # Find input data
        input_data = event.get('input', event)
        print(f"Input keys: {list(input_data.keys())}")
        
        # CRITICAL: First check if file_name contains _001 or _002 for individual processing
        file_name = input_data.get('file_name', '')
        if '_001' in file_name or '_002' in file_name:
            print(f"FORCING INDIVIDUAL PROCESSING for {file_name}")
            # Force individual processing even if 'images' is present
            input_data.pop('images', None)
        
        # Check if this is a combined request for images 3-6
        if 'images' in input_data and isinstance(input_data['images'], list) and len(input_data['images']) > 0:
            # Combined processing for images 3-6
            print(f"Processing combined images: {len(input_data['images'])} images")
            
            # CRITICAL: Check ALL file names to determine route
            all_files = [img.get('file_name', '') for img in input_data['images']]
            print(f"All file names: {all_files}")
            
            # Check if ANY file contains specific numbers
            has_001_002 = any('_001' in f or '_002' in f for f in all_files)
            has_005_006 = any('_005' in f or '_006' in f for f in all_files)
            has_003_004 = any('_003' in f or '_004' in f for f in all_files)
            has_007_008_009 = any('_007' in f or '_008' in f or '_009' in f for f in all_files)
            
            # REJECT if trying to combine 001 or 002
            if has_001_002:
                raise ValueError("Images 001 and 002 must be processed individually, not combined!")
            
            # Set flags based on file names
            if has_007_008_009:
                # This is route 5 (images 7-8-9) - three images
                print("Detected as route 5 (images 7-8-9) - Will add COLOR section at bottom")
                include_md = False
                include_colors = True
                include_design_point = False
            elif has_005_006:
                # This is route 4 (images 5-6) - with DESIGN POINT between them
                print("Detected as route 4 (images 5-6) - Will add DESIGN POINT between images")
                include_md = False
                include_colors = False
                include_design_point = True
            elif has_003_004:
                # This is route 3 (images 3-4)
                print("Detected as route 3 (images 3-4) - Will add MD TALK section")
                include_md = True
                include_colors = False
                include_design_point = False
            else:
                # Default based on explicit parameters
                include_colors = input_data.get('include_color_options', False)
                include_md = input_data.get('include_md_talk', True)
                include_design_point = input_data.get('include_design_point', False)
            
            html_content = input_data.get('html_section_content', '')
            design_content = input_data.get('design_content', '')
            
            detail_page = process_combined_images(
                input_data['images'], 
                html_section_content=html_content,
                include_color_options=include_colors,
                include_md_talk=include_md,
                include_design_point=include_design_point,
                design_content=design_content
            )
            
            # Save to base64
            buffer = io.BytesIO()
            detail_page.save(buffer, format='JPEG', quality=95, optimize=True)
            buffer.seek(0)
            result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Remove padding for Make.com
            result_base64_no_padding = result_base64.rstrip('=')
            
            print(f"Successfully created combined detail page")
            
            # Send webhook if provided
            webhook_url = input_data.get('webhook')
            if webhook_url:
                try:
                    webhook_data = {
                        "handler_type": "detail",
                        "file_name": "combined_3_to_9",
                        "runpod_result": {
                            "output": {
                                "detail_page": result_base64_no_padding,
                                "page_type": "combined_" + ("3_4" if include_md else "5_6" if include_design_point else "7_8_9" if include_colors else "unknown"),
                                "image_count": len(input_data['images']),
                                "dimensions": {
                                    "width": detail_page.width,
                                    "height": detail_page.height
                                },
                                "has_md_talk": include_md,
                                "has_color_options": include_colors,
                                "has_design_point": include_design_point,
                                "format": "base64_no_padding"
                            }
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    response = requests.post(
                        webhook_url,
                        json=webhook_data,
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                    
                    print(f"Webhook sent successfully: {response.status_code}")
                    
                    return {
                        "output": {
                            "detail_page": result_base64_no_padding,
                            "page_type": "combined_" + ("3_4" if include_md else "5_6" if include_design_point else "7_8_9" if include_colors else "unknown"),
                            "image_count": len(input_data['images']),
                            "dimensions": {
                                "width": detail_page.width,
                                "height": detail_page.height
                            },
                            "has_md_talk": include_md,
                            "has_color_options": include_colors,
                            "format": "base64_no_padding",
                            "webhook_sent": True,
                            "webhook_status": response.status_code
                        }
                    }
                except Exception as webhook_error:
                    print(f"Webhook error: {str(webhook_error)}")
            
            return {
                "output": {
                    "detail_page": result_base64_no_padding,
                    "page_type": "combined_3_to_6",
                    "image_count": len(input_data['images']),
                    "dimensions": {
                        "width": detail_page.width,
                        "height": detail_page.height
                    },
                    "has_md_talk": include_md,
                    "has_color_options": include_colors,
                    "has_design_point": include_design_point,
                    "format": "base64_no_padding"
                }
            }
        
        # Individual image processing (for images 1 and 2) - NO MD TALK!
        claude_advice = input_data.get('claude_advice', '')
        image_number = int(input_data.get('image_number', 1))
        file_name = input_data.get('file_name', 'unknown.jpg')
        
        # Auto-detect image number from filename if not provided
        if '_001' in file_name:
            image_number = 1
        elif '_002' in file_name:
            image_number = 2
        
        print(f"Processing INDIVIDUAL image: {file_name} (Image #{image_number})")
        print(f"INDIVIDUAL PROCESSING - NO MD TALK, NO COMBINING")
        
        # Get image
        img = get_image_from_input(input_data)
        
        # Design settings based on image number
        if image_number == 1:  # Main hero
            PAGE_WIDTH = 1200
            IMAGE_HEIGHT = 1600
            CONTENT_WIDTH = 1100
            LOGO_HEIGHT = 150  # Space for logo
        elif image_number == 2:  # Sub hero
            PAGE_WIDTH = 1000
            IMAGE_HEIGHT = 1333
            CONTENT_WIDTH = 900
            LOGO_HEIGHT = 0  # No logo for image 2
        else:  # Should not happen
            PAGE_WIDTH = 860
            IMAGE_HEIGHT = 1147
            CONTENT_WIDTH = 760
            LOGO_HEIGHT = 0
        
        # Section heights (simplified)
        TOP_MARGIN = 50
        BOTTOM_MARGIN = 50
        TEXT_HEIGHT = 200 if claude_advice else 0
        
        # Total height - NO MD TALK for individual images!
        TOTAL_HEIGHT = TOP_MARGIN + LOGO_HEIGHT + IMAGE_HEIGHT + TEXT_HEIGHT + BOTTOM_MARGIN
        
        # Create canvas
        detail_page = Image.new('RGB', (PAGE_WIDTH, TOTAL_HEIGHT), '#FFFFFF')
        
        current_y = TOP_MARGIN
        
        # Add logo for image 1 ONLY
        if image_number == 1 and LOGO_HEIGHT > 0:
            print("Creating and adding twinkring logo for image 1")
            logo_img = create_logo_text("twinkring", PAGE_WIDTH, LOGO_HEIGHT)
            detail_page.paste(logo_img, (0, current_y))
            current_y += LOGO_HEIGHT
            print(f"Logo added at y={current_y-LOGO_HEIGHT}, size={PAGE_WIDTH}x{LOGO_HEIGHT}")
        
        # Resize image with aspect ratio
        height_ratio = IMAGE_HEIGHT / img.height
        temp_width = int(img.width * height_ratio)
        
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
            
        img_resized = img.resize((temp_width, IMAGE_HEIGHT), resample_filter)
        
        # Center crop if needed
        if temp_width > CONTENT_WIDTH:
            left = (temp_width - CONTENT_WIDTH) // 2
            img_cropped = img_resized.crop((left, 0, left + CONTENT_WIDTH, IMAGE_HEIGHT))
        else:
            img_cropped = img_resized
        
        # Apply subtle enhancement
        if claude_advice and ('luxury' in claude_advice.lower() or 'premium' in claude_advice.lower() or 
                             '프리미엄' in claude_advice or '럭셔리' in claude_advice):
            enhancer = ImageEnhance.Brightness(img_cropped)
            img_cropped = enhancer.enhance(1.05)
            enhancer = ImageEnhance.Contrast(img_cropped)
            img_cropped = enhancer.enhance(1.1)
        
        # Paste image
        x_position = (PAGE_WIDTH - img_cropped.width) // 2
        detail_page.paste(img_cropped, (x_position, current_y))
        current_y += IMAGE_HEIGHT
        
        # Add Claude's advice text if exists (but NO MD TALK!)
        if claude_advice and claude_advice.strip():
            text_img = create_text_block(claude_advice, CONTENT_WIDTH)
            if text_img.width > 1 and text_img.height > 1:
                text_x = (PAGE_WIDTH - text_img.width) // 2
                text_y = current_y + 30
                
                if text_img.mode == 'RGBA':
                    detail_page.paste(text_img, (text_x, text_y), text_img)
                else:
                    detail_page.paste(text_img, (text_x, text_y))
        
        # Add page indicator
        draw = ImageDraw.Draw(detail_page)
        page_text = f"- {image_number} -"
        
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
        draw.text((PAGE_WIDTH//2 - text_width//2, TOTAL_HEIGHT - 30), 
                 page_text, fill=(200, 200, 200), font=small_font)
        
        # Save to base64
        buffer = io.BytesIO()
        detail_page.save(buffer, format='JPEG', quality=95, optimize=True)
        buffer.seek(0)
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        result_base64_no_padding = result_base64.rstrip('=')
        
        print(f"Successfully created INDIVIDUAL detail page: {PAGE_WIDTH}x{TOTAL_HEIGHT}")
        print(f"Has logo: {image_number == 1}, Has MD TALK: FALSE")
        
        # Send webhook if provided
        webhook_url = input_data.get('webhook')
        if webhook_url:
            try:
                webhook_data = {
                    "handler_type": "detail",
                    "file_name": file_name,
                    "runpod_result": {
                        "output": {
                            "detail_page": result_base64_no_padding,
                            "page_number": image_number,
                            "file_name": file_name,
                            "dimensions": {
                                "width": PAGE_WIDTH,
                                "height": TOTAL_HEIGHT
                            },
                            "has_claude_advice": bool(claude_advice),
                            "has_logo": image_number == 1,
                            "has_md_talk": False,  # ALWAYS FALSE for individual images
                            "format": "base64_no_padding"
                        }
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                response = requests.post(
                    webhook_url,
                    json=webhook_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                print(f"Webhook sent successfully: {response.status_code}")
                
                return {
                    "output": {
                        "detail_page": result_base64_no_padding,
                        "page_number": image_number,
                        "file_name": file_name,
                        "dimensions": {
                            "width": PAGE_WIDTH,
                            "height": TOTAL_HEIGHT
                        },
                        "has_claude_advice": bool(claude_advice),
                        "has_logo": image_number == 1,
                        "has_md_talk": False,
                        "format": "base64_no_padding",
                        "webhook_sent": True,
                        "webhook_status": response.status_code
                    }
                }
            except Exception as webhook_error:
                print(f"Webhook error: {str(webhook_error)}")
        
        return {
            "output": {
                "detail_page": result_base64_no_padding,
                "page_number": image_number,
                "file_name": file_name,
                "dimensions": {
                    "width": PAGE_WIDTH,
                    "height": TOTAL_HEIGHT
                },
                "has_claude_advice": bool(claude_advice),
                "has_logo": image_number == 1,
                "has_md_talk": False,
                "format": "base64_no_padding"
            }
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "output": {
                "error": str(e),
                "error_type": type(e).__name__,
                "file_name": input_data.get('file_name', 'unknown') if 'input_data' in locals() else 'unknown'
            }
        }

# RunPod handler registration
runpod.serverless.start({"handler": handler})
