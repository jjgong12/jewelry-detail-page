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
    """Create elegant logo text for image 1"""
    # RGB로 만들고 흰색 배경 사용
    logo_img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(logo_img)
    
    # Try to use elegant fonts
    font = None
    font_size = 72
    font_paths = [
        "/tmp/Playfair_Display.ttf",  # Elegant serif font
        "/tmp/Cormorant_Garamond.ttf",  # Elegant serif
        "/tmp/EB_Garamond.ttf",  # Classic serif
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
    
    # If no elegant font found, use default with larger size
    if font is None:
        try:
            font = ImageFont.load_default()
            # Try to make it larger if possible
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Center the text
    text_width, text_height = get_text_dimensions(draw, text, font)
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text with subtle shadow for elegance
    # Shadow
    draw.text((x+2, y+2), text, font=font, fill=(200, 200, 200))
    # Main text - 진하고 선명하게
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
    
    # Word wrap
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
    """Create HTML-like section (MD TALK)"""
    section_img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Default MD TALK style content if no content provided
    if not html_content:
        html_content = """MD TALK

신부의 부케처럼 풍성하고,
드레스처럼 우아한 분위기를 담은 커플링이에요.
결혼이라는 가장 빛나는 순간을
손끝에 남기고 싶은 분들께 추천드립니다:)"""
    
    # Font setup
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
    
    # Parse content (simple version)
    lines = html_content.strip().split('\n')
    y_position = 80
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            y_position += 20
            continue
            
        # First line as title
        if i == 0:
            text_width, text_height = get_text_dimensions(draw, line, title_font)
            x = (width - text_width) // 2
            draw.text((x, y_position), line, font=title_font, fill=(40, 40, 40))
            y_position += text_height + 40
        else:
            # Body text
            text_width, text_height = get_text_dimensions(draw, line, body_font)
            x = (width - text_width) // 2
            draw.text((x, y_position), line, font=body_font, fill=(60, 60, 60))
            y_position += text_height + 15
    
    return section_img

def create_color_options_section(width=860, thumbnail_images=None):
    """Create color options section with actual thumbnails or placeholders"""
    section_height = 350
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Title
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
    
    # Draw color boxes
    box_size = 120
    spacing = 50
    total_width = len(colors) * box_size + (len(colors) - 1) * spacing
    start_x = (width - total_width) // 2
    y = 120
    
    for i, (name, color, label) in enumerate(colors):
        x = start_x + i * (box_size + spacing)
        
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
        
        # Extract file ID
        file_id = extract_file_id_from_url(url)
        if not file_id:
            raise ValueError(f"Could not extract file ID from URL: {url}")
        
        print(f"Extracted file ID: {file_id}")
        
        # Try multiple download URLs
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
                
                # Check if we got an image
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
                # Regular URL download
                response = requests.get(image_url, timeout=30)
                return Image.open(BytesIO(response.content))
        
        # Check for base64
        image_base64 = (input_data.get('image_base64') or 
                       input_data.get('base64') or 
                       input_data.get('image_data') or 
                       input_data.get('enhanced_image') or '')
        
        if image_base64:
            print(f"Using base64 data, length: {len(image_base64)}")
            # Remove data URL prefix if present
            if image_base64.startswith('data:'):
                image_base64 = image_base64.split(',')[1]
            
            # Add padding if needed
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
                          include_md_talk=True, PAGE_WIDTH=860, IMAGE_HEIGHT=1147, CONTENT_WIDTH=760):
    """Process images combined - with MD TALK for 3-4, with COLOR for 5-6"""
    print(f"Processing {len(images_data)} images for combined layout")
    print(f"MD TALK: {include_md_talk}, COLOR: {include_color_options}")
    
    # Debug: Print all image data
    for i, img_data in enumerate(images_data):
        print(f"Image {i+1}: {img_data.get('file_name', 'unknown')}")
    
    # Calculate total height (NO SPACING between images)
    TOP_MARGIN = 100
    BOTTOM_MARGIN = 100
    MD_TALK_HEIGHT = 400 if include_md_talk else 0
    MD_TALK_SPACING = 50 if include_md_talk else 0
    COLOR_SECTION_HEIGHT = 350 if include_color_options else 0
    COLOR_SECTION_SPACING = 50 if include_color_options else 0
    
    total_height = TOP_MARGIN + BOTTOM_MARGIN
    total_height += len(images_data) * IMAGE_HEIGHT  # No spacing between images
    total_height += MD_TALK_HEIGHT + MD_TALK_SPACING
    total_height += COLOR_SECTION_HEIGHT + COLOR_SECTION_SPACING
    
    # Add height for Claude advice texts
    for img_data in images_data:
        if img_data.get('claude_advice'):
            total_height += 200  # Text block height
    
    print(f"Creating combined canvas: {PAGE_WIDTH}x{total_height}")
    
    # Create canvas
    detail_page = Image.new('RGB', (PAGE_WIDTH, total_height), '#FFFFFF')
    
    # Current Y position
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
        current_y += IMAGE_HEIGHT  # NO spacing after image
        
        # Add Claude's advice text if exists
        if claude_advice and claude_advice.strip():
            text_img = create_text_block(claude_advice, CONTENT_WIDTH)
            if text_img.width > 1 and text_img.height > 1:
                text_x = (PAGE_WIDTH - text_img.width) // 2
                text_y = current_y + 30  # Small spacing before text
                
                if text_img.mode == 'RGBA':
                    detail_page.paste(text_img, (text_x, text_y), text_img)
                else:
                    detail_page.paste(text_img, (text_x, text_y))
                
                current_y = text_y + text_img.height + 30  # Small spacing after text
    
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
    elif include_color_options:
        page_text = "- Details 5-6 -"
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
        
        # Check if this is a combined request for images 3-6
        if 'images' in input_data and isinstance(input_data['images'], list):
            # Combined processing for images 3-6
            print(f"Processing combined images: {len(input_data['images'])} images")
            
            # Determine type based on parameters or image count
            # Route 3: MD TALK + images 3-4 (no color options)
            # Route 4: images 5-6 + COLOR (no MD TALK)
            
            html_content = input_data.get('html_section_content', '')
            include_colors = input_data.get('include_color_options', False)
            include_md = input_data.get('include_md_talk', True)
            
            # Auto-detect based on file names if not explicitly set
            if len(input_data['images']) >= 1:
                first_file = input_data['images'][0].get('file_name', '')
                print(f"First file name: {first_file}")
                
                # 모든 파일명 출력
                all_files = [img.get('file_name', '') for img in input_data['images']]
                print(f"All files: {all_files}")
                
                if '_005' in first_file or '_006' in first_file:
                    # This is route 4 (images 5-6)
                    print("Detected as route 4 (images 5-6) - Will add COLOR section")
                    include_md = False
                    include_colors = True
                elif '_003' in first_file or '_004' in first_file:
                    # This is route 3 (images 3-4)
                    print("Detected as route 3 (images 3-4) - Will add MD TALK section")
                    include_md = True
                    include_colors = False
            
            detail_page = process_combined_images(
                input_data['images'], 
                html_section_content=html_content,
                include_color_options=include_colors,
                include_md_talk=include_md
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
                        "file_name": "combined_3_to_6",
                        "runpod_result": {
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
                            "page_type": "combined_3_to_6",
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
                    # Return result even if webhook fails
            
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
                    "format": "base64_no_padding"
                }
            }
        
        # Individual image processing (for images 1 and 2)
        # Get parameters
        claude_advice = input_data.get('claude_advice', '')
        image_number = int(input_data.get('image_number', 1))
        file_name = input_data.get('file_name', 'unknown.jpg')
        
        print(f"Processing individual image: {file_name} (Image #{image_number})")
        
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
        else:  # Details (3-6) - but this should not happen in individual mode
            PAGE_WIDTH = 860
            IMAGE_HEIGHT = 1147
            CONTENT_WIDTH = 760
            LOGO_HEIGHT = 0
        
        # Section heights (simplified)
        TOP_MARGIN = 50
        BOTTOM_MARGIN = 50
        TEXT_HEIGHT = 200 if claude_advice else 0
        
        # Total height
        TOTAL_HEIGHT = TOP_MARGIN + LOGO_HEIGHT + IMAGE_HEIGHT + TEXT_HEIGHT + BOTTOM_MARGIN
        
        # Create canvas
        detail_page = Image.new('RGB', (PAGE_WIDTH, TOTAL_HEIGHT), '#FFFFFF')
        
        current_y = TOP_MARGIN
        
        # Add logo for image 1
        if image_number == 1 and LOGO_HEIGHT > 0:
            print("Creating and adding twinkring logo")
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
        
        # Add Claude's advice text
        if claude_advice and claude_advice.strip():
            text_img = create_text_block(claude_advice, CONTENT_WIDTH)
            if text_img.width > 1 and text_img.height > 1:
                text_x = (PAGE_WIDTH - text_img.width) // 2
                text_y = current_y + 30  # Small spacing
                
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
        
        print(f"Successfully created detail page: {PAGE_WIDTH}x{TOTAL_HEIGHT}")
        print(f"Output length: {len(result_base64_no_padding)}")
        
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
                        "format": "base64_no_padding",
                        "webhook_sent": True,
                        "webhook_status": response.status_code
                    }
                }
            except Exception as webhook_error:
                print(f"Webhook error: {str(webhook_error)}")
                # Return result even if webhook fails
        
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
