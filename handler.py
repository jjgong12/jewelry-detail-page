import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import io
import json
import os
import re
from datetime import datetime

# Webhook URL - Google Apps Script Web App URL
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbzOQ7SaTtIXRubvSNXNY53pphacVmJg_XKV5sIyOgxjpDykiWsAHN7ecKFHcygGFrYi/exec"

def get_text_dimensions(draw, text, font):
    """Get text dimensions compatible with all PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def create_md_talk_section(width=860):
    """Create MD'Talk section for route 3 (images 3-4)"""
    section_height = 500
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    body_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 36)
                body_font = ImageFont.truetype(font_path, 20)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Draw title
    title = "MD'Talk"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 50), title, font=title_font, fill=(60, 60, 60))
    
    # Draw subtitle
    subtitle = "편안함을 선물하는 주얼리"
    subtitle_width, _ = get_text_dimensions(draw, subtitle, body_font)
    draw.text((width//2 - subtitle_width//2, 120), subtitle, font=body_font, fill=(100, 100, 100))
    
    # Draw main text
    main_text = [
        "매일 착용해도 부담스럽지 않은",
        "가볍고 편안한 착용감",
        "일상 속에서 빛나는 특별함"
    ]
    
    y_pos = 200
    for line in main_text:
        line_width, _ = get_text_dimensions(draw, line, body_font)
        draw.text((width//2 - line_width//2, y_pos), line, font=body_font, fill=(80, 80, 80))
        y_pos += 50
    
    # Draw decorative element
    draw.rectangle([width//2 - 100, 380, width//2 + 100, 382], fill=(200, 200, 200))
    
    return section_img

def create_design_point_section(width=860):
    """Create DESIGN POINT section for route 4 (images 5-6)"""
    section_height = 600
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    subtitle_font = None
    body_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 42)
                subtitle_font = ImageFont.truetype(font_path, 24)
                body_font = ImageFont.truetype(font_path, 18)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Draw title
    title = "DESIGN POINT"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 50), title, font=title_font, fill=(40, 40, 40))
    
    # Draw design points
    points = [
        {
            "title": "Point 1. 우아한 곡선 디자인",
            "desc": "자연스러운 곡선이 만들어내는 부드러운 실루엣"
        },
        {
            "title": "Point 2. 섬세한 디테일",
            "desc": "정교한 세공으로 완성된 고급스러운 마감"
        },
        {
            "title": "Point 3. 편안한 착용감",
            "desc": "인체공학적 설계로 하루 종일 편안함"
        }
    ]
    
    y_pos = 150
    for point in points:
        # Draw point title
        draw.text((80, y_pos), point["title"], font=subtitle_font, fill=(60, 60, 60))
        y_pos += 40
        
        # Draw point description
        draw.text((100, y_pos), point["desc"], font=body_font, fill=(100, 100, 100))
        y_pos += 80
    
    # Draw bottom line
    draw.rectangle([80, y_pos - 20, width - 80, y_pos - 18], fill=(220, 220, 220))
    
    return section_img

def create_color_options_only(width=860, thumbnail_image=None):
    """Create standalone color options section for route 6 (image 9 only)"""
    section_height = 600
    section_img = Image.new('RGB', (width, section_height), '#F8F8F8')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    label_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 48)
                label_font = ImageFont.truetype(font_path, 20)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw title
    title = "COLOR OPTIONS"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 60), title, font=title_font, fill=(40, 40, 40))
    
    # Color information with improved layout
    colors = [
        ("yellow", "#FFD700", "Yellow Gold", "옐로우 골드"),
        ("rose", "#FFC0CB", "Rose Gold", "로즈 골드"),
        ("white", "#E8E8E8", "White Gold", "화이트 골드"),
        ("antique", "#D2B48C", "Antique White", "무도금 화이트")
    ]
    
    # Draw color boxes in 2x2 grid with larger size
    box_size = 160
    h_spacing = 140
    v_spacing = 220
    
    # Calculate starting positions for centered 2x2 grid
    grid_width = 2 * box_size + h_spacing
    grid_height = 2 * v_spacing
    start_x = (width - grid_width) // 2
    start_y = 150
    
    for i, (name, color, label_en, label_kr) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (box_size + h_spacing)
        y = start_y + row * v_spacing
        
        # Draw shadow
        shadow_offset = 3
        draw.ellipse([x+shadow_offset, y+shadow_offset, x+box_size+shadow_offset, y+box_size+shadow_offset], 
                    fill=(230, 230, 230), outline=None)
        
        # Draw ring placeholder with gradient effect
        draw.ellipse([x, y, x+box_size, y+box_size], 
                    fill=color, outline=(150, 150, 150), width=2)
        draw.ellipse([x+40, y+40, x+box_size-40, y+box_size-40], 
                    fill=(255, 255, 255), outline=(200, 200, 200), width=1)
        
        # Draw labels (English and Korean)
        label_width, _ = get_text_dimensions(draw, label_en, label_font)
        draw.text((x + box_size//2 - label_width//2, y + box_size + 15), 
                 label_en, font=label_font, fill=(60, 60, 60))
        
        label_kr_width, _ = get_text_dimensions(draw, label_kr, label_font)
        draw.text((x + box_size//2 - label_kr_width//2, y + box_size + 40), 
                 label_kr, font=label_font, fill=(100, 100, 100))
    
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
                    input_data.get('webContentLink') or
                    input_data.get('image') or '')
        
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

def process_combined_images(images_data, route_number, PAGE_WIDTH=860, IMAGE_HEIGHT=1147, CONTENT_WIDTH=760):
    """Process images 3-4, 5-6, 7-8-9 with MD'Talk and DESIGN POINT sections"""
    print(f"Processing {len(images_data)} images for combined layout, route: {route_number}")
    
    # Calculate total height WITH sections and spacing
    TOP_MARGIN = 100
    BOTTOM_MARGIN = 100
    IMAGE_SPACING = 120
    
    # Add section heights based on route
    section_height = 0
    if route_number == 3:
        section_height = 500  # MD'Talk section
    elif route_number == 4:
        section_height = 600  # DESIGN POINT section
    
    # Calculate total height
    total_height = TOP_MARGIN + BOTTOM_MARGIN + section_height
    total_height += len(images_data) * IMAGE_HEIGHT
    total_height += (len(images_data) - 1) * IMAGE_SPACING
    
    print(f"Creating combined canvas: {PAGE_WIDTH}x{total_height}")
    
    # Create canvas
    detail_page = Image.new('RGB', (PAGE_WIDTH, total_height), '#FFFFFF')
    
    current_y = TOP_MARGIN
    
    # Add section based on route
    if route_number == 3:
        print("Adding MD'Talk section at top")
        md_talk_section = create_md_talk_section(PAGE_WIDTH)
        detail_page.paste(md_talk_section, (0, current_y))
        current_y += 500
        current_y += 80  # Extra spacing after section
    elif route_number == 4:
        print("Adding DESIGN POINT section at top")
        design_point_section = create_design_point_section(PAGE_WIDTH)
        detail_page.paste(design_point_section, (0, current_y))
        current_y += 600
        current_y += 80  # Extra spacing after section
    
    # Process each image
    for idx, img_data in enumerate(images_data):
        print(f"Processing image {idx + 1}/{len(images_data)}: {img_data.get('file_name', 'unknown')}")
        
        # Add spacing between images (except before first image)
        if idx > 0:
            current_y += IMAGE_SPACING
            print(f"Added {IMAGE_SPACING}px spacing, current_y: {current_y}")
        
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
        
        # Paste image
        x_position = (PAGE_WIDTH - img_cropped.width) // 2
        detail_page.paste(img_cropped, (x_position, current_y))
        print(f"Pasted image at ({x_position}, {current_y})")
        
        current_y += IMAGE_HEIGHT
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
    
    # Determine page text based on route
    if route_number == 3:
        page_text = "- Details 3-4 -"
    elif route_number == 4:
        page_text = "- Details 5-6 -"
    elif route_number == 5:
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

def process_color_section_only(image_data, PAGE_WIDTH=860):
    """Process image 9 for standalone color section (route 6)"""
    print("Creating standalone color section from image 9")
    
    # Get the thumbnail image
    img = get_image_from_input(image_data)
    
    # Create color section
    color_section = create_color_options_only(PAGE_WIDTH, img)
    
    return color_section

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
        
        # Add base64 image
        webhook_data["runpod_result"]["output"]["output"]["enhanced_image"] = image_base64
        
        print(f"Sending to webhook: {handler_type} for {file_name}")
        print(f"Webhook URL: {WEBHOOK_URL}")
        
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
        import traceback
        traceback.print_exc()
        return None

def handler(event):
    """Create jewelry detail page with text sections"""
    try:
        print(f"=== V101 Detail Page Handler - With Text Sections ===")
        print(f"Webhook URL configured: {WEBHOOK_URL}")
        
        # Find input data
        input_data = event.get('input', event)
        print(f"Input keys: {list(input_data.keys())}")
        
        # Handle semicolon-separated URLs
        if 'image' in input_data and isinstance(input_data['image'], str) and ';' in input_data['image']:
            print(f"Processing semicolon-separated URLs")
            urls = input_data['image'].split(';')
            input_data['images'] = []
            for i, url in enumerate(urls):
                url = url.strip()
                if url:
                    input_data['images'].append({
                        'url': url,
                        'file_name': f'image_{i+1}.jpg'
                    })
            print(f"Created {len(input_data['images'])} images from semicolon-separated URLs")
        
        # Check if file_name contains _001 or _002 for individual processing
        file_name = input_data.get('file_name', '')
        if '_001' in file_name or '_002' in file_name:
            print(f"FORCING INDIVIDUAL PROCESSING for {file_name}")
            input_data.pop('images', None)
        
        # Get route number
        route_number = input_data.get('route_number', 0)
        
        # Check if this is route 6 (color section only)
        if route_number == 6:
            print("Processing route 6 - Color section only")
            
            # Get image 9 data
            if 'images' in input_data and len(input_data['images']) > 0:
                image_data = input_data['images'][0]  # Should only have one image
            else:
                image_data = input_data
            
            # Create color section
            detail_page = process_color_section_only(image_data)
            
            # Save to base64
            buffer = io.BytesIO()
            detail_page.save(buffer, format='PNG', quality=95, optimize=True)
            buffer.seek(0)
            result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Remove padding for Make.com
            result_base64_no_padding = result_base64.rstrip('=')
            
            # Prepare metadata
            metadata = {
                "page_type": "color_section",
                "page_number": 6,
                "route_number": 6,
                "dimensions": {
                    "width": detail_page.width,
                    "height": detail_page.height
                },
                "has_text_overlay": True,
                "format": "base64_no_padding",
                "status": "success",
                "version": "V101"
            }
            
            # Send to webhook
            webhook_result = send_to_webhook(
                result_base64_no_padding,
                "detail",
                "color_section.png",
                6,
                metadata
            )
            
            if webhook_result:
                metadata["webhook_result"] = webhook_result
            
            print("Successfully created color section")
            
            return {
                "output": {
                    "enhanced_image": result_base64_no_padding,
                    **metadata
                }
            }
        
        # Check if this is a combined request
        if 'images' in input_data and isinstance(input_data['images'], list) and len(input_data['images']) > 0:
            # Combined processing (3-4, 5-6, 7-8-9)
            print(f"Processing combined images: {len(input_data['images'])} images")
            
            # Check ALL file names to determine route
            all_files = [img.get('file_name', '') for img in input_data['images']]
            print(f"All file names: {all_files}")
            
            # Process combined images WITH text sections
            detail_page = process_combined_images(input_data['images'], route_number)
            
            # Save to base64
            buffer = io.BytesIO()
            detail_page.save(buffer, format='PNG', quality=95, optimize=True)
            buffer.seek(0)
            result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Remove padding for Make.com
            result_base64_no_padding = result_base64.rstrip('=')
            
            # Determine page type
            if any('_003' in f or '_004' in f for f in all_files) or route_number == 3:
                page_type = "combined_3_4"
            elif any('_005' in f or '_006' in f for f in all_files) or route_number == 4:
                page_type = "combined_5_6"
            elif any('_007' in f or '_008' in f or '_009' in f or 'thumb' in f.lower() for f in all_files) or route_number == 5:
                page_type = "combined_7_8_9"
            else:
                page_type = "combined_unknown"
            
            # Prepare metadata
            metadata = {
                "page_type": page_type,
                "page_number": route_number,
                "image_count": len(input_data['images']),
                "dimensions": {
                    "width": detail_page.width,
                    "height": detail_page.height
                },
                "has_text_overlay": route_number in [3, 4],  # MD'Talk and DESIGN POINT
                "format": "base64_no_padding",
                "status": "success",
                "version": "V101"
            }
            
            # Send to webhook
            webhook_result = send_to_webhook(
                result_base64_no_padding,
                "detail",
                f"combined_{route_number}.png",
                route_number,
                metadata
            )
            
            if webhook_result:
                metadata["webhook_result"] = webhook_result
            
            print(f"Successfully created combined detail page")
            
            return {
                "output": {
                    "enhanced_image": result_base64_no_padding,
                    **metadata
                }
            }
        
        # Individual image processing (for images 1 and 2) - NO TEXT OVERLAY
        image_number = int(input_data.get('image_number', 1))
        file_name = input_data.get('file_name', 'unknown.jpg')
        
        # Auto-detect image number from filename
        if '_001' in file_name:
            image_number = 1
        elif '_002' in file_name:
            image_number = 2
        
        print(f"Processing INDIVIDUAL image: {file_name} (Image #{image_number}) - CLEAN VERSION")
        
        # Handle single image URL in 'image' field
        if 'image' in input_data and not input_data.get('image_url'):
            input_data['image_url'] = input_data['image']
            print(f"Using 'image' field as image_url: {input_data['image_url']}")
        
        # Get image
        img = get_image_from_input(input_data)
        
        # Design settings based on image number
        if image_number == 1:  # Main hero
            PAGE_WIDTH = 1200
            IMAGE_HEIGHT = 1600
            CONTENT_WIDTH = 1100
        elif image_number == 2:  # Sub hero
            PAGE_WIDTH = 1000
            IMAGE_HEIGHT = 1333
            CONTENT_WIDTH = 900
        else:
            PAGE_WIDTH = 860
            IMAGE_HEIGHT = 1147
            CONTENT_WIDTH = 760
        
        # Section heights
        TOP_MARGIN = 50
        BOTTOM_MARGIN = 50
        
        # Total height
        TOTAL_HEIGHT = TOP_MARGIN + IMAGE_HEIGHT + BOTTOM_MARGIN
        
        # Create canvas
        detail_page = Image.new('RGB', (PAGE_WIDTH, TOTAL_HEIGHT), '#FFFFFF')
        
        current_y = TOP_MARGIN
        
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
        
        # Paste image
        x_position = (PAGE_WIDTH - img_cropped.width) // 2
        detail_page.paste(img_cropped, (x_position, current_y))
        
        # NO TEXT OVERLAY FOR IMAGES 1 AND 2
        print("Clean image created without text overlay")
        
        current_y += IMAGE_HEIGHT
        
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
        detail_page.save(buffer, format='PNG', quality=95, optimize=True)
        buffer.seek(0)
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        result_base64_no_padding = result_base64.rstrip('=')
        
        # Prepare metadata
        metadata = {
            "page_number": image_number,
            "page_type": "individual",
            "file_name": file_name,
            "dimensions": {
                "width": PAGE_WIDTH,
                "height": TOTAL_HEIGHT
            },
            "has_text_overlay": False,  # Clean image
            "format": "base64_no_padding",
            "status": "success",
            "version": "V101"
        }
        
        # Send to webhook
        webhook_result = send_to_webhook(
            result_base64_no_padding,
            "detail",
            file_name,
            image_number,
            metadata
        )
        
        if webhook_result:
            metadata["webhook_result"] = webhook_result
        
        print(f"Successfully created INDIVIDUAL detail page: {PAGE_WIDTH}x{TOTAL_HEIGHT}")
        print(f"Clean image without text overlay")
        
        return {
            "output": {
                "enhanced_image": result_base64_no_padding,
                **metadata
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
                "file_name": input_data.get('file_name', 'unknown') if 'input_data' in locals() else 'unknown',
                "status": "error",
                "version": "V101"
            }
        }

# RunPod handler registration
runpod.serverless.start({"handler": handler})
