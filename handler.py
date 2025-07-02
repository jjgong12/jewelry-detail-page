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

# FIXED WIDTH FOR ALL IMAGES
FIXED_WIDTH = 1200

def get_text_dimensions(draw, text, font):
    """Get text dimensions compatible with all PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def create_md_talk_section(width=FIXED_WIDTH):
    """Create MD'Talk section for group 3 (images 3-4)"""
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

def create_design_point_section(width=FIXED_WIDTH):
    """Create DESIGN POINT section for group 4 (images 5-6)"""
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

def create_color_options_section(width=FIXED_WIDTH, ring_image=None):
    """Create COLOR section for group 6 (image 9) - Figma design style"""
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
                label_font = ImageFont.truetype(font_path, 24)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw title - COLOR
    title = "COLOR"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 80), title, font=title_font, fill=(60, 60, 60))
    
    # Color information
    colors = [
        ("Yellow Gold", "#FFD700", (1.1, 1.05, 0.9)),
        ("Rose Gold", "#FFC0CB", (1.1, 0.95, 0.95)),
        ("White Gold", "#E8E8E8", (0.98, 0.98, 1.02)),
        ("White", "#F5F5F5", (1.0, 1.0, 1.0))
    ]
    
    # Image settings for 2x2 grid
    img_size = 400  # Size for each ring image
    h_spacing = 80   # Horizontal spacing between images
    v_spacing = 450  # Vertical spacing (includes label space)
    
    # Calculate starting positions for centered 2x2 grid
    grid_width = 2 * img_size + h_spacing
    start_x = (width - grid_width) // 2
    start_y = 200
    
    # Process ring image if provided
    if ring_image:
        try:
            # Extract the ring (remove background)
            ring_with_bg = extract_ring_from_background(ring_image)
            
            for i, (name, color_hex, color_mult) in enumerate(colors):
                row = i // 2
                col = i % 2
                
                x = start_x + col * (img_size + h_spacing)
                y = start_y + row * v_spacing
                
                # Create a copy and resize
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                ring_resized = ring_with_bg.resize((img_size, img_size), resample_filter)
                
                # Apply color tint
                ring_tinted = apply_metal_color_filter(ring_resized, color_mult)
                
                # Add subtle shadow
                shadow_img = Image.new('RGBA', (img_size + 20, img_size + 20), (255, 255, 255, 0))
                shadow_draw = ImageDraw.Draw(shadow_img)
                shadow_draw.ellipse([10, 10, img_size + 10, img_size + 10], 
                                  fill=(200, 200, 200, 80))
                
                # Paste shadow first
                section_img.paste(shadow_img, (x - 10, y - 5), shadow_img)
                
                # Paste the ring
                if ring_tinted.mode == 'RGBA':
                    section_img.paste(ring_tinted, (x, y), ring_tinted)
                else:
                    section_img.paste(ring_tinted, (x, y))
                
                # Draw label below image
                label_width, _ = get_text_dimensions(draw, name, label_font)
                draw.text((x + img_size//2 - label_width//2, y + img_size + 30), 
                         name, font=label_font, fill=(80, 80, 80))
        
        except Exception as e:
            print(f"Error processing ring image: {e}")
            # Fall back to color circles
            create_color_circles_fallback_figma(section_img, draw, colors, start_x, start_y, 
                                              img_size, h_spacing, v_spacing, label_font)
    else:
        # No image provided, use color circles
        create_color_circles_fallback_figma(section_img, draw, colors, start_x, start_y, 
                                          img_size, h_spacing, v_spacing, label_font)
    
    return section_img

def extract_ring_from_background(img):
    """Extract ring from background - simple version"""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Simple white background removal
    data = img.getdata()
    new_data = []
    
    for item in data:
        # If pixel is close to white, make it transparent
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    
    img.putdata(new_data)
    return img

def apply_metal_color_filter(img, color_multipliers):
    """Apply metal color filter to image"""
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
    else:
        img = img.convert('RGBA')
        r, g, b, a = img.split()
    
    # Apply multipliers
    r = r.point(lambda x: min(255, int(x * color_multipliers[0])))
    g = g.point(lambda x: min(255, int(x * color_multipliers[1])))
    b = b.point(lambda x: min(255, int(x * color_multipliers[2])))
    
    return Image.merge('RGBA', (r, g, b, a))

def create_color_circles_fallback_figma(section_img, draw, colors, start_x, start_y, 
                                       img_size, h_spacing, v_spacing, label_font):
    """Fallback function to create color circles in Figma style"""
    for i, (name, color_hex, _) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (img_size + h_spacing)
        y = start_y + row * v_spacing
        
        # Draw ring shape with metallic effect
        # Outer circle
        draw.ellipse([x, y, x+img_size, y+img_size], 
                    fill=color_hex, outline=(180, 180, 180), width=2)
        
        # Inner circle (hole)
        inner_margin = img_size // 3
        draw.ellipse([x+inner_margin, y+inner_margin, 
                     x+img_size-inner_margin, y+img_size-inner_margin], 
                    fill=(255, 255, 255), outline=(200, 200, 200), width=1)
        
        # Add highlight for metallic effect
        highlight_size = img_size // 8
        draw.ellipse([x+highlight_size, y+highlight_size, 
                     x+highlight_size*3, y+highlight_size*3], 
                    fill=(255, 255, 255, 180))
        
        # Draw label
        label_width, _ = get_text_dimensions(draw, name, label_font)
        draw.text((x + img_size//2 - label_width//2, y + img_size + 30), 
                 name, font=label_font, fill=(80, 80, 80))

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

def calculate_image_height(original_width, original_height, target_width):
    """Calculate proportional height for target width"""
    ratio = target_width / original_width
    return int(original_height * ratio)

def process_single_image(input_data, group_number):
    """Process single image (groups 1, 2)"""
    print(f"Processing single image for group {group_number}")
    
    # Get image
    img = get_image_from_input(input_data)
    print(f"Original image size: {img.size}")
    
    # Calculate proportional height
    new_height = calculate_image_height(img.width, img.height, FIXED_WIDTH)
    
    # Margins
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    TOTAL_HEIGHT = TOP_MARGIN + new_height + BOTTOM_MARGIN
    
    # Create canvas
    detail_page = Image.new('RGB', (FIXED_WIDTH, TOTAL_HEIGHT), '#FFFFFF')
    
    # Resize image
    resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    img_resized = img.resize((FIXED_WIDTH, new_height), resample_filter)
    
    # Paste image
    detail_page.paste(img_resized, (0, TOP_MARGIN))
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
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

def process_combined_images(images_data, group_number):
    """Process combined images (groups 3, 4, 5) with sections"""
    print(f"Processing {len(images_data)} images for group {group_number}")
    
    # Calculate heights
    TOP_MARGIN = 100
    BOTTOM_MARGIN = 100
    IMAGE_SPACING = 120
    
    # Add section heights based on group
    section_height = 0
    if group_number == 3:
        section_height = 500  # MD'Talk section
    elif group_number == 4:
        section_height = 600  # DESIGN POINT section
    
    # Calculate total height
    total_height = TOP_MARGIN + BOTTOM_MARGIN + section_height
    if section_height > 0:
        total_height += 80  # Extra spacing after section
    
    # Add image heights
    for img_data in images_data:
        img = get_image_from_input(img_data)
        img_height = calculate_image_height(img.width, img.height, FIXED_WIDTH)
        total_height += img_height
        img.close()
    
    # Add spacing between images
    total_height += (len(images_data) - 1) * IMAGE_SPACING
    
    print(f"Creating combined canvas: {FIXED_WIDTH}x{total_height}")
    
    # Create canvas
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    
    current_y = TOP_MARGIN
    
    # Add section based on group
    if group_number == 3:
        print("Adding MD'Talk section")
        md_talk_section = create_md_talk_section()
        detail_page.paste(md_talk_section, (0, current_y))
        current_y += 500 + 80
    elif group_number == 4:
        print("Adding DESIGN POINT section")
        design_point_section = create_design_point_section()
        detail_page.paste(design_point_section, (0, current_y))
        current_y += 600 + 80
    
    # Process each image
    for idx, img_data in enumerate(images_data):
        if idx > 0:
            current_y += IMAGE_SPACING
        
        # Get image
        img = get_image_from_input(img_data)
        
        # Calculate proportional height
        img_height = calculate_image_height(img.width, img.height, FIXED_WIDTH)
        
        # Resize image
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        img_resized = img.resize((FIXED_WIDTH, img_height), resample_filter)
        
        # Paste image
        detail_page.paste(img_resized, (0, current_y))
        current_y += img_height
        
        img.close()
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
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
    """Process group 6 - COLOR section with ring image"""
    print("Processing group 6 - COLOR section")
    
    # Get the ring image
    img = get_image_from_input(input_data)
    
    # Create color section with the ring image
    color_section = create_color_options_section(ring_image=img)
    
    img.close()
    
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
        print(f"=== V103 Detail Page Handler - Fixed Width 1200px ===")
        
        # Find input data
        input_data = event.get('input', event)
        print(f"Input keys: {list(input_data.keys())}")
        
        # Get route/group number
        route_number = input_data.get('route_number', 0)
        group_number = input_data.get('group_number', route_number)
        
        # Handle different group types
        if group_number == 6:
            # Group 6: COLOR section with image 9
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif 'images' in input_data and isinstance(input_data['images'], list) and len(input_data['images']) > 0:
            # Groups 3, 4, 5: Combined images
            detail_page = process_combined_images(input_data['images'], group_number)
            
            if group_number == 3:
                page_type = "combined_3_4_mdtalk"
            elif group_number == 4:
                page_type = "combined_5_6_design"
            elif group_number == 5:
                page_type = "combined_7_8"
            else:
                page_type = f"combined_group_{group_number}"
            
        else:
            # Groups 1, 2: Single images
            # Auto-detect from filename
            file_name = input_data.get('file_name', '')
            if '_001' in file_name:
                group_number = 1
            elif '_002' in file_name:
                group_number = 2
            
            detail_page = process_single_image(input_data, group_number)
            page_type = f"single_image_{group_number}"
        
        # Save to base64
        buffer = io.BytesIO()
        detail_page.save(buffer, format='PNG', quality=95, optimize=True)
        buffer.seek(0)
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        result_base64_no_padding = result_base64.rstrip('=')
        
        # Prepare metadata
        metadata = {
            "page_type": page_type,
            "group_number": group_number,
            "dimensions": {
                "width": detail_page.width,
                "height": detail_page.height
            },
            "fixed_width": FIXED_WIDTH,
            "has_text_overlay": group_number in [3, 4, 6],
            "format": "base64_no_padding",
            "status": "success",
            "version": "V103"
        }
        
        # Send to webhook
        webhook_result = send_to_webhook(
            result_base64_no_padding,
            "detail",
            f"group_{group_number}_{page_type}.png",
            group_number,
            metadata
        )
        
        if webhook_result:
            metadata["webhook_result"] = webhook_result
        
        print(f"Successfully created {page_type} with dimensions: {detail_page.width}x{detail_page.height}")
        
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
                "status": "error",
                "version": "V103"
            }
        }

# RunPod handler registration
runpod.serverless.start({"handler": handler})
