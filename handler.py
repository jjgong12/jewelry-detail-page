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

def create_color_options_section(width=860, thumbnail_images=None):
    """Create color options section with 2x2 layout - ONLY for 7-8-9"""
    section_height = 400
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
    h_spacing = 100
    v_spacing = 180
    
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

def process_combined_images(images_data, PAGE_WIDTH=860, IMAGE_HEIGHT=1147, CONTENT_WIDTH=760):
    """Process images 3-4, 5-6, 7-8-9 - WITHOUT any text sections (clean images only)"""
    print(f"Processing {len(images_data)} images for combined layout")
    
    # Calculate total height WITH SPACING between images
    TOP_MARGIN = 100
    BOTTOM_MARGIN = 100
    IMAGE_SPACING = 120  # Spacing between images
    
    # Calculate total height
    total_height = TOP_MARGIN + BOTTOM_MARGIN
    total_height += len(images_data) * IMAGE_HEIGHT
    total_height += (len(images_data) - 1) * IMAGE_SPACING  # Spacing between images
    
    print(f"Creating combined canvas: {PAGE_WIDTH}x{total_height}")
    
    # Create canvas
    detail_page = Image.new('RGB', (PAGE_WIDTH, total_height), '#FFFFFF')
    
    current_y = TOP_MARGIN
    
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
        
        # Check if this is route 5 (7-8-9) and last image
        file_name = img_data.get('file_name', '')
        if len(images_data) == 3 and idx == 2:
            # Check if any file contains thumbnail indicators
            all_files = [img.get('file_name', '') for img in images_data]
            is_thumbnail_group = any('thumb' in f.lower() or '_007' in f or '_008' in f or '_009' in f for f in all_files)
            
            if is_thumbnail_group:
                print("Adding COLOR section on bottom of last image (7-8-9)")
                # Create color section
                color_section = create_color_options_section(PAGE_WIDTH)
                # Paste at bottom of current image
                color_y = current_y + IMAGE_HEIGHT - 400  # 400 is height of color section
                detail_page.paste(color_section, (0, color_y))
        
        current_y += IMAGE_HEIGHT
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
    
    # Determine page text based on file names
    all_files = [img.get('file_name', '') for img in images_data]
    if any('_003' in f or '_004' in f for f in all_files):
        page_text = "- Details 3-4 -"
    elif any('_005' in f or '_006' in f for f in all_files):
        page_text = "- Details 5-6 -"
    elif any('_007' in f or '_008' in f or '_009' in f or 'thumb' in f.lower() for f in all_files):
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
        
        # Add base64 image to the appropriate field
        if handler_type == "detail":
            webhook_data["runpod_result"]["output"]["output"]["enhanced_image"] = image_base64
        else:
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
    """Create jewelry detail page - clean images for Makeshop HTML editing"""
    try:
        print(f"=== V100 Detail Page Handler - Clean Images ===")
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
        
        # Check if this is a combined request
        if 'images' in input_data and isinstance(input_data['images'], list) and len(input_data['images']) > 0:
            # Combined processing (3-4, 5-6, 7-8-9)
            print(f"Processing combined images: {len(input_data['images'])} images")
            
            # Check ALL file names to determine route
            all_files = [img.get('file_name', '') for img in input_data['images']]
            print(f"All file names: {all_files}")
            
            # Determine route based on input data
            route_number = input_data.get('route_number', 0)
            
            # Process combined images WITHOUT text sections
            detail_page = process_combined_images(input_data['images'])
            
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
                "page_number": 0,
                "image_count": len(input_data['images']),
                "dimensions": {
                    "width": detail_page.width,
                    "height": detail_page.height
                },
                "has_text_overlay": False,  # Clean images
                "format": "base64_no_padding",
                "status": "success",
                "version": "V100"
            }
            
            # Send to webhook if configured
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
            "version": "V100"
        }
        
        # Send to webhook if configured
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
                "version": "V100"
            }
        }

# RunPod handler registration
runpod.serverless.start({"handler": handler})
