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

def get_text_dimensions(draw, text, font):
    """Get text dimensions compatible with all PIL versions"""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)

def create_md_talk_section(width=FIXED_WIDTH):
    """Create MD'Talk section for group 3 (images 3-4)"""
    section_height = 400
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
    
    # Draw MD TALK title
    title = "MD TALK"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 100), title, font=title_font, fill=(40, 40, 40))
    
    # Draw Korean text
    korean_text = [
        "고급스러운 텍스처와 균형 잡힌 디테일이",
        "감성의 깊이를 더하는 커플링입니다.",
        "'섬세한 연결'을 느끼고 싶은 커플에게 추천드립니다."
    ]
    
    y_pos = 200
    for line in korean_text:
        line_width, _ = get_text_dimensions(draw, line, body_font)
        draw.text((width//2 - line_width//2, y_pos), line, font=body_font, fill=(80, 80, 80))
        y_pos += 40
    
    return section_img

def create_design_point_section(width=FIXED_WIDTH):
    """Create DESIGN POINT section for group 4 (images 5-6)"""
    section_height = 500
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    font_paths = ["/tmp/NanumMyeongjo.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]
    title_font = None
    body_font = None
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                title_font = ImageFont.truetype(font_path, 48)
                body_font = ImageFont.truetype(font_path, 18)
                break
            except:
                continue
    
    if title_font is None:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Draw DESIGN POINT title
    title = "DESIGN POINT"
    title_width, _ = get_text_dimensions(draw, title, title_font)
    draw.text((width//2 - title_width//2, 80), title, font=title_font, fill=(40, 40, 40))
    
    # Draw Korean text
    korean_text = [
        "리프링 무광 텍스처와 유광 라인의 조화가 견고한 감성을 전하고,",
        "여자 단품은 파베 세팅과 섬세한 밀그레인의 디테일",
        "화려하면서도 고급스러운 반영영을 표현합니다.",
        "메인 스톤이 두 반지를 하나의 결로 이어주는 상징이 됩니다."
    ]
    
    y_pos = 200
    for line in korean_text:
        line_width, _ = get_text_dimensions(draw, line, body_font)
        draw.text((width//2 - line_width//2, y_pos), line, font=body_font, fill=(80, 80, 80))
        y_pos += 45
    
    return section_img

def extract_ring_with_replicate(img):
    """Extract ring from background using Replicate API"""
    try:
        # Check if Replicate is available and API token is set
        if not REPLICATE_AVAILABLE:
            print("Replicate not available, using local fallback")
            return extract_ring_local_fallback(img)
            
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print("Replicate API token not found, using local fallback")
            return extract_ring_local_fallback(img)
            
        print("Starting Replicate background removal...")
        
        # Convert image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Call Replicate API for background removal
        output = replicate.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            input={
                "image": f"data:image/png;base64,{img_base64}"
            }
        )
        
        # Download the result
        response = requests.get(output)
        result_img = Image.open(BytesIO(response.content))
        
        # Ensure RGBA mode
        if result_img.mode != 'RGBA':
            result_img = result_img.convert('RGBA')
        
        print("Replicate background removal completed successfully")
        return result_img
        
    except Exception as e:
        print(f"Error with Replicate API: {e}")
        print("Falling back to local method...")
        
        # Fallback to local method
        return extract_ring_local_fallback(img)

def extract_ring_local_fallback(img):
    """Local fallback method for background removal"""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    width, height = img.size
    img_array = np.array(img)
    
    # Sample corners for background color
    corner_size = 10
    corners = []
    corners.extend(img_array[:corner_size, :corner_size].reshape(-1, 4))
    corners.extend(img_array[:corner_size, -corner_size:].reshape(-1, 4))
    corners.extend(img_array[-corner_size:, :corner_size].reshape(-1, 4))
    corners.extend(img_array[-corner_size:, -corner_size:].reshape(-1, 4))
    
    corners_array = np.array(corners)
    bg_color = np.median(corners_array, axis=0)[:3]
    
    # Calculate color distance
    color_distance = np.sqrt(
        (img_array[:,:,0] - bg_color[0])**2 +
        (img_array[:,:,1] - bg_color[1])**2 +
        (img_array[:,:,2] - bg_color[2])**2
    )
    
    threshold = np.percentile(color_distance, 30)
    mask = color_distance > threshold
    
    # Clean up mask
    mask = mask.astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, 'L')
    
    # Morphological operations
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    mask_img = mask_img.filter(ImageFilter.SMOOTH_MORE)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Apply mask
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
    
    # Apply multipliers
    r = r.point(lambda x: min(255, int(x * color_multipliers[0])))
    g = g.point(lambda x: min(255, int(x * color_multipliers[1])))
    b = b.point(lambda x: min(255, int(x * color_multipliers[2])))
    
    return Image.merge('RGBA', (r, g, b, a))

def create_color_options_section(width=FIXED_WIDTH, ring_image=None):
    """Create COLOR section for group 6 (image 9) - Figma design style with perfect background removal"""
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
    
    # Color information - exactly like Figma
    colors = [
        ("Yellow Gold", "#FFD700", (1.15, 1.10, 0.85)),
        ("Rose Gold", "#F4C2C2", (1.15, 0.95, 0.90)),
        ("White Gold", "#E5E4E2", (0.95, 0.95, 1.05)),
        ("White", "#FFFFFF", (1.0, 1.0, 1.0))
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
            # Extract the ring with Replicate API
            ring_extracted = extract_ring_with_replicate(ring_image)
            
            for i, (name, color_hex, color_mult) in enumerate(colors):
                row = i // 2
                col = i % 2
                
                x = start_x + col * (img_size + h_spacing)
                y = start_y + row * v_spacing
                
                # Create a white background for each ring
                ring_bg = Image.new('RGBA', (img_size, img_size), (255, 255, 255, 255))
                
                # Resize ring maintaining aspect ratio
                ring_aspect = ring_extracted.width / ring_extracted.height
                if ring_aspect > 1:
                    new_width = int(img_size * 0.75)  # Slightly smaller for padding
                    new_height = int(new_width / ring_aspect)
                else:
                    new_height = int(img_size * 0.75)
                    new_width = int(new_height * ring_aspect)
                
                resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
                ring_resized = ring_extracted.resize((new_width, new_height), resample_filter)
                
                # Apply color tint
                ring_tinted = apply_metal_color_filter(ring_resized, color_mult)
                
                # Add subtle drop shadow
                shadow_offset = 5
                shadow_img = Image.new('RGBA', (img_size, img_size), (255, 255, 255, 0))
                shadow_draw = ImageDraw.Draw(shadow_img)
                
                # Create soft shadow
                shadow_x = (img_size - new_width) // 2 + shadow_offset
                shadow_y = (img_size - new_height) // 2 + shadow_offset
                
                # Multiple layers for softer shadow
                for j in range(3):
                    opacity = 20 - j * 5
                    shadow_draw.ellipse([
                        shadow_x - j*2, 
                        shadow_y - j*2, 
                        shadow_x + new_width + j*2, 
                        shadow_y + new_height + j*2
                    ], fill=(180, 180, 180, opacity))
                
                # Combine shadow with background
                combined = Image.alpha_composite(ring_bg, shadow_img)
                
                # Center the ring on background
                paste_x = (img_size - new_width) // 2
                paste_y = (img_size - new_height) // 2
                
                # Create a temporary image for the ring
                temp_img = Image.new('RGBA', (img_size, img_size), (255, 255, 255, 0))
                temp_img.paste(ring_tinted, (paste_x, paste_y), ring_tinted)
                
                # Composite ring onto shadow background
                final_ring = Image.alpha_composite(combined, temp_img)
                
                # Add subtle border
                border_img = Image.new('RGBA', (img_size, img_size), (255, 255, 255, 0))
                border_draw = ImageDraw.Draw(border_img)
                border_draw.rectangle([0, 0, img_size-1, img_size-1], outline=(230, 230, 230, 255), width=1)
                final_ring = Image.alpha_composite(final_ring, border_img)
                
                # Paste to main image
                section_img.paste(final_ring, (x, y), final_ring)
                
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
    
    # IMPORTANT: Group 5 should only have 2 images (7, 8)
    if group_number == 5 and len(images_data) != 2:
        print(f"WARNING: Group 5 should have exactly 2 images, got {len(images_data)}")
    
    # Calculate heights
    TOP_MARGIN = 100
    BOTTOM_MARGIN = 100
    IMAGE_SPACING = 200  # 200픽셀 간격으로 증가
    
    # Add section heights based on group
    section_height = 0
    if group_number == 3:
        section_height = 400  # MD'Talk section
    elif group_number == 4:
        section_height = 500  # DESIGN POINT section
    
    # Calculate total height
    total_height = TOP_MARGIN + BOTTOM_MARGIN
    
    # First add section height if needed
    if section_height > 0:
        total_height += section_height + 100  # Extra spacing after section
    
    # Then add all image heights
    image_heights = []
    for img_data in images_data:
        img = get_image_from_input(img_data)
        img_height = calculate_image_height(img.width, img.height, FIXED_WIDTH)
        image_heights.append(img_height)
        total_height += img_height
        img.close()
    
    # Add spacing between images
    total_height += (len(images_data) - 1) * IMAGE_SPACING
    
    print(f"Creating combined canvas: {FIXED_WIDTH}x{total_height}")
    
    # Create canvas
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    
    current_y = TOP_MARGIN
    
    # Add section FIRST (위에 오도록)
    if group_number == 3:
        print("Adding MD'Talk section at top")
        md_talk_section = create_md_talk_section()
        detail_page.paste(md_talk_section, (0, current_y))
        current_y += 400 + 100  # section height + spacing
    elif group_number == 4:
        print("Adding DESIGN POINT section at top")
        design_point_section = create_design_point_section()
        detail_page.paste(design_point_section, (0, current_y))
        current_y += 500 + 100  # section height + spacing
    
    # Process each image AFTER section
    for idx, (img_data, img_height) in enumerate(zip(images_data, image_heights)):
        if idx > 0:
            current_y += IMAGE_SPACING  # 200픽셀 간격
        
        # Get image
        img = get_image_from_input(img_data)
        
        # Resize image
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        img_resized = img.resize((FIXED_WIDTH, img_height), resample_filter)
        
        # Paste image
        detail_page.paste(img_resized, (0, current_y))
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
    print("Processing group 6 - COLOR section with image 9 ONLY")
    
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
        print(f"=== V106 Detail Page Handler - Final Perfect Fix ===")
        
        # Find input data
        input_data = event.get('input', event)
        print(f"Input keys: {list(input_data.keys())}")
        print(f"Full input data: {json.dumps(input_data, indent=2)}")
        
        # Get route/group number - CRITICAL for Make.com
        route_number = input_data.get('route_number', 0)
        group_number = input_data.get('group_number', route_number)
        
        # Make.com sends data with specific keys, check for them
        if group_number == 0:
            # Try to detect from Make.com data structure
            if 'image1' in input_data:
                group_number = 1
            elif 'image2' in input_data:
                group_number = 2
            elif 'image3' in input_data:
                group_number = 3
            elif 'image4' in input_data:
                group_number = 4
            elif 'image5' in input_data:
                group_number = 5
            elif 'image6' in input_data:
                group_number = 6
        
        print(f"Final group_number: {group_number}")
        
        # Handle different input formats from Make.com
        if 'combined_urls' in input_data and input_data['combined_urls']:
            # URLs are semicolon-separated
            urls = input_data['combined_urls'].split(';')
            input_data['images'] = []
            for url in urls:
                url = url.strip()
                if url:
                    input_data['images'].append({'url': url})
            print(f"Converted combined_urls to {len(input_data['images'])} images")
        
        # Also check for Make.com specific image keys
        elif f'image{group_number}' in input_data:
            # Single URL from Make.com
            image_url = input_data[f'image{group_number}']
            if ';' in image_url:
                # Multiple URLs
                urls = image_url.split(';')
                input_data['images'] = [{'url': url.strip()} for url in urls if url.strip()]
            else:
                # Single URL
                input_data['url'] = image_url
        
        # Handle different group types based on group_number
        if group_number == 6:
            # Group 6: COLOR section with image 9 ONLY
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif group_number in [3, 4, 5]:
            # Groups 3, 4, 5: Combined images with sections
            if 'images' not in input_data or not isinstance(input_data['images'], list):
                # Convert single image to images array
                input_data['images'] = [input_data]
            
            # CRITICAL: Group 5 should only have 2 images (7, 8), NOT 3!
            if group_number == 5 and len(input_data['images']) > 2:
                print(f"ERROR: Group 5 has {len(input_data['images'])} images, should have 2. Using first 2 only.")
                input_data['images'] = input_data['images'][:2]  # Use only first 2 images
            
            detail_page = process_combined_images(input_data['images'], group_number)
            
            if group_number == 3:
                page_type = "combined_3_4_mdtalk"
            elif group_number == 4:
                page_type = "combined_5_6_design"
            elif group_number == 5:
                page_type = "combined_7_8_gallery"  # Only 7-8
            
        else:
            # Groups 1, 2: Single images
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
            "has_background_removal": group_number == 6,
            "uses_replicate_api": group_number == 6,
            "format": "base64_no_padding",
            "status": "success",
            "version": "V106",
            "image_spacing": 200,
            "group_info": {
                "1": "Single image 1",
                "2": "Single image 2", 
                "3": "Images 3-4 with MD'Talk at top",
                "4": "Images 5-6 with DESIGN POINT at top",
                "5": "Images 7-8 ONLY (no 9)",  # CRITICAL
                "6": "Image 9 ONLY for COLOR section"
            }.get(str(group_number), f"Group {group_number}")
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
                "version": "V106"
            }
        }

# RunPod handler registration
runpod.serverless.start({"handler": handler})
