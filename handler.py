import runpod
import os
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import re
import numpy as np

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
        font_path = '/tmp/NanumGothic.ttf'
        
        if os.path.exists(font_path):
            try:
                test_font = ImageFont.truetype(font_path, 20)
                print("Korean font already exists and is valid")
                return True
            except:
                print("Korean font exists but is corrupted, re-downloading...")
                os.remove(font_path)
        
        font_urls = [
            'https://github.com/naver/nanumfont/raw/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
            'https://cdn.jsdelivr.net/gh/naver/nanumfont@master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf',
            'https://raw.githubusercontent.com/naver/nanumfont/master/fonts/NanumFontSetup_TTF_GOTHIC/NanumGothic.ttf'
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

def safe_draw_text(draw, position, text, font, fill):
    """Safely draw text handling encoding issues with better Korean support"""
    try:
        # Ensure text is string and properly encoded
        if text:
            text = str(text)
            # Force UTF-8 encoding for Korean text
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
            draw.text(position, text, font=font, fill=fill)
    except Exception as e:
        print(f"Error drawing text '{text}': {str(e)}")
        # Try with basic ASCII only as last resort
        try:
            safe_text = ''.join(c for c in text if ord(c) < 128 or 0xAC00 <= ord(c) <= 0xD7A3)
            draw.text(position, safe_text or "[Text Error]", font=font, fill=fill)
        except:
            draw.text(position, "[Error]", font=font, fill=fill)

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

def remove_background_from_image(image):
    """Remove background from image using available methods"""
    try:
        # Method 1: Try local rembg first (fastest)
        if REMBG_AVAILABLE:
            try:
                print("Removing background using local rembg...")
                
                # Initialize session if not already done
                if not hasattr(remove_background_from_image, 'session'):
                    remove_background_from_image.session = new_session('u2netp')
                
                # Convert to bytes
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                
                # Remove background
                output = remove(
                    buffered.getvalue(),
                    session=remove_background_from_image.session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=200,
                    alpha_matting_background_threshold=80,
                    alpha_matting_erode_size=2
                )
                
                result_image = Image.open(BytesIO(output))
                print("Background removed successfully with local rembg")
                return result_image
                
            except Exception as e:
                print(f"Local rembg failed: {e}")
        
        # Method 2: Try Replicate API
        if REPLICATE_AVAILABLE and REPLICATE_CLIENT:
            try:
                print("Removing background using Replicate API...")
                
                # Convert to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                img_data_url = f"data:image/png;base64,{img_base64}"
                
                # Use rembg model
                output = REPLICATE_CLIENT.run(
                    "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
                    input={
                        "image": img_data_url,
                        "model": "u2netp",
                        "alpha_matting": True,
                        "alpha_matting_foreground_threshold": 200,
                        "alpha_matting_background_threshold": 80,
                        "alpha_matting_erode_size": 2
                    }
                )
                
                if output:
                    if isinstance(output, str):
                        response = requests.get(output)
                        result_image = Image.open(BytesIO(response.content))
                    else:
                        result_image = Image.open(BytesIO(base64.b64decode(output)))
                    
                    print("Background removed successfully with Replicate")
                    return result_image
                    
            except Exception as e:
                print(f"Replicate background removal failed: {e}")
        
        # Method 3: Basic manual background removal (fallback)
        print("Using basic background removal (fallback)")
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Create a simple white background mask
        # This is very basic and won't work well for complex images
        data = np.array(image)
        # Assume white/light gray background
        mask = (data[:,:,0] > 240) & (data[:,:,1] > 240) & (data[:,:,2] > 240)
        data[mask] = [255, 255, 255, 0]
        
        return Image.fromarray(data, 'RGBA')
        
    except Exception as e:
        print(f"All background removal methods failed: {e}")
        # Return original image if all methods fail
        return image

def create_ai_generated_md_talk(claude_text, width=FIXED_WIDTH):
    """Create AI-generated MD Talk section with safe text rendering"""
    section_height = 600
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Ensure Korean font is downloaded
    download_korean_font()
    
    # Get fonts with better fallback
    title_font = None
    body_font = None
    
    # Try Korean font first
    if os.path.exists("/tmp/NanumGothic.ttf"):
        try:
            title_font = ImageFont.truetype("/tmp/NanumGothic.ttf", 48)
            body_font = ImageFont.truetype("/tmp/NanumGothic.ttf", 28)
        except Exception as e:
            print(f"Failed to load Korean font: {e}")
    
    # Fallback fonts
    if not title_font:
        font_paths = ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    title_font = ImageFont.truetype(font_path, 48)
                    body_font = ImageFont.truetype(font_path, 28)
                    break
                except:
                    continue
    
    if not title_font:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Title
    title = "MD TALK"
    try:
        title_width, _ = get_text_dimensions(draw, title, title_font)
        safe_draw_text(draw, (width//2 - title_width//2, 60), title, title_font, (40, 40, 40))
    except:
        safe_draw_text(draw, (width//2 - 100, 60), title, title_font, (40, 40, 40))
    
    # Process Claude text
    if claude_text:
        # Remove any MD TALK prefix if exists
        text = claude_text.replace('MD TALK', '').replace('MD Talk', '').strip()
        
        # Split text into lines
        words = text.split()
        lines = []
        current_line = ""
        max_line_width = width - 120  # 60px margin on each side
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            try:
                test_width, _ = get_text_dimensions(draw, test_line, body_font)
                if test_width > max_line_width:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            except:
                # If text measurement fails, use character count
                if len(test_line) > 30:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        # Limit to reasonable number of lines
        lines = lines[:8]
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
            try:
                line_width, _ = get_text_dimensions(draw, line, body_font)
                safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
            except:
                safe_draw_text(draw, (60, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    return section_img

def create_ai_generated_design_point(claude_text, width=FIXED_WIDTH):
    """Create AI-generated Design Point section with safe text rendering"""
    section_height = 700
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Ensure Korean font is downloaded
    download_korean_font()
    
    # Get fonts with better fallback
    title_font = None
    body_font = None
    
    # Try Korean font first
    if os.path.exists("/tmp/NanumGothic.ttf"):
        try:
            title_font = ImageFont.truetype("/tmp/NanumGothic.ttf", 48)
            body_font = ImageFont.truetype("/tmp/NanumGothic.ttf", 24)
        except Exception as e:
            print(f"Failed to load Korean font: {e}")
    
    # Fallback fonts
    if not title_font:
        font_paths = ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    title_font = ImageFont.truetype(font_path, 48)
                    body_font = ImageFont.truetype(font_path, 24)
                    break
                except:
                    continue
    
    if not title_font:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Title
    title = "DESIGN POINT"
    try:
        title_width, _ = get_text_dimensions(draw, title, title_font)
        safe_draw_text(draw, (width//2 - title_width//2, 60), title, title_font, (40, 40, 40))
    except:
        safe_draw_text(draw, (width//2 - 150, 60), title, title_font, (40, 40, 40))
    
    # Process Claude text
    if claude_text:
        # Remove any DESIGN POINT prefix if exists
        text = claude_text.replace('DESIGN POINT', '').replace('Design Point', '').strip()
        
        # Split text into lines with better handling
        words = text.split()
        lines = []
        current_line = ""
        max_line_width = width - 100  # 50px margin on each side
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            try:
                test_width, _ = get_text_dimensions(draw, test_line, body_font)
                if test_width > max_line_width:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            except:
                if len(test_line) > 35:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
        
        if current_line:
            lines.append(current_line)
        
        # Limit to reasonable number of lines
        lines = lines[:10]
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
            try:
                line_width, _ = get_text_dimensions(draw, line, body_font)
                safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
            except:
                safe_draw_text(draw, (50, y_pos), line, body_font, (80, 80, 80))
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
        
        # Check for image9 specifically (for color section)
        if 'image9' in input_data and input_data['image9']:
            image_data = input_data['image9']
            if image_data.startswith('http'):
                return download_image_from_google_drive(image_data)
            else:
                return Image.open(BytesIO(base64.b64decode(image_data)))
        
        raise ValueError("No valid image data found in input")
        
    except Exception as e:
        print(f"Error getting image: {str(e)}")
        raise

def detect_group_number_from_input(input_data):
    """Detect which group this is from input data - ENHANCED LOGIC"""
    # PRIORITY 1: Check for explicit route_number (highest priority)
    route_number = input_data.get('route_number', 0)
    if route_number and str(route_number).isdigit():
        group_num = int(route_number)
        print(f"Found explicit route_number: {group_num}")
        return group_num
    
    # PRIORITY 2: Check for group_number
    group_number = input_data.get('group_number', 0)
    if group_number and str(group_number).isdigit():
        group_num = int(group_number)
        print(f"Found explicit group_number: {group_num}")
        return group_num
    
    # PRIORITY 3: Check for text type hints (for groups 7, 8)
    text_type = input_data.get('text_type', '').lower()
    if 'md_talk' in text_type or 'md talk' in text_type:
        print("Detected MD Talk text type -> Group 7")
        return 7
    elif 'design_point' in text_type or 'design point' in text_type:
        print("Detected Design Point text type -> Group 8")
        return 8
    
    # PRIORITY 4: Check for specific image keys
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
    
    # PRIORITY 5: Check for color section indicators
    if any(key in str(input_data).lower() for key in ['color', 'colour', 'image9']):
        print("Detected color section indicators -> Group 6")
        return 6
    
    # PRIORITY 6: Check images array count
    if 'images' in input_data and isinstance(input_data['images'], list):
        images_count = len(input_data['images'])
        print(f"Found images array with {images_count} images")
        
        if images_count == 1:
            # Single image could be 1, 2, or 6
            # Check for any hints in the data
            data_str = str(input_data).lower()
            if 'color' in data_str or 'image9' in data_str:
                return 6
            elif 'image2' in data_str or 'sub' in data_str:
                return 2
            return 1  # Default to group 1 for single image
            
        elif images_count == 2:
            # 2 images could be 3, 4, or 5
            data_str = str(input_data).lower()
            if 'image7' in data_str or 'image8' in data_str:
                return 5
            elif 'image5' in data_str or 'image6' in data_str:
                return 4
            elif 'image3' in data_str or 'image4' in data_str:
                return 3
            return 3  # Default to group 3 for 2 images
            
        elif images_count >= 3:
            # 3 or more images is likely group 5, but should be limited to 2
            return 5
    
    # DEFAULT: If no clear indicators, default to group 1
    print("No clear group indicators found, defaulting to group 1")
    return 1

def create_color_options_section(ring_image=None):
    """Create COLOR section with 4 color options and proper background removal"""
    width = FIXED_WIDTH
    height = 800
    
    section_img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Title with better font handling
    title_font = None
    label_font = None
    
    # Try Korean font first
    if download_korean_font():
        try:
            title_font = ImageFont.truetype('/tmp/NanumGothic.ttf', 56)
            label_font = ImageFont.truetype('/tmp/NanumGothic.ttf', 24)
        except:
            pass
    
    # Fallback fonts
    if not title_font:
        font_paths = ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    title_font = ImageFont.truetype(font_path, 56)
                    label_font = ImageFont.truetype(font_path, 24)
                    break
                except:
                    continue
    
    if not title_font:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw title
    title = "COLOR"
    try:
        title_width, _ = get_text_dimensions(draw, title, title_font)
        safe_draw_text(draw, (width//2 - title_width//2, 60), title, title_font, (40, 40, 40))
    except:
        safe_draw_text(draw, (width//2 - 100, 60), title, title_font, (40, 40, 40))
    
    # Remove background from ring image if provided
    ring_no_bg = None
    if ring_image:
        try:
            print("Removing background from ring image for color section")
            ring_no_bg = remove_background_from_image(ring_image)
            
            # Ensure it's RGBA
            if ring_no_bg.mode != 'RGBA':
                ring_no_bg = ring_no_bg.convert('RGBA')
            
            print(f"Background removed successfully, image mode: {ring_no_bg.mode}")
                
        except Exception as e:
            print(f"Failed to remove background: {e}")
            ring_no_bg = ring_image.convert('RGBA') if ring_image.mode != 'RGBA' else ring_image
    
    # Color options with proper colors
    colors = [
        ("yellow", "옐로우골드", (255, 215, 0)),      # Gold color
        ("rose", "로즈골드", (183, 110, 121)),        # Rose gold color  
        ("white", "화이트골드", (250, 250, 250)),     # White gold (slight gray)
        ("antique", "무도금화이트", (255, 255, 255))  # Pure white for unplated
    ]
    
    # Grid layout - 2x2
    grid_size = 280  # Larger size for better visibility
    padding = 60
    start_x = (width - (grid_size * 2 + padding)) // 2
    start_y = 180
    
    for i, (color_id, label, color_rgb) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (grid_size + padding)
        y = start_y + row * (grid_size + 100)
        
        # Create white background container
        container = Image.new('RGBA', (grid_size, grid_size), (255, 255, 255, 255))
        container_draw = ImageDraw.Draw(container)
        
        # Draw border
        container_draw.rectangle([0, 0, grid_size-1, grid_size-1], 
                                fill=None, outline=(230, 230, 230), width=2)
        
        # If ring image provided, apply color tinting
        if ring_no_bg:
            try:
                # Create a copy and resize
                ring_copy = ring_no_bg.copy()
                ring_copy.thumbnail((int(grid_size * 0.8), int(grid_size * 0.8)), 
                                  Image.Resampling.LANCZOS)
                
                # Apply color tint based on metal type
                if color_id == "yellow":
                    # Yellow gold - warm golden tint
                    ring_tinted = apply_metal_color_effect(ring_copy, (255, 215, 0), strength=0.25)
                elif color_id == "rose":
                    # Rose gold - pinkish tint
                    ring_tinted = apply_metal_color_effect(ring_copy, (183, 110, 121), strength=0.25)
                elif color_id == "white":
                    # White gold - very slight cool tint
                    ring_tinted = apply_metal_color_effect(ring_copy, (240, 240, 255), strength=0.05)
                else:
                    # Antique/unplated - pure white, no tint
                    ring_tinted = ring_copy
                
                # Center the ring in container
                paste_x = (grid_size - ring_tinted.width) // 2
                paste_y = (grid_size - ring_tinted.height) // 2
                
                # Paste ring with transparency
                container.paste(ring_tinted, (paste_x, paste_y), ring_tinted)
                
            except Exception as e:
                print(f"Error applying color {color_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Paste container onto main image
        section_img.paste(container, (x, y))
        
        # Draw label with Korean text
        try:
            label_width, _ = get_text_dimensions(draw, label, label_font)
            safe_draw_text(draw, (x + grid_size//2 - label_width//2, y + grid_size + 20), 
                         label, label_font, (80, 80, 80))
        except:
            safe_draw_text(draw, (x + 20, y + grid_size + 20), 
                         label, label_font, (80, 80, 80))
    
    return section_img

def apply_metal_color_effect(image, metal_color, strength=0.2):
    """Apply realistic metal color effect to jewelry image"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Split channels
    r, g, b, a = image.split()
    
    # Convert to arrays for processing
    r_array = np.array(r, dtype=np.float32)
    g_array = np.array(g, dtype=np.float32)
    b_array = np.array(b, dtype=np.float32)
    
    # Only apply color to non-transparent pixels
    mask = np.array(a) > 0
    
    # Calculate luminance
    luminance = (0.299 * r_array + 0.587 * g_array + 0.114 * b_array) / 255.0
    
    # Apply metal color based on luminance (preserves highlights and shadows)
    if mask.any():
        # Normalize metal color
        metal_r, metal_g, metal_b = metal_color
        metal_r, metal_g, metal_b = metal_r/255.0, metal_g/255.0, metal_b/255.0
        
        # Apply color with luminance preservation
        r_array[mask] = r_array[mask] * (1 - strength) + (metal_r * luminance[mask] * 255) * strength
        g_array[mask] = g_array[mask] * (1 - strength) + (metal_g * luminance[mask] * 255) * strength
        b_array[mask] = b_array[mask] * (1 - strength) + (metal_b * luminance[mask] * 255) * strength
    
    # Ensure values are in valid range
    r_array = np.clip(r_array, 0, 255)
    g_array = np.clip(g_array, 0, 255)
    b_array = np.clip(b_array, 0, 255)
    
    # Convert back to PIL Images
    r_new = Image.fromarray(r_array.astype(np.uint8))
    g_new = Image.fromarray(g_array.astype(np.uint8))
    b_new = Image.fromarray(b_array.astype(np.uint8))
    
    # Merge channels
    return Image.merge('RGBA', (r_new, g_new, b_new, a))

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
    for font_path in ["/tmp/NanumGothic.ttf", 
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if os.path.exists(font_path):
            try:
                small_font = ImageFont.truetype(font_path, 16)
                break
            except:
                continue
    
    if small_font is None:
        small_font = ImageFont.load_default()
    
    try:
        text_width, _ = get_text_dimensions(draw, page_text, small_font)
        safe_draw_text(draw, (FIXED_WIDTH//2 - text_width//2, total_height - 30), 
                     page_text, small_font, (200, 200, 200))
    except:
        safe_draw_text(draw, (FIXED_WIDTH//2 - 50, total_height - 30), 
                     page_text, small_font, (200, 200, 200))
    
    return detail_page

def process_combined_images(input_data, group_number):
    """Process combined images WITHOUT text sections (groups 3, 4, 5) - FIXED WIDTH 1200"""
    print(f"Processing combined images for group {group_number} with FIXED WIDTH 1200")
    
    # Get images data - ENHANCED parsing logic
    images_data = []
    
    # Priority 1: Check for specific image keys based on group
    if group_number == 3:
        # Look for image3 and image4
        if 'image3' in input_data and 'image4' in input_data:
            images_data = [
                {'image': input_data['image3']},
                {'image': input_data['image4']}
            ]
        elif 'images' in input_data and isinstance(input_data['images'], list) and len(input_data['images']) >= 2:
            images_data = input_data['images'][:2]
            
    elif group_number == 4:
        # Look for image5 and image6
        if 'image5' in input_data and 'image6' in input_data:
            images_data = [
                {'image': input_data['image5']},
                {'image': input_data['image6']}
            ]
        elif 'images' in input_data and isinstance(input_data['images'], list) and len(input_data['images']) >= 2:
            images_data = input_data['images'][:2]
            
    elif group_number == 5:
        # Look for image7 and image8
        if 'image7' in input_data and 'image8' in input_data:
            images_data = [
                {'image': input_data['image7']},
                {'image': input_data['image8']}
            ]
        elif 'images' in input_data and isinstance(input_data['images'], list) and len(input_data['images']) >= 2:
            images_data = input_data['images'][:2]
    
    # Priority 2: If still no images, check 'images' array
    if not images_data and 'images' in input_data and isinstance(input_data['images'], list):
        images_data = input_data['images']
    
    # CRITICAL: Ensure exactly 2 images for groups 3, 4, 5
    if len(images_data) > 2:
        print(f"WARNING: Group {group_number} has {len(images_data)} images, using first 2 only")
        images_data = images_data[:2]
    elif len(images_data) < 2:
        raise ValueError(f"Group {group_number} requires exactly 2 images, but only {len(images_data)} found")
    
    print(f"Processing exactly {len(images_data)} images for group {group_number}")
    
    # Calculate dimensions for each image
    target_width = FIXED_WIDTH  # 1200
    image_height = int(target_width * 1.3)  # 1200 x 1560 (same ratio as original)
    
    # Calculate total height with 200px spacing between images
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    IMAGE_SPACING = 200  # 200px between images as requested
    
    # Total height = top margin + image1 + spacing + image2 + bottom margin
    total_height = TOP_MARGIN + (2 * image_height) + IMAGE_SPACING + BOTTOM_MARGIN
    
    print(f"Creating combined page: {FIXED_WIDTH}x{total_height} (2 images with 200px spacing)")
    
    # Create combined page with white background
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    
    current_y = TOP_MARGIN
    
    # Process exactly 2 images
    for idx, img_data in enumerate(images_data[:2]):  # Ensure max 2 images
        if idx == 1:  # Add spacing before second image
            current_y += IMAGE_SPACING
        
        try:
            # Get image from the data
            if isinstance(img_data, dict):
                # Try different keys to find the image
                img = None
                for key in ['image', 'url', 'enhanced_image', f'image{group_number*2+idx-1}']:
                    if key in img_data and img_data[key]:
                        temp_data = {'image': img_data[key]}
                        img = get_image_from_input(temp_data)
                        break
                
                if not img:
                    img = get_image_from_input(img_data)
            else:
                # img_data might be a string (base64 or URL)
                temp_data = {'image': img_data}
                img = get_image_from_input(temp_data)
            
            # Resize image to exact dimensions
            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            img_resized = img.resize((target_width, image_height), resample_filter)
            
            # Paste on page centered horizontally
            x_position = (FIXED_WIDTH - target_width) // 2  # Should be 0 since widths match
            detail_page.paste(img_resized, (x_position, current_y))
            current_y += image_height
            
            img.close()
            print(f"Successfully added image {idx + 1} at position y={current_y - image_height}")
            
        except Exception as e:
            print(f"Error processing image {idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Add page indicator at bottom
    draw = ImageDraw.Draw(detail_page)
    page_texts = {
        3: "- Images 3-4 -",
        4: "- Images 5-6 -", 
        5: "- Images 7-8 -"
    }
    page_text = page_texts.get(group_number, f"- Details {group_number} -")
    
    small_font = None
    for font_path in ["/tmp/NanumGothic.ttf", 
                     "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
        if os.path.exists(font_path):
            try:
                small_font = ImageFont.truetype(font_path, 16)
                break
            except:
                continue
    
    if small_font is None:
        small_font = ImageFont.load_default()
    
    try:
        text_width, _ = get_text_dimensions(draw, page_text, small_font)
        safe_draw_text(draw, (FIXED_WIDTH//2 - text_width//2, total_height - 30), 
                     page_text, small_font, (200, 200, 200))
    except:
        safe_draw_text(draw, (FIXED_WIDTH//2 - 50, total_height - 30), 
                     page_text, small_font, (200, 200, 200))
    
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
    
    if group_number == 7 or 'md_talk' in text_type.lower():
        # Group 7: MD Talk text section
        text_section = create_ai_generated_md_talk(claude_text)
        section_type = "md_talk"
    elif group_number == 8 or 'design_point' in text_type.lower():
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
    """Main handler for detail page creation"""
    try:
        print(f"=== V114 Detail Page Handler - Complete Version ===")
        
        # Find input data
        input_data = event.get('input', event)
        print(f"Input keys: {list(input_data.keys())}")
        
        # Detect group number with ENHANCED logic
        group_number = detect_group_number_from_input(input_data)
        print(f"Detected group number: {group_number}")
        
        if group_number < 1 or group_number > 8:
            raise ValueError(f"Invalid group number: {group_number}. Must be 1-8.")
        
        # Process based on group number
        if group_number == 6:
            # Group 6: COLOR section ONLY (using image 9)
            print("=== Processing Group 6: COLOR section with background removal ===")
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
            # Groups 3, 4, 5: Combined images (2 images with 200px gap)
            print(f"=== Processing Group {group_number}: Combined images (2 images, 200px gap) ===")
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
        
        # Prepare metadata with CORRECT route_number
        metadata = {
            "enhanced_image": detail_base64_no_padding,
            "status": "success",
            "page_type": page_type,
            "page_number": group_number,
            "route_number": group_number,  # ENSURE route_number matches group_number
            "dimensions": {
                "width": detail_page.width,
                "height": detail_page.height
            },
            "has_text_overlay": group_number in [7, 8],
            "has_background_removal": group_number == 6,
            "format": "base64_no_padding",
            "version": "V114_COMPLETE"
        }
        
        # Send to webhook with CORRECT route_number
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
                "version": "V114_COMPLETE"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("V114 Detail Handler Started - Complete Version!")
    runpod.serverless.start({"handler": handler})
