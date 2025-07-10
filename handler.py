import runpod
import os
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json
import re
import numpy as np
from scipy import ndimage

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

def download_korean_font():
    """Download Korean font for text rendering with better error handling"""
    try:
        font_path = '/tmp/NanumGothic.ttf'
        
        # Check if font already exists and is valid
        if os.path.exists(font_path):
            try:
                # Test if font loads properly
                test_font = ImageFont.truetype(font_path, 20)
                # Test Korean character rendering
                img_test = Image.new('RGB', (100, 100), 'white')
                draw_test = ImageDraw.Draw(img_test)
                draw_test.text((10, 10), "테스트", font=test_font, fill='black')
                print("Korean font already exists and works properly")
                return font_path
            except Exception as e:
                print(f"Korean font exists but has issues: {e}")
                os.remove(font_path)
        
        # Download font
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
                    
                    # Verify font works
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
    
    # Add Korean font if available
    if korean_font_path and os.path.exists(korean_font_path):
        fonts_to_try.append(korean_font_path)
    
    # Add system fonts
    fonts_to_try.extend([
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS Korean font
        "C:/Windows/Fonts/malgun.ttf",  # Windows Korean font
        "C:/Windows/Fonts/NanumGothic.ttf"  # Windows Nanum font
    ])
    
    for font_path in fonts_to_try:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, size)
                # Test if font can render Korean
                img_test = Image.new('RGB', (100, 100), 'white')
                draw_test = ImageDraw.Draw(img_test)
                draw_test.text((10, 10), "한글", font=font, fill='black')
                return font
            except Exception as e:
                print(f"Failed to load font {font_path}: {e}")
                continue
    
    # Last resort - default font
    print("Using default font as last resort")
    return ImageFont.load_default()

def safe_draw_text(draw, position, text, font, fill):
    """Safely draw text with proper encoding"""
    try:
        # Ensure text is properly encoded
        if text:
            text = str(text)
            # Draw text directly without additional encoding
            draw.text(position, text, font=font, fill=fill)
    except Exception as e:
        print(f"Error drawing text '{text}': {str(e)}")
        # Fallback to safe characters only
        try:
            safe_text = ''.join(c if ord(c) < 128 or 0xAC00 <= ord(c) <= 0xD7A3 else '?' for c in text)
            draw.text(position, safe_text or "[Error]", font=font, fill=fill)
        except:
            draw.text(position, "[Error]", font=font, fill=fill)

def get_text_size(draw, text, font):
    """Get text size compatible with different PIL versions"""
    try:
        # Try newer method first
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fallback to older method
        return draw.textsize(text, font=font)

def clean_claude_text(text):
    """Clean text for safe rendering"""
    if not text:
        return ""
    
    text = str(text) if text is not None else ""
    
    # Remove escape sequences
    text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Remove markdown
    text = re.sub(r'[#*_`]', '', text)
    
    # Clean multiple spaces
    text = ' '.join(text.split())
    
    return text.strip()

def remove_background_from_image(image):
    """Remove background including ring center holes - ULTRA PRECISE"""
    try:
        # Method 1: Try local rembg first with ultra aggressive settings
        if REMBG_AVAILABLE:
            try:
                print("Removing background using local rembg with ULTRA PRECISE settings...")
                
                # Initialize session with best model
                if not hasattr(remove_background_from_image, 'session'):
                    remove_background_from_image.session = new_session('u2netp')
                
                # Convert to bytes
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                
                # Remove background with ultra aggressive settings
                output = remove(
                    buffered.getvalue(),
                    session=remove_background_from_image.session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,  # Lower for more aggressive
                    alpha_matting_background_threshold=10,   # Very low for aggressive removal
                    alpha_matting_erode_size=2,
                    only_mask=False
                )
                
                result_image = Image.open(BytesIO(output))
                
                # Ultra aggressive post-process
                result_image = ultra_precise_ring_transparency(result_image)
                
                print("Background removed successfully with ULTRA PRECISE method")
                return result_image
                
            except Exception as e:
                print(f"Local rembg failed: {e}")
        
        # Method 2: Try Replicate API with ultra settings
        if REPLICATE_AVAILABLE and REPLICATE_CLIENT:
            try:
                print("Removing background using Replicate API with ULTRA PRECISE settings...")
                
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
                        "alpha_matting_foreground_threshold": 240,
                        "alpha_matting_background_threshold": 10,
                        "alpha_matting_erode_size": 2
                    }
                )
                
                if output:
                    if isinstance(output, str):
                        response = requests.get(output)
                        result_image = Image.open(BytesIO(response.content))
                    else:
                        result_image = Image.open(BytesIO(base64.b64decode(output)))
                    
                    # Ultra precise post-process
                    result_image = ultra_precise_ring_transparency(result_image)
                    
                    print("Background removed successfully with Replicate ULTRA PRECISE")
                    return result_image
                    
            except Exception as e:
                print(f"Replicate background removal failed: {e}")
        
        # Method 3: Ultra precise manual background removal
        print("Using ULTRA PRECISE manual background removal")
        result = ultra_precise_manual_remove_background(image)
        return ultra_precise_ring_transparency(result)
        
    except Exception as e:
        print(f"All background removal methods failed: {e}")
        return image

def ultra_precise_ring_transparency(image):
    """ULTRA PRECISE post-process for ring center hole detection"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    data = np.array(image)
    alpha_channel = data[:,:,3]
    
    # Find the center of the image
    height, width = data.shape[:2]
    center_y, center_x = height // 2, width // 2
    
    # Convert to grayscale for analysis
    gray = np.mean(data[:,:,:3], axis=2)
    
    # Method 1: Ultra aggressive bright area detection
    bright_thresholds = [250, 245, 240, 235, 230, 225]  # Multiple thresholds
    
    for threshold in bright_thresholds:
        bright_mask = gray > threshold
        opaque_bright = bright_mask & (alpha_channel > 100)  # Lower alpha threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(opaque_bright)
        
        # Analyze each component
        for i in range(1, num_features + 1):
            region = labeled == i
            region_coords = np.where(region)
            
            if len(region_coords[0]) > 3:  # Even smaller regions
                # Calculate region properties
                region_center_y = np.mean(region_coords[0])
                region_center_x = np.mean(region_coords[1])
                region_size = len(region_coords[0])
                
                # Check if region is roughly in the center
                dist_from_center = np.sqrt((region_center_y - center_y)**2 + (region_center_x - center_x)**2)
                
                # Ultra aggressive criteria
                is_centered = dist_from_center < min(height, width) * 0.5  # Larger area
                is_reasonable_size = region_size < (height * width * 0.3)  # Bigger threshold
                
                # Check if surrounded by non-transparent pixels
                dilated = ndimage.binary_dilation(region, iterations=10)  # More iterations
                touches_edge = (dilated[0,:].any() or dilated[-1,:].any() or 
                               dilated[:,0].any() or dilated[:,-1].any())
                
                if is_centered and is_reasonable_size and not touches_edge:
                    data[region] = [255, 255, 255, 0]
                    print(f"Removed bright region at threshold {threshold} (size: {region_size})")
    
    # Method 2: Edge detection for ring holes
    # Detect edges using gradient
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Find strong edges
    edge_mask = grad_magnitude > 10
    
    # Find enclosed areas using flood fill from edges
    # This helps identify the ring hole boundaries
    filled_from_edges = ndimage.binary_fill_holes(edge_mask)
    potential_holes = filled_from_edges & ~edge_mask
    
    # Check for bright areas within potential holes
    hole_bright = potential_holes & (gray > 220)
    
    if hole_bright.any():
        data[hole_bright] = [255, 255, 255, 0]
        print("Removed hole regions using edge detection")
    
    # Method 3: Color uniformity detection
    # Ring holes often have uniform color
    color_std = np.std(data[:,:,:3], axis=2)
    uniform_mask = color_std < 5  # Very uniform color
    bright_uniform = uniform_mask & (gray > 220) & (alpha_channel > 100)
    
    # Label and check uniform bright regions
    uniform_labeled, num_uniform = ndimage.label(bright_uniform)
    
    for i in range(1, num_uniform + 1):
        uniform_region = uniform_labeled == i
        if uniform_region.sum() > 5:
            # Check if it's in center area
            coords = np.where(uniform_region)
            region_center_y = np.mean(coords[0])
            region_center_x = np.mean(coords[1])
            dist = np.sqrt((region_center_y - center_y)**2 + (region_center_x - center_x)**2)
            
            if dist < min(height, width) * 0.4:
                data[uniform_region] = [255, 255, 255, 0]
                print("Removed uniform color region")
    
    # Method 4: Circular hole detection with multiple radii
    if width > 30 and height > 30:
        Y, X = np.ogrid[:height, :width]
        
        # Try multiple radius percentages
        for radius_percent in [0.15, 0.2, 0.25, 0.3, 0.35]:
            center_area_radius = min(height, width) * radius_percent
            center_mask = (X - center_x)**2 + (Y - center_y)**2 <= center_area_radius**2
            
            # Check average brightness in this circular area
            center_brightness = np.mean(gray[center_mask & (alpha_channel > 100)])
            
            if center_brightness > 225:  # If center is bright
                center_bright = center_mask & (gray > 220) & (alpha_channel > 50)
                if center_bright.any():
                    data[center_bright] = [255, 255, 255, 0]
                    print(f"Removed bright center at radius {radius_percent}")
                    break
    
    return Image.fromarray(data, 'RGBA')

def ultra_precise_manual_remove_background(image):
    """ULTRA PRECISE manual background removal for jewelry"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    data = np.array(image, dtype=np.float32)
    
    # Ultra aggressive multi-threshold approach
    # 1. Pure white
    white_mask = (data[:,:,0] > 252) & (data[:,:,1] > 252) & (data[:,:,2] > 252)
    
    # 2. Near white with multiple thresholds
    near_white_masks = []
    for threshold in [245, 240, 235, 230, 225]:
        mask = (data[:,:,0] > threshold) & (data[:,:,1] > threshold) & (data[:,:,2] > threshold)
        near_white_masks.append(mask)
    
    # 3. Gray detection with tighter tolerance
    max_diff = 10  # Very strict
    color_diff = np.abs(data[:,:,0] - data[:,:,1]) + np.abs(data[:,:,1] - data[:,:,2])
    gray_mask = color_diff < max_diff
    
    # 4. Light colors (any channel bright)
    any_bright = (data[:,:,0] > 240) | (data[:,:,1] > 240) | (data[:,:,2] > 240)
    
    # Combine all masks progressively
    background_mask = white_mask
    
    for near_white in near_white_masks:
        # Add near white that's also gray
        background_mask |= (near_white & gray_mask)
    
    # Add any bright areas that are also uniform in color
    background_mask |= (any_bright & gray_mask)
    
    # Make background transparent
    data[background_mask] = [255, 255, 255, 0]
    
    # Edge cleaning - remove semi-transparent edges
    alpha = data[:,:,3]
    edge_mask = (alpha > 0) & (alpha < 200)  # Semi-transparent pixels
    gray_edges = edge_mask & (gray_mask | (data[:,:,0] > 230))
    data[gray_edges] = [255, 255, 255, 0]
    
    return Image.fromarray(data.astype(np.uint8), 'RGBA')

def auto_crop_transparent(image):
    """Auto-crop transparent borders from image with padding"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Get image data
    data = np.array(image)
    alpha = data[:,:,3]
    
    # Find non-transparent pixels
    non_transparent = np.where(alpha > 10)  # Threshold for transparency
    
    if len(non_transparent[0]) == 0:
        return image  # Return original if all transparent
    
    # Get bounding box
    min_y = non_transparent[0].min()
    max_y = non_transparent[0].max()
    min_x = non_transparent[1].min()
    max_x = non_transparent[1].max()
    
    # Add small padding
    padding = 10  # Increased padding
    min_y = max(0, min_y - padding)
    max_y = min(data.shape[0] - 1, max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(data.shape[1] - 1, max_x + padding)
    
    # Crop
    cropped = image.crop((min_x, min_y, max_x + 1, max_y + 1))
    return cropped

def create_ai_generated_md_talk(claude_text, width=FIXED_WIDTH):
    """Create MD Talk section with dynamic height based on content"""
    # Get Korean font
    korean_font_path = download_korean_font()
    title_font = get_font(48, korean_font_path)
    body_font = get_font(28, korean_font_path)
    
    # Create temporary image for text measurement
    temp_img = Image.new('RGB', (width, 1000), '#FFFFFF')
    draw = ImageDraw.Draw(temp_img)
    
    # Title
    title = "MD TALK"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    # Process Claude text
    if claude_text:
        text = clean_claude_text(claude_text)
        text = text.replace('MD TALK', '').replace('MD Talk', '').strip()
        
        # Word wrap
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
    
    # Calculate dynamic height
    top_margin = 60
    title_bottom_margin = 140  # Gap between title and content
    line_height = 50
    bottom_margin = 80
    
    content_height = len(lines) * line_height
    total_height = top_margin + title_height + title_bottom_margin + content_height + bottom_margin
    
    # Create actual image with calculated height
    section_img = Image.new('RGB', (width, total_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Draw title
    safe_draw_text(draw, (width//2 - title_width//2, top_margin), title, title_font, (40, 40, 40))
    
    # Draw content
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    return section_img

def create_ai_generated_design_point(claude_text, width=FIXED_WIDTH):
    """Create Design Point section with dynamic height based on content"""
    # Get Korean font
    korean_font_path = download_korean_font()
    title_font = get_font(48, korean_font_path)
    body_font = get_font(24, korean_font_path)
    
    # Create temporary image for text measurement
    temp_img = Image.new('RGB', (width, 1000), '#FFFFFF')
    draw = ImageDraw.Draw(temp_img)
    
    # Title
    title = "DESIGN POINT"
    title_width, title_height = get_text_size(draw, title, title_font)
    
    # Process Claude text
    if claude_text:
        text = clean_claude_text(claude_text)
        text = text.replace('DESIGN POINT', '').replace('Design Point', '').strip()
        
        # Word wrap
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
    
    # Calculate dynamic height
    top_margin = 60
    title_bottom_margin = 160  # Gap between title and content
    line_height = 55
    bottom_margin = 100  # Extra space for decorative line
    
    content_height = len(lines) * line_height
    total_height = top_margin + title_height + title_bottom_margin + content_height + bottom_margin
    
    # Create actual image with calculated height
    section_img = Image.new('RGB', (width, total_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Draw title
    safe_draw_text(draw, (width//2 - title_width//2, top_margin), title, title_font, (40, 40, 40))
    
    # Draw content
    y_pos = top_margin + title_height + title_bottom_margin
    
    for line in lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    # Decorative line
    draw.rectangle([100, y_pos + 30, width - 100, y_pos + 32], fill=(220, 220, 220))
    
    return section_img

def create_color_options_section(ring_image=None):
    """Create COLOR section with English labels and enhanced colors"""
    width = FIXED_WIDTH
    height = 850  # Fixed height for color section
    
    section_img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Get fonts
    korean_font_path = download_korean_font()
    title_font = get_font(56, korean_font_path)
    label_font = get_font(24, korean_font_path)
    
    # Draw title
    title = "COLOR"
    title_width, _ = get_text_size(draw, title, title_font)
    safe_draw_text(draw, (width//2 - title_width//2, 60), title, title_font, (40, 40, 40))
    
    # Remove background from ring with ultra precise detection
    ring_no_bg = None
    if ring_image:
        try:
            print("Removing background from ring for color section with ULTRA PRECISION")
            ring_no_bg = remove_background_from_image(ring_image)
            
            if ring_no_bg.mode != 'RGBA':
                ring_no_bg = ring_no_bg.convert('RGBA')
            
            # Additional aggressive crop
            ring_no_bg = auto_crop_transparent(ring_no_bg)
                
        except Exception as e:
            print(f"Failed to remove background: {e}")
            ring_no_bg = ring_image.convert('RGBA') if ring_image else None
    
    # Updated color definitions with English labels and distinct antique color
    colors = [
        ("yellow", "YELLOW", (255, 200, 50), 0.3),           # Golden yellow
        ("rose", "ROSE", (255, 160, 120), 0.35),            # More orange-tinted rose gold
        ("white", "WHITE", (255, 255, 255), 0.0),           # Pure white
        ("antique", "ANTIQUE", (245, 235, 225), 0.1)        # Warm ivory/grayish tone with slight tint
    ]
    
    # Tighter grid layout
    grid_size = 260
    padding = 60
    start_x = (width - (grid_size * 2 + padding)) // 2
    start_y = 160
    
    for i, (color_id, label, color_rgb, strength) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (grid_size + padding)
        y = start_y + row * (grid_size + 100)
        
        # Create container with subtle background
        container = Image.new('RGBA', (grid_size, grid_size), (252, 252, 252, 255))
        container_draw = ImageDraw.Draw(container)
        
        # Draw softer border
        container_draw.rectangle([0, 0, grid_size-1, grid_size-1], 
                                fill=None, outline=(240, 240, 240), width=1)
        
        # Apply ring with color
        if ring_no_bg:
            try:
                ring_copy = ring_no_bg.copy()
                # Smaller ring size for clearer boundaries
                max_size = int(grid_size * 0.7)
                ring_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Apply enhanced metal color effect
                ring_tinted = apply_enhanced_metal_color(ring_copy, color_rgb, strength, color_id)
                
                # Center and paste with padding
                paste_x = (grid_size - ring_tinted.width) // 2
                paste_y = (grid_size - ring_tinted.height) // 2
                container.paste(ring_tinted, (paste_x, paste_y), ring_tinted)
                
            except Exception as e:
                print(f"Error applying color {color_id}: {e}")
        
        # Paste container
        section_img.paste(container, (x, y))
        
        # Draw label in English
        label_width, _ = get_text_size(draw, label, label_font)
        safe_draw_text(draw, (x + grid_size//2 - label_width//2, y + grid_size + 20), 
                     label, label_font, (80, 80, 80))
    
    return section_img

def apply_enhanced_metal_color(image, metal_color, strength=0.3, color_id=""):
    """Apply enhanced metal color effect with special handling for white and rose"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Split channels
    r, g, b, a = image.split()
    
    # Convert to arrays
    r_array = np.array(r, dtype=np.float32)
    g_array = np.array(g, dtype=np.float32)
    b_array = np.array(b, dtype=np.float32)
    a_array = np.array(a)
    
    # Only apply to non-transparent pixels
    mask = a_array > 0
    
    if mask.any():
        # Calculate luminance
        luminance = (0.299 * r_array + 0.587 * g_array + 0.114 * b_array) / 255.0
        
        # Normalize metal color
        metal_r, metal_g, metal_b = [c/255.0 for c in metal_color]
        
        # Special handling for white
        if color_id == "white":
            # Pure bright white
            brightness_boost = 1.15
            r_array[mask] = np.clip(luminance[mask] * 255 * brightness_boost, 240, 255)
            g_array[mask] = np.clip(luminance[mask] * 255 * brightness_boost, 240, 255)
            b_array[mask] = np.clip(luminance[mask] * 255 * brightness_boost, 240, 255)
        
        # Special handling for rose gold - more orange tint
        elif color_id == "rose":
            # Apply stronger orange-pink tint
            highlight_mask = luminance > 0.85
            shadow_mask = luminance < 0.15
            midtone_mask = ~highlight_mask & ~shadow_mask & mask
            
            # Stronger orange tint for midtones
            if midtone_mask.any():
                blend_factor = 0.5  # Stronger blend
                r_array[midtone_mask] = r_array[midtone_mask] * (1 - blend_factor) + (255 * luminance[midtone_mask]) * blend_factor
                g_array[midtone_mask] = g_array[midtone_mask] * (1 - blend_factor) + (160 * luminance[midtone_mask]) * blend_factor
                b_array[midtone_mask] = b_array[midtone_mask] * (1 - blend_factor) + (120 * luminance[midtone_mask]) * blend_factor
            
            # Orange tint for highlights
            if highlight_mask.any():
                r_array[highlight_mask] = np.clip(r_array[highlight_mask] * 0.5 + 255 * 0.5, 0, 255)
                g_array[highlight_mask] = np.clip(g_array[highlight_mask] * 0.5 + 160 * 0.5, 0, 255)
                b_array[highlight_mask] = np.clip(b_array[highlight_mask] * 0.5 + 120 * 0.5, 0, 255)
            
            # Preserve shadows with orange tint
            if shadow_mask.any():
                r_array[shadow_mask] = r_array[shadow_mask] * 0.8 + 50 * 0.2
                g_array[shadow_mask] = g_array[shadow_mask] * 0.8 + 30 * 0.2
                b_array[shadow_mask] = b_array[shadow_mask] * 0.8 + 20 * 0.2
        
        else:
            # Standard metal color application (for yellow gold)
            highlight_mask = luminance > 0.85
            shadow_mask = luminance < 0.15
            midtone_mask = ~highlight_mask & ~shadow_mask & mask
            
            # Apply color more strongly to midtones
            if midtone_mask.any():
                blend_factor = strength * 2.0
                r_array[midtone_mask] = r_array[midtone_mask] * (1 - blend_factor) + (metal_r * 255 * luminance[midtone_mask]) * blend_factor
                g_array[midtone_mask] = g_array[midtone_mask] * (1 - blend_factor) + (metal_g * 255 * luminance[midtone_mask]) * blend_factor
                b_array[midtone_mask] = b_array[midtone_mask] * (1 - blend_factor) + (metal_b * 255 * luminance[midtone_mask]) * blend_factor
            
            # Lighter tint for highlights
            if highlight_mask.any():
                tint_factor = strength * 0.5
                r_array[highlight_mask] = r_array[highlight_mask] * (1 - tint_factor) + (metal_r * 255) * tint_factor
                g_array[highlight_mask] = g_array[highlight_mask] * (1 - tint_factor) + (metal_g * 255) * tint_factor
                b_array[highlight_mask] = b_array[highlight_mask] * (1 - tint_factor) + (metal_b * 255) * tint_factor
            
            # Preserve shadows with light tint
            if shadow_mask.any():
                shadow_tint = strength * 0.2
                r_array[shadow_mask] = r_array[shadow_mask] * (1 - shadow_tint) + (metal_r * r_array[shadow_mask]) * shadow_tint
                g_array[shadow_mask] = g_array[shadow_mask] * (1 - shadow_tint) + (metal_g * g_array[shadow_mask]) * shadow_tint
                b_array[shadow_mask] = b_array[shadow_mask] * (1 - shadow_tint) + (metal_b * b_array[shadow_mask]) * shadow_tint
    
    # Ensure valid range
    r_array = np.clip(r_array, 0, 255)
    g_array = np.clip(g_array, 0, 255)
    b_array = np.clip(b_array, 0, 255)
    
    # Convert back
    r_new = Image.fromarray(r_array.astype(np.uint8))
    g_new = Image.fromarray(g_array.astype(np.uint8))
    b_new = Image.fromarray(b_array.astype(np.uint8))
    
    return Image.merge('RGBA', (r_new, g_new, b_new, a))

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
        # Try different keys
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
                        # Assume base64
                        return Image.open(BytesIO(base64.b64decode(image_data)))
        
        raise ValueError("No valid image data found in input")
        
    except Exception as e:
        print(f"Error getting image: {str(e)}")
        raise

def parse_semicolon_separated_urls(url_string):
    """Parse semicolon-separated URLs from Google Script - ENHANCED"""
    if not url_string:
        return []
    
    # Remove any whitespace and split by semicolon
    url_string = url_string.strip()
    
    # Split by semicolon and clean each URL
    urls = []
    for url in url_string.split(';'):
        url = url.strip()
        if url and url.startswith('http'):  # Make sure it's a valid URL
            urls.append(url)
    
    print(f"Parsed {len(urls)} URLs from semicolon-separated string")
    for i, url in enumerate(urls):
        print(f"  URL {i+1}: {url[:60]}...")
    
    return urls

def process_wearing_shots(input_data):
    """Process group 2 - Generate wearing shots using Stable Diffusion"""
    print("=== Processing Wearing Shots Generation ===")
    
    # Get ring images from input_data
    ring_images = []
    for key in ['image1', 'image2', 'image']:
        if key in input_data and input_data[key]:
            try:
                img = get_image_from_input({key: input_data[key]})
                ring_images.append(img)
                if len(ring_images) >= 2:
                    break
            except:
                continue
    
    if len(ring_images) < 2:
        raise ValueError("Need at least 2 ring images for wearing shot generation")
    
    # Remove backgrounds from rings
    print("Removing backgrounds from rings for wearing shots...")
    ring1_no_bg = remove_background_from_image(ring_images[0])
    ring2_no_bg = remove_background_from_image(ring_images[1])
    
    # Generate wearing shots using Replicate
    wearing_shots = []
    
    if REPLICATE_AVAILABLE and REPLICATE_CLIENT:
        try:
            print("Generating wearing shots with Stable Diffusion...")
            
            # Convert rings to base64 for prompts
            def ring_to_base64(ring_img):
                buffered = BytesIO()
                ring_img.save(buffered, format="PNG")
                buffered.seek(0)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            ring1_b64 = ring_to_base64(ring1_no_bg)
            ring2_b64 = ring_to_base64(ring2_no_bg)
            
            # Define prompts for each shot
            shot_configs = [
                {
                    "prompt": "Professional jewelry product photography, elegant male hand wearing a gold wedding ring on ring finger, clean manicured hand, soft studio lighting, pure white background, close-up shot from wrist forward showing palm and fingers clearly, high detail, commercial photography style, shallow depth of field",
                    "title": "Male Hand",
                    "ring_type": "male"
                },
                {
                    "prompt": "Professional jewelry product photography, elegant female hand wearing a delicate gold wedding ring on ring finger, beautiful french manicure, soft studio lighting, pure white background, close-up shot from wrist forward showing palm and fingers clearly, high detail, commercial photography style, shallow depth of field",
                    "title": "Female Hand",
                    "ring_type": "female"
                },
                {
                    "prompt": "Professional jewelry product photography, romantic couple holding hands showing matching gold wedding rings clearly visible, male and female hands interlocked, soft studio lighting, pure white background, close-up shot focusing on the rings, high detail, commercial photography style",
                    "title": "Couple Hands",
                    "ring_type": "both"
                }
            ]
            
            for i, config in enumerate(shot_configs):
                try:
                    print(f"Generating shot {i+1}: {config['title']}")
                    
                    # Use SDXL for better quality
                    output = REPLICATE_CLIENT.run(
                        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                        input={
                            "prompt": config["prompt"],
                            "negative_prompt": "deformed hands, extra fingers, missing fingers, bad anatomy, ugly hands, low quality, blurry, distorted, amateur photo, dark lighting, colored background",
                            "width": 1024,
                            "height": 768,
                            "num_outputs": 1,
                            "scheduler": "KarrasDPM",
                            "num_inference_steps": 30,
                            "guidance_scale": 7.5,
                            "apply_watermark": False,
                            "high_noise_frac": 0.8,
                            "prompt_strength": 0.9
                        }
                    )
                    
                    if output and len(output) > 0:
                        # Download generated image
                        response = requests.get(output[0])
                        hand_img = Image.open(BytesIO(response.content))
                        
                        # Resize to target width maintaining aspect ratio
                        aspect_ratio = hand_img.height / hand_img.width
                        new_height = int(FIXED_WIDTH * aspect_ratio)
                        hand_img = hand_img.resize((FIXED_WIDTH, new_height), Image.Resampling.LANCZOS)
                        
                        # Crop to standard height if needed
                        if new_height > 900:
                            hand_img = hand_img.crop((0, 0, FIXED_WIDTH, 900))
                        
                        wearing_shots.append({
                            "image": hand_img,
                            "title": config["title"]
                        })
                        
                        print(f"Successfully generated {config['title']}")
                    else:
                        raise Exception("No output from Stable Diffusion")
                    
                except Exception as e:
                    print(f"Failed to generate shot {i+1}: {e}")
                    # Create fallback image
                    fallback = create_fallback_wearing_shot(config['title'], ring_images[0] if config['ring_type'] == 'male' else ring_images[1])
                    wearing_shots.append({
                        "image": fallback,
                        "title": config['title']
                    })
            
        except Exception as e:
            print(f"Stable Diffusion generation failed: {e}")
            # Create all fallback images with actual rings
            titles = ["Male Hand", "Female Hand", "Couple Hands"]
            for i, title in enumerate(titles):
                ring_to_use = ring_images[0] if i == 0 else ring_images[1]
                fallback = create_fallback_wearing_shot(title, ring_to_use)
                wearing_shots.append({
                    "image": fallback,
                    "title": title
                })
    else:
        print("Replicate not available, creating placeholder wearing shots")
        # Create placeholder images with actual rings
        titles = ["Male Hand", "Female Hand", "Couple Hands"]
        for i, title in enumerate(titles):
            ring_to_use = ring_images[0] if i == 0 else ring_images[1]
            placeholder = create_fallback_wearing_shot(title, ring_to_use)
            wearing_shots.append({
                "image": placeholder,
                "title": title
            })
    
    # Clean up
    for img in ring_images:
        img.close()
    
    # Combine all wearing shots into final layout
    return create_wearing_shots_layout(wearing_shots)

def create_fallback_wearing_shot(title, ring_image=None):
    """Create a fallback/placeholder wearing shot with actual ring"""
    width = FIXED_WIDTH
    height = 900
    
    # Create gradient background
    img = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(img)
    
    # Add subtle gradient
    for y in range(height):
        gray_value = 255 - int((y / height) * 10)  # Very subtle gradient
        draw.rectangle([0, y, width, y+1], fill=(gray_value, gray_value, gray_value))
    
    # If we have a ring image, place it in center
    if ring_image:
        try:
            # Remove background from ring if not already done
            if ring_image.mode != 'RGBA':
                ring_no_bg = remove_background_from_image(ring_image)
            else:
                ring_no_bg = ring_image
            
            # Resize ring to appropriate size
            ring_size = 300
            ring_no_bg.thumbnail((ring_size, ring_size), Image.Resampling.LANCZOS)
            
            # Place ring in center
            ring_x = (width - ring_no_bg.width) // 2
            ring_y = (height - ring_no_bg.height) // 2 - 50
            
            img.paste(ring_no_bg, (ring_x, ring_y), ring_no_bg)
        except Exception as e:
            print(f"Failed to add ring to fallback image: {e}")
    
    # Draw title and subtitle
    korean_font_path = download_korean_font()
    title_font = get_font(42, korean_font_path)
    subtitle_font = get_font(28, korean_font_path)
    
    # Title
    title_width, _ = get_text_size(draw, title, title_font)
    safe_draw_text(draw, (width//2 - title_width//2, height - 200), 
                  title, title_font, (60, 60, 60))
    
    # Subtitle
    subtitle = "Wearing Shot Preview"
    subtitle_width, _ = get_text_size(draw, subtitle, subtitle_font)
    safe_draw_text(draw, (width//2 - subtitle_width//2, height - 150), 
                  subtitle, subtitle_font, (120, 120, 120))
    
    # Add decorative elements
    draw.ellipse([width//2 - 100, 50, width//2 - 90, 60], fill=(220, 220, 220))
    draw.ellipse([width//2 - 5, 50, width//2 + 5, 60], fill=(220, 220, 220))
    draw.ellipse([width//2 + 90, 50, width//2 + 100, 60], fill=(220, 220, 220))
    
    return img

def create_wearing_shots_layout(wearing_shots):
    """Create final layout with all 3 wearing shots"""
    width = FIXED_WIDTH
    shot_height = 900  # Height for each shot
    spacing = 100  # Space between shots
    top_margin = 80
    bottom_margin = 80
    
    # Calculate total height
    total_height = top_margin + (shot_height * 3) + (spacing * 2) + bottom_margin
    
    # Create final image
    final_img = Image.new('RGB', (width, total_height), '#FFFFFF')
    draw = ImageDraw.Draw(final_img)
    
    # Add title
    korean_font_path = download_korean_font()
    title_font = get_font(56, korean_font_path)
    subtitle_font = get_font(20, korean_font_path)
    
    title = "WEARING SHOTS"
    title_width, _ = get_text_size(draw, title, title_font)
    safe_draw_text(draw, (width//2 - title_width//2, 30), title, title_font, (40, 40, 40))
    
    # Place each wearing shot
    current_y = top_margin
    
    for i, shot_data in enumerate(wearing_shots):
        shot_img = shot_data["image"]
        shot_title = shot_data["title"]
        
        # Resize shot to fit if needed
        if shot_img.width != width:
            aspect_ratio = shot_img.height / shot_img.width
            new_height = int(width * aspect_ratio)
            shot_img = shot_img.resize((width, new_height), Image.Resampling.LANCZOS)
        
        # Crop to exact height if needed
        if shot_img.height > shot_height:
            shot_img = shot_img.crop((0, 0, width, shot_height))
        
        # Paste wearing shot
        final_img.paste(shot_img, (0, current_y))
        
        # Add shot label
        label_y = current_y + shot_height - 40
        label_width, _ = get_text_size(draw, shot_title, subtitle_font)
        
        # Draw label background
        label_bg_rect = [width//2 - label_width//2 - 20, label_y - 5,
                        width//2 + label_width//2 + 20, label_y + 25]
        draw.rectangle(label_bg_rect, fill=(255, 255, 255, 200))
        
        # Draw label text
        safe_draw_text(draw, (width//2 - label_width//2, label_y), 
                      shot_title, subtitle_font, (80, 80, 80))
        
        current_y += shot_height + spacing
    
    # Add page indicator
    page_text = "- Wearing Shots -"
    small_font = get_font(16, korean_font_path)
    text_width, _ = get_text_size(draw, page_text, small_font)
    safe_draw_text(draw, (width//2 - text_width//2, total_height - 30), 
                  page_text, small_font, (200, 200, 200))
    
    return final_img

def process_combined_images(input_data, group_number):
    """Process combined images (groups 1, 3, 4, 5) - ENHANCED PARSING"""
    print(f"Processing combined images for group {group_number}")
    print(f"Available input keys: {list(input_data.keys())}")
    
    # Get exactly 2 images based on group number
    images = []
    
    # First priority: Check for semicolon-separated URLs
    # Updated mapping for new group structure
    main_keys = {
        1: ['image1', 'image2', 'image'],  # Group 1 now uses image1+2
        3: ['image3', 'image'],
        4: ['image4', 'image'],
        5: ['image5', 'image']
    }
    
    urls_found = False
    
    # Try main keys for this group
    for key in main_keys.get(group_number, []):
        if key in input_data and input_data[key]:
            value = input_data[key]
            print(f"Checking key '{key}' with value type: {type(value)}")
            
            if isinstance(value, str):
                # Clean the string
                value = value.strip()
                
                # Check if it contains semicolon
                if ';' in value:
                    print(f"Found semicolon-separated URLs in {key}")
                    urls = parse_semicolon_separated_urls(value)
                    
                    if len(urls) >= 2:
                        # Download each URL
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
                    # Single URL - look for second image
                    print(f"Found single URL in {key}, looking for second image...")
                    try:
                        img = download_image_from_google_drive(value)
                        images.append(img)
                    except Exception as e:
                        print(f"Failed to download single URL: {e}")
    
    # Fallback: Look for individual image keys if no semicolon-separated URLs found
    if not urls_found and len(images) < 2:
        print("No semicolon-separated URLs found, trying individual keys...")
        
        key_pairs = {
            1: ['image1', 'image2'],  # Group 1 uses image1+2
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
    
    # Validate we have exactly 2 images
    if len(images) != 2:
        print(f"ERROR: Group {group_number} requires exactly 2 images, but {len(images)} found")
        print(f"Debug info:")
        print(f"  - Group number: {group_number}")
        print(f"  - Keys checked: {main_keys.get(group_number, [])}")
        print(f"  - Images found: {len(images)}")
        
        # Print what we found in input_data
        for key in ['image', 'image1', 'image2', 'image3', 'image4', 'image5']:
            if key in input_data:
                value = input_data[key]
                if isinstance(value, str):
                    print(f"  - {key}: {value[:100]}... (contains ';': {';' in value})")
                else:
                    print(f"  - {key}: {type(value)}")
        
        raise ValueError(f"Group {group_number} requires exactly 2 images, but {len(images)} found")
    
    print(f"Successfully loaded 2 images for group {group_number}")
    
    # Calculate dimensions
    target_width = FIXED_WIDTH
    image_height = int(target_width * 1.3)  # 1200 x 1560
    
    # Layout parameters
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    IMAGE_SPACING = 200  # 200px between images
    
    total_height = TOP_MARGIN + (2 * image_height) + IMAGE_SPACING + BOTTOM_MARGIN
    
    # Create combined page
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    
    # Place images
    current_y = TOP_MARGIN
    
    for idx, img in enumerate(images):
        try:
            # Resize image
            resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
            img_resized = img.resize((target_width, image_height), resample_filter)
            
            # Paste image
            detail_page.paste(img_resized, (0, current_y))
            
            # Update position for next image
            current_y += image_height
            if idx == 0:  # Add spacing after first image
                current_y += IMAGE_SPACING
            
            img.close()
            print(f"Placed image {idx + 1} at y={current_y - image_height - (IMAGE_SPACING if idx == 1 else 0)}")
            
        except Exception as e:
            print(f"Error processing image {idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
    page_texts = {
        1: "- Images 1-2 -",  # Updated
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

def process_color_section(input_data):
    """Process group 6 - COLOR section"""
    print("Processing COLOR section")
    
    # Get the ring image
    img = get_image_from_input(input_data)
    
    # Create color section
    color_section = create_color_options_section(ring_image=img)
    
    img.close()
    
    return color_section

def process_text_section(input_data, group_number):
    """Process text-only sections (groups 7, 8)"""
    print(f"Processing text section for group {group_number}")
    
    # Get Claude text
    claude_text = (input_data.get('claude_text') or 
                  input_data.get('text_content') or 
                  input_data.get('ai_text') or 
                  input_data.get('generated_text') or '')
    
    # Clean text
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
    """Detect group number from input data with enhanced logic"""
    # Priority 1: Explicit route_number
    route_number = input_data.get('route_number', 0)
    if route_number and str(route_number).isdigit():
        group_num = int(route_number)
        print(f"Found explicit route_number: {group_num}")
        return group_num
    
    # Priority 2: group_number
    group_number = input_data.get('group_number', 0)
    if group_number and str(group_number).isdigit():
        group_num = int(group_number)
        print(f"Found explicit group_number: {group_num}")
        return group_num
    
    # Priority 3: Check for wearing shot indicators
    if any(key in str(input_data).lower() for key in ['wearing', 'wear', 'shot', 'hand']):
        return 2
    
    # Priority 4: Text type hints
    text_type = input_data.get('text_type', '').lower()
    if 'md_talk' in text_type:
        return 7
    elif 'design_point' in text_type:
        return 8
    
    # Priority 5: Check for Google Script format (semicolon-separated URLs)
    # This helps identify groups 3, 4, 5 from Google Script
    for key, group in [('image3', 3), ('image4', 4), ('image5', 5)]:
        if key in input_data and input_data[key]:
            value = input_data[key]
            if isinstance(value, str) and ';' in value:
                print(f"Detected group {group} from semicolon-separated URLs in {key}")
                return group
    
    # Priority 6: Check for color section indicators
    if 'image6' in input_data or 'image9' in input_data:
        return 6
    
    # Priority 7: Specific image keys
    # Group 1 is now default for image1/image2
    if 'image1' in input_data or 'image2' in input_data:
        return 1
    elif 'image3' in input_data:
        return 3
    elif 'image4' in input_data:
        return 4
    elif 'image5' in input_data:
        return 5
    
    # Priority 8: Check for color indicators in any field
    if any(key in str(input_data).lower() for key in ['color', 'colour', 'gold']):
        return 6
    
    # Default
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
    """Main handler for detail page creation"""
    try:
        print(f"=== V120 Detail Page Handler with Wearing Shots ===")
        
        # Get input data
        input_data = event.get('input', event)
        print(f"Input keys: {list(input_data.keys())}")
        
        # Detect group number
        group_number = detect_group_number_from_input(input_data)
        print(f"Detected group number: {group_number}")
        
        if group_number < 1 or group_number > 8:
            raise ValueError(f"Invalid group number: {group_number}")
        
        # Process based on group
        if group_number == 1:
            print("=== Processing Group 1: Combined images 1-2 ===")
            detail_page = process_combined_images(input_data, group_number)
            page_type = "combined_main"
            
        elif group_number == 2:
            print("=== Processing Group 2: Wearing shots generation ===")
            detail_page = process_wearing_shots(input_data)
            page_type = "wearing_shots"
            
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
        
        # Convert to base64
        buffered = BytesIO()
        detail_page.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue())
        
        # Remove padding for Make.com compatibility
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
            "has_background_removal": group_number in [2, 6],  # Wearing shots also remove bg
            "has_ai_generation": group_number == 2,  # New flag for AI generated content
            "format": "base64_no_padding",
            "version": "V120_WEARING_SHOTS"
        }
        
        # Send to webhook
        file_name = f"detail_group_{group_number}_{page_type}.png"
        webhook_result = send_to_webhook(detail_base64_no_padding, "detail", file_name, group_number, metadata)
        
        # Return with proper structure for Make.com
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
                "version": "V120_WEARING_SHOTS"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("V120 Detail Handler with Wearing Shots Started!")
    runpod.serverless.start({"handler": handler})
