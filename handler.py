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
    """Remove background including ring center holes - ENHANCED"""
    try:
        # Method 1: Try local rembg first with more aggressive settings
        if REMBG_AVAILABLE:
            try:
                print("Removing background using local rembg with enhanced settings...")
                
                # Initialize session with best model
                if not hasattr(remove_background_from_image, 'session'):
                    remove_background_from_image.session = new_session('u2netp')
                
                # Convert to bytes
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                
                # Remove background with very aggressive settings for jewelry
                output = remove(
                    buffered.getvalue(),
                    session=remove_background_from_image.session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=250,  # Very high threshold
                    alpha_matting_background_threshold=30,   # Very low threshold
                    alpha_matting_erode_size=1,
                    only_mask=False
                )
                
                result_image = Image.open(BytesIO(output))
                
                # Post-process to ensure ring holes are transparent
                result_image = post_process_ring_transparency(result_image)
                
                print("Background removed successfully with local rembg")
                return result_image
                
            except Exception as e:
                print(f"Local rembg failed: {e}")
        
        # Method 2: Try Replicate API with enhanced settings
        if REPLICATE_AVAILABLE and REPLICATE_CLIENT:
            try:
                print("Removing background using Replicate API with enhanced settings...")
                
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
                        "alpha_matting_foreground_threshold": 250,
                        "alpha_matting_background_threshold": 30,
                        "alpha_matting_erode_size": 1
                    }
                )
                
                if output:
                    if isinstance(output, str):
                        response = requests.get(output)
                        result_image = Image.open(BytesIO(response.content))
                    else:
                        result_image = Image.open(BytesIO(base64.b64decode(output)))
                    
                    # Post-process for ring holes
                    result_image = post_process_ring_transparency(result_image)
                    
                    print("Background removed successfully with Replicate")
                    return result_image
                    
            except Exception as e:
                print(f"Replicate background removal failed: {e}")
        
        # Method 3: Enhanced manual background removal
        print("Using enhanced manual background removal")
        result = manual_remove_background_enhanced(image)
        return post_process_ring_transparency(result)
        
    except Exception as e:
        print(f"All background removal methods failed: {e}")
        return image

def post_process_ring_transparency(image):
    """Enhanced post-process to aggressively detect and remove ring center holes"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    data = np.array(image)
    alpha_channel = data[:,:,3]
    
    # Find the center of the image
    height, width = data.shape[:2]
    center_y, center_x = height // 2, width // 2
    
    # Convert to grayscale for analysis
    gray = np.mean(data[:,:,:3], axis=2)
    
    # Method 1: Find bright/white enclosed areas more aggressively
    bright_threshold = 235  # Lower threshold to catch more areas
    bright_mask = gray > bright_threshold
    
    # Also check alpha channel - areas that weren't made transparent yet
    opaque_bright = bright_mask & (alpha_channel > 200)
    
    # Label connected components
    labeled, num_features = ndimage.label(opaque_bright)
    
    # Analyze each component
    for i in range(1, num_features + 1):
        region = labeled == i
        region_coords = np.where(region)
        
        if len(region_coords[0]) > 5:  # Minimum size
            # Calculate region properties
            region_center_y = np.mean(region_coords[0])
            region_center_x = np.mean(region_coords[1])
            region_size = len(region_coords[0])
            
            # Check if region is roughly in the center
            dist_from_center = np.sqrt((region_center_y - center_y)**2 + (region_center_x - center_x)**2)
            
            # More aggressive criteria for ring holes
            is_centered = dist_from_center < min(height, width) * 0.4
            is_reasonable_size = region_size < (height * width * 0.2)  # Not too big
            
            # Check if surrounded by non-transparent pixels (enclosed)
            # Expand region slightly and check if it hits edges
            dilated = ndimage.binary_dilation(region, iterations=5)
            touches_edge = (dilated[0,:].any() or dilated[-1,:].any() or 
                           dilated[:,0].any() or dilated[:,-1].any())
            
            if is_centered and is_reasonable_size and not touches_edge:
                data[region] = [255, 255, 255, 0]
                print(f"Removed enclosed bright region (size: {region_size}, dist from center: {dist_from_center:.1f})")
    
    # Method 2: Morphological approach to find holes
    # Find opaque regions
    opaque_mask = alpha_channel > 200
    
    # Fill holes to find the outer ring
    filled = ndimage.binary_fill_holes(opaque_mask)
    
    # Holes are the difference
    holes = filled & ~opaque_mask
    
    # Also consider bright areas within the filled region
    interior_bright = filled & bright_mask
    
    # Label and process holes
    hole_labels, num_holes = ndimage.label(holes | interior_bright)
    
    for i in range(1, num_holes + 1):
        hole_region = hole_labels == i
        hole_coords = np.where(hole_region)
        
        if len(hole_coords[0]) > 10:
            # Make hole transparent
            data[hole_region] = [255, 255, 255, 0]
            print(f"Removed hole region using morphological method")
    
    # Method 3: Circle detection for ring holes
    # This is specifically for detecting circular holes in rings
    if width > 50 and height > 50:  # Only for reasonably sized images
        # Create a circular mask for the center area
        Y, X = np.ogrid[:height, :width]
        center_area_radius = min(height, width) * 0.3
        center_mask = (X - center_x)**2 + (Y - center_y)**2 <= center_area_radius**2
        
        # Check for bright areas in the center
        center_bright = center_mask & (gray > 230) & (alpha_channel > 200)
        
        if center_bright.any():
            # Make the bright center area transparent
            data[center_bright] = [255, 255, 255, 0]
            print("Removed bright center area using circular detection")
    
    return Image.fromarray(data, 'RGBA')

def manual_remove_background(image):
    """Basic manual background removal for rings"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    data = np.array(image, dtype=np.float32)
    
    # Multiple threshold approach
    # 1. Pure white background
    white_mask = (data[:,:,0] > 240) & (data[:,:,1] > 240) & (data[:,:,2] > 240)
    
    # 2. Near white with some tolerance
    near_white = (data[:,:,0] > 230) & (data[:,:,1] > 230) & (data[:,:,2] > 230)
    
    # 3. Check color similarity (all channels similar = gray/white)
    color_diff = np.abs(data[:,:,0] - data[:,:,1]) + np.abs(data[:,:,1] - data[:,:,2])
    gray_mask = color_diff < 20
    
    # Combine masks
    background_mask = white_mask | (near_white & gray_mask)
    
    # Make background transparent
    data[background_mask] = [255, 255, 255, 0]
    
    return Image.fromarray(data.astype(np.uint8), 'RGBA')

def manual_remove_background_enhanced(image):
    """Enhanced manual background removal specifically for jewelry/rings"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    data = np.array(image, dtype=np.float32)
    
    # Multiple aggressive threshold approaches
    # 1. Pure white and near-white background
    white_mask = (data[:,:,0] > 245) & (data[:,:,1] > 245) & (data[:,:,2] > 245)
    near_white = (data[:,:,0] > 235) & (data[:,:,1] > 235) & (data[:,:,2] > 235)
    
    # 2. Gray background detection (all channels similar)
    max_diff = 15  # Stricter threshold
    color_diff = np.abs(data[:,:,0] - data[:,:,1]) + np.abs(data[:,:,1] - data[:,:,2])
    gray_mask = color_diff < max_diff
    
    # 3. Light gray detection
    light_gray = (data[:,:,0] > 220) & (data[:,:,1] > 220) & (data[:,:,2] > 220) & gray_mask
    
    # Combine all masks
    background_mask = white_mask | near_white | light_gray
    
    # Make background transparent
    data[background_mask] = [255, 255, 255, 0]
    
    # Additional step: find and remove enclosed light areas (ring holes)
    result = Image.fromarray(data.astype(np.uint8), 'RGBA')
    result = post_process_ring_transparency(result)
    
    return result

def create_ai_generated_md_talk(claude_text, width=FIXED_WIDTH):
    """Create MD Talk section with proper Korean text rendering"""
    section_height = 600
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Get Korean font
    korean_font_path = download_korean_font()
    title_font = get_font(48, korean_font_path)
    body_font = get_font(28, korean_font_path)
    
    # Title
    title = "MD TALK"
    title_width, title_height = get_text_size(draw, title, title_font)
    safe_draw_text(draw, (width//2 - title_width//2, 60), title, title_font, (40, 40, 40))
    
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
        
        lines = lines[:8]  # Limit lines
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
    
    # Draw content
    y_pos = 180
    line_height = 40
    
    for line in lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    return section_img

def create_ai_generated_design_point(claude_text, width=FIXED_WIDTH):
    """Create Design Point section with proper Korean text rendering"""
    section_height = 700
    section_img = Image.new('RGB', (width, section_height), '#FFFFFF')
    draw = ImageDraw.Draw(section_img)
    
    # Get Korean font
    korean_font_path = download_korean_font()
    title_font = get_font(48, korean_font_path)
    body_font = get_font(24, korean_font_path)
    
    # Title
    title = "DESIGN POINT"
    title_width, title_height = get_text_size(draw, title, title_font)
    safe_draw_text(draw, (width//2 - title_width//2, 60), title, title_font, (40, 40, 40))
    
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
        
        lines = lines[:10]
    else:
        lines = [
            "남성 단품은 무광 텍스처와 유광 라인의 조화가",
            "견고한 감성을 전하고 여자 단품은",
            "파베 세팅과 섬세한 밀그레인의 디테일",
            "화려하면서도 고급스러운 반영을 표현합니다"
        ]
    
    # Draw content
    y_pos = 200
    line_height = 45
    
    for line in lines:
        if line:
            line_width, _ = get_text_size(draw, line, body_font)
            safe_draw_text(draw, (width//2 - line_width//2, y_pos), line, body_font, (80, 80, 80))
        y_pos += line_height
    
    # Decorative line
    draw.rectangle([100, y_pos + 30, width - 100, y_pos + 32], fill=(220, 220, 220))
    
    return section_img

def create_color_options_section(ring_image=None):
    """Create COLOR section with enhanced color application"""
    width = FIXED_WIDTH
    height = 800
    
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
    
    # Remove background from ring
    ring_no_bg = None
    if ring_image:
        try:
            print("Removing background from ring for color section")
            ring_no_bg = remove_background_from_image(ring_image)
            
            if ring_no_bg.mode != 'RGBA':
                ring_no_bg = ring_no_bg.convert('RGBA')
                
        except Exception as e:
            print(f"Failed to remove background: {e}")
            ring_no_bg = ring_image.convert('RGBA') if ring_image else None
    
    # Enhanced color definitions with adjusted strengths
    colors = [
        ("yellow", "옐로우골드", (255, 200, 50), 0.3),      # Lighter gold (연하게)
        ("rose", "로즈골드", (235, 155, 155), 0.2),        # Less saturated rose gold (채도 낮춤)
        ("white", "화이트골드", (250, 250, 255), 0.05),    # More white (더 하얗게)
        ("antique", "무도금화이트", (255, 255, 255), 0.0)  # Pure white
    ]
    
    # Grid layout
    grid_size = 280
    padding = 60
    start_x = (width - (grid_size * 2 + padding)) // 2
    start_y = 180
    
    for i, (color_id, label, color_rgb, strength) in enumerate(colors):
        row = i // 2
        col = i % 2
        
        x = start_x + col * (grid_size + padding)
        y = start_y + row * (grid_size + 100)
        
        # Create container
        container = Image.new('RGBA', (grid_size, grid_size), (255, 255, 255, 255))
        container_draw = ImageDraw.Draw(container)
        
        # Draw border
        container_draw.rectangle([0, 0, grid_size-1, grid_size-1], 
                                fill=None, outline=(230, 230, 230), width=2)
        
        # Apply ring with color
        if ring_no_bg:
            try:
                ring_copy = ring_no_bg.copy()
                # Increase ring size from 0.8 to 0.9
                ring_copy.thumbnail((int(grid_size * 0.9), int(grid_size * 0.9)), 
                                  Image.Resampling.LANCZOS)
                
                # Apply enhanced metal color effect
                ring_tinted = apply_enhanced_metal_color(ring_copy, color_rgb, strength)
                
                # Center and paste
                paste_x = (grid_size - ring_tinted.width) // 2
                paste_y = (grid_size - ring_tinted.height) // 2
                container.paste(ring_tinted, (paste_x, paste_y), ring_tinted)
                
            except Exception as e:
                print(f"Error applying color {color_id}: {e}")
        
        # Paste container
        section_img.paste(container, (x, y))
        
        # Draw label
        label_width, _ = get_text_size(draw, label, label_font)
        safe_draw_text(draw, (x + grid_size//2 - label_width//2, y + grid_size + 20), 
                     label, label_font, (80, 80, 80))
    
    return section_img

def apply_enhanced_metal_color(image, metal_color, strength=0.3):
    """Apply enhanced metal color effect for jewelry"""
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
        
        # Apply color with luminance preservation and stronger effect
        # Preserve highlights and shadows while tinting midtones more
        highlight_mask = luminance > 0.8
        shadow_mask = luminance < 0.2
        midtone_mask = ~highlight_mask & ~shadow_mask & mask
        
        # Apply stronger color to midtones
        if midtone_mask.any():
            r_array[midtone_mask] = r_array[midtone_mask] * (1 - strength * 1.5) + (metal_r * 255) * (strength * 1.5)
            g_array[midtone_mask] = g_array[midtone_mask] * (1 - strength * 1.5) + (metal_g * 255) * (strength * 1.5)
            b_array[midtone_mask] = b_array[midtone_mask] * (1 - strength * 1.5) + (metal_b * 255) * (strength * 1.5)
        
        # Apply lighter color to highlights
        if highlight_mask.any():
            r_array[highlight_mask] = r_array[highlight_mask] * (1 - strength * 0.3) + (metal_r * 255) * (strength * 0.3)
            g_array[highlight_mask] = g_array[highlight_mask] * (1 - strength * 0.3) + (metal_g * 255) * (strength * 0.3)
            b_array[highlight_mask] = b_array[highlight_mask] * (1 - strength * 0.3) + (metal_b * 255) * (strength * 0.3)
        
        # Preserve shadows with slight tint
        if shadow_mask.any():
            r_array[shadow_mask] = r_array[shadow_mask] * (1 - strength * 0.1) + (metal_r * luminance[shadow_mask] * 255) * (strength * 0.1)
            g_array[shadow_mask] = g_array[shadow_mask] * (1 - strength * 0.1) + (metal_g * luminance[shadow_mask] * 255) * (strength * 0.1)
            b_array[shadow_mask] = b_array[shadow_mask] * (1 - strength * 0.1) + (metal_b * luminance[shadow_mask] * 255) * (strength * 0.1)
    
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
    """Parse semicolon-separated URLs from Google Script"""
    if not url_string:
        return []
    
    # Split by semicolon and clean each URL
    urls = [url.strip() for url in url_string.split(';') if url.strip()]
    print(f"Parsed {len(urls)} URLs from semicolon-separated string")
    return urls

def process_single_image(input_data, group_number):
    """Process individual images (groups 1, 2)"""
    print(f"Processing single image for group {group_number}")
    
    # Get image
    img = get_image_from_input(input_data)
    
    # Calculate dimensions
    if group_number == 1:
        new_height = int(FIXED_WIDTH * 1.375)  # 1200 x 1650
    else:
        new_height = int(FIXED_WIDTH * 1.15)   # 1200 x 1383
    
    # Resize
    resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    img_resized = img.resize((FIXED_WIDTH, new_height), resample_filter)
    
    # Create page
    TOP_MARGIN = 50
    BOTTOM_MARGIN = 50
    total_height = new_height + TOP_MARGIN + BOTTOM_MARGIN
    
    detail_page = Image.new('RGB', (FIXED_WIDTH, total_height), '#FFFFFF')
    detail_page.paste(img_resized, (0, TOP_MARGIN))
    
    img.close()
    
    # Add page indicator
    draw = ImageDraw.Draw(detail_page)
    page_text = f"- Image {group_number} -"
    
    korean_font_path = download_korean_font()
    small_font = get_font(16, korean_font_path)
    
    text_width, _ = get_text_size(draw, page_text, small_font)
    safe_draw_text(draw, (FIXED_WIDTH//2 - text_width//2, total_height - 30), 
                 page_text, small_font, (200, 200, 200))
    
    return detail_page

def process_combined_images(input_data, group_number):
    """Process combined images (groups 3, 4, 5) - FIXED for semicolon-separated URLs"""
    print(f"Processing combined images for group {group_number}")
    print(f"Input data for debugging: {input_data}")
    
    # Get exactly 2 images based on group number
    images = []
    
    # Map group numbers to expected keys from Google Script
    group_to_key_map = {
        3: 'image3',  # Google Script sends combined URLs as image3
        4: 'image4',  # Google Script sends combined URLs as image4
        5: 'image5'   # Google Script sends combined URLs as image5
    }
    
    # First, check if we have semicolon-separated URLs from Google Script
    main_key = group_to_key_map.get(group_number)
    if main_key and main_key in input_data and input_data[main_key]:
        url_string = input_data[main_key]
        print(f"Found {main_key} with value: {url_string[:100]}...")  # Debug log
        
        if isinstance(url_string, str) and ';' in url_string:
            # Parse semicolon-separated URLs
            urls = parse_semicolon_separated_urls(url_string)
            print(f"Parsed {len(urls)} URLs from semicolon-separated string")
            
            for i, url in enumerate(urls[:2]):  # Take only first 2
                try:
                    print(f"Downloading image {i+1} from: {url[:80]}...")
                    img = download_image_from_google_drive(url)
                    images.append(img)
                    print(f"Successfully downloaded image {i+1}")
                except Exception as e:
                    print(f"Failed to download image {i+1}: {e}")
        else:
            # Single URL case (shouldn't happen for groups 3-5 from Google Script)
            print(f"WARNING: Expected semicolon-separated URLs but got single URL")
            try:
                img = download_image_from_google_drive(url_string)
                images.append(img)
            except Exception as e:
                print(f"Failed to get image from {main_key}: {e}")
    else:
        print(f"ERROR: Key '{main_key}' not found or empty in input_data")
        print(f"Available keys: {list(input_data.keys())}")
    
    # If we don't have exactly 2 images, there's a problem
    if len(images) != 2:
        print(f"ERROR: Group {group_number} found {len(images)} images")
        print(f"Input data keys: {list(input_data.keys())}")
        if main_key in input_data:
            print(f"Main key '{main_key}' value: {input_data.get(main_key, 'NOT FOUND')[:200]}")
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
    
    # Priority 3: Text type hints
    text_type = input_data.get('text_type', '').lower()
    if 'md_talk' in text_type:
        return 7
    elif 'design_point' in text_type:
        return 8
    
    # Priority 4: Check for Google Script format (semicolon-separated URLs)
    # This helps identify groups 3, 4, 5 from Google Script
    for key, group in [('image3', 3), ('image4', 4), ('image5', 5)]:
        if key in input_data and input_data[key]:
            value = input_data[key]
            if isinstance(value, str) and ';' in value:
                print(f"Detected group {group} from semicolon-separated URLs in {key}")
                return group
    
    # Priority 5: Specific image keys
    if 'image1' in input_data:
        return 1
    elif 'image2' in input_data:
        return 2
    elif 'image3' in input_data or 'image4' in input_data:
        return 3
    elif 'image5' in input_data or 'image6' in input_data:
        return 4
    elif 'image7' in input_data or 'image8' in input_data:
        return 5
    elif 'image9' in input_data:
        return 6
    
    # Priority 6: Check for color indicators
    if any(key in str(input_data).lower() for key in ['color', 'colour']):
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
        print(f"=== V116 Fixed Detail Page Handler ===")
        
        # Get input data
        input_data = event.get('input', event)
        print(f"Input keys: {list(input_data.keys())}")
        
        # Detect group number
        group_number = detect_group_number_from_input(input_data)
        print(f"Detected group number: {group_number}")
        
        if group_number < 1 or group_number > 8:
            raise ValueError(f"Invalid group number: {group_number}")
        
        # Process based on group
        if group_number == 6:
            print("=== Processing Group 6: COLOR section ===")
            detail_page = process_color_section(input_data)
            page_type = "color_section"
            
        elif group_number in [7, 8]:
            print(f"=== Processing Group {group_number}: Text section ===")
            detail_page, section_type = process_text_section(input_data, group_number)
            page_type = f"text_section_{section_type}"
            
        elif group_number in [1, 2]:
            print(f"=== Processing Group {group_number}: Single image ===")
            detail_page = process_single_image(input_data, group_number)
            page_type = "individual"
            
        elif group_number in [3, 4, 5]:
            print(f"=== Processing Group {group_number}: Combined images ===")
            detail_page = process_combined_images(input_data, group_number)
            page_type = "combined"
        
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
            "has_background_removal": group_number == 6,
            "format": "base64_no_padding",
            "version": "V116_FIXED"
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
                "version": "V116_FIXED"
            }
        }

# RunPod handler
if __name__ == "__main__":
    print("V116 Fixed Detail Handler Started!")
    runpod.serverless.start({"handler": handler})
