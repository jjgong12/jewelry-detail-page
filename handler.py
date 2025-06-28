import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io
import json
import os

def get_text_dimensions(draw, text, font):
    """Get text dimensions compatible with all PIL versions"""
    try:
        # Try new method first (PIL 8.0+)
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fall back to old method
        return draw.textsize(text, font=font)

def create_text_block(text, width=760):
    """Create elegant text block with Korean font support"""
    if not text:
        return Image.new('RGBA', (1, 1), (255, 255, 255, 0))
    
    # Create temporary image for text
    temp_img = Image.new('RGBA', (width, 500), (255, 255, 255, 0))
    draw = ImageDraw.Draw(temp_img)
    
    # Font selection with fallbacks
    font = None
    font_size = 28
    font_paths = [
        "/tmp/NanumMyeongjo.ttf",  # Downloaded Korean font
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
    
    # Fallback to default if no font found
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
    
    # Word wrap
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        text_width, _ = get_text_dimensions(draw, test_line, font)
            
        if text_width > width - 40:  # 40px padding
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
        else:
            current_line.append(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate text height
    line_height = 40
    text_height = max(len(lines) * line_height + 40, 100)
    
    # Create actual text image
    text_img = Image.new('RGBA', (width, text_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_img)
    
    # Draw text centered
    y = 20
    for line in lines:
        text_width, _ = get_text_dimensions(draw, line, font)
        x = (width - text_width) // 2
        draw.text((x, y), line, font=font, fill=(80, 80, 80))
        y += line_height
    
    return text_img

def get_image_from_input(input_data):
    """Get image from base64 string (from Google Script)"""
    try:
        # Priority 1: Direct base64 fields
        image_base64 = None
        base64_fields = [
            'image_base64', 
            'image_base64_with_padding',
            'imageBase64', 
            'base64', 
            'image_data',
            'image'
        ]
        
        for field in base64_fields:
            if field in input_data and input_data[field]:
                image_base64 = input_data[field]
                print(f"Found base64 data in field: {field}")
                break
        
        if not image_base64:
            raise ValueError("No base64 image data found")
        
        print(f"Base64 data length: {len(image_base64)}")
        
        # Add padding if needed
        missing_padding = len(image_base64) % 4
        if missing_padding:
            image_base64 += '=' * (4 - missing_padding)
            print(f"Added {4 - missing_padding} padding characters")
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        img = Image.open(BytesIO(image_data))
        print(f"Image decoded successfully: {img.size}")
        return img
        
    except Exception as e:
        print(f"Error getting image: {e}")
        raise

def handler(event):
    """Create individual jewelry detail page"""
    try:
        # Debug: Print event structure
        print(f"=== EVENT STRUCTURE ===")
        print(f"Event type: {type(event)}")
        print(f"Event keys: {event.keys() if isinstance(event, dict) else 'Not a dict'}")
        
        # Find input data
        input_data = event.get('input', event)
        print(f"Input data keys: {input_data.keys() if isinstance(input_data, dict) else 'Not a dict'}")
        
        # Get parameters
        claude_advice = input_data.get('claude_advice', '')
        image_number = int(input_data.get('image_number', 1))
        file_name = input_data.get('file_name', '')
        
        print(f"Processing {file_name} (Image #{image_number})")
        print(f"Claude advice: {claude_advice[:50]}..." if claude_advice else "No Claude advice")
        
        # Get image from base64
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
        else:  # Details (3-6)
            PAGE_WIDTH = 860
            IMAGE_HEIGHT = 1147
            CONTENT_WIDTH = 760
        
        # Section heights
        TOP_MARGIN = 100
        IMAGE_SECTION = IMAGE_HEIGHT + 100
        TEXT_SECTION = 300 if claude_advice else 100
        BOTTOM_MARGIN = 100
        
        # Total height
        TOTAL_HEIGHT = TOP_MARGIN + IMAGE_SECTION + TEXT_SECTION + BOTTOM_MARGIN
        
        # Create canvas
        detail_page = Image.new('RGB', (PAGE_WIDTH, TOTAL_HEIGHT), '#FAFAFA')
        
        # Resize image with aspect ratio
        height_ratio = IMAGE_HEIGHT / img.height
        temp_width = int(img.width * height_ratio)
        
        # Use compatible resampling filter
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
        
        # Apply subtle enhancement for luxury feel
        if claude_advice and ('luxury' in claude_advice.lower() or 'premium' in claude_advice.lower() or 
                             '프리미엄' in claude_advice or '럭셔리' in claude_advice):
            enhancer = ImageEnhance.Brightness(img_cropped)
            img_cropped = enhancer.enhance(1.05)
            enhancer = ImageEnhance.Contrast(img_cropped)
            img_cropped = enhancer.enhance(1.1)
        
        # Paste image
        x_position = (PAGE_WIDTH - img_cropped.width) // 2
        detail_page.paste(img_cropped, (x_position, TOP_MARGIN))
        
        # Add Claude's advice text (if available)
        if claude_advice and claude_advice.strip():
            text_img = create_text_block(claude_advice, CONTENT_WIDTH)
            if text_img.width > 1 and text_img.height > 1:
                text_x = (PAGE_WIDTH - text_img.width) // 2
                text_y = TOP_MARGIN + IMAGE_HEIGHT + 50
                
                if text_img.mode == 'RGBA':
                    detail_page.paste(text_img, (text_x, text_y), text_img)
                else:
                    detail_page.paste(text_img, (text_x, text_y))
        
        # Add subtle page indicator
        draw = ImageDraw.Draw(detail_page)
        page_text = f"- {image_number} -"
        
        # Use same font selection logic
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
        draw.text((PAGE_WIDTH//2 - text_width//2, TOTAL_HEIGHT - 50), 
                 page_text, fill=(200, 200, 200), font=small_font)
        
        # Save to base64
        buffer = io.BytesIO()
        detail_page.save(buffer, format='JPEG', quality=95, optimize=True)
        buffer.seek(0)
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        result_base64_no_padding = result_base64.rstrip('=')
        
        print(f"Successfully created detail page for {file_name}")
        print(f"Output base64 length: {len(result_base64_no_padding)}")
        
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
