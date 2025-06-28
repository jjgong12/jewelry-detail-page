import runpod
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io
import json

def download_image_from_url(url):
    """Download image from URL (Google Drive or direct link)"""
    try:
        # Handle Google Drive URLs
        if 'drive.google.com' in url:
            # Convert to direct download link
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
                url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        raise

def create_text_block(text, width=760):
    """Create elegant text block with Korean font support"""
    # Create temporary image for text
    temp_img = Image.new('RGBA', (width, 500), (255, 255, 255, 0))
    draw = ImageDraw.Draw(temp_img)
    
    try:
        # Try to use a nice font
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 28)
    except:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Word wrap
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] > width - 40:  # 40px padding
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
    text_height = len(lines) * line_height + 40
    
    # Create actual text image
    text_img = Image.new('RGBA', (width, text_height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_img)
    
    # Draw text centered
    y = 20
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y), line, font=font, fill=(80, 80, 80))
        y += line_height
    
    return text_img

def handler(event):
    """Create individual jewelry detail page"""
    try:
        # Find input data
        input_data = event.get('input', event)
        
        # Get parameters with proper key names
        image_url = input_data.get('image_url', '')
        claude_advice = input_data.get('claude_advice', '')
        image_number = int(input_data.get('image_number', 1))
        file_name = input_data.get('file_name', '')
        
        print(f"Processing {file_name} (Image #{image_number})")
        
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
        
        # 1. Download and process image
        img = download_image_from_url(image_url)
        
        # Resize with aspect ratio
        height_ratio = IMAGE_HEIGHT / img.height
        temp_width = int(img.width * height_ratio)
        img_resized = img.resize((temp_width, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        
        # Center crop if needed
        if temp_width > CONTENT_WIDTH:
            left = (temp_width - CONTENT_WIDTH) // 2
            img_cropped = img_resized.crop((left, 0, left + CONTENT_WIDTH, IMAGE_HEIGHT))
        else:
            img_cropped = img_resized
        
        # Apply subtle enhancement for luxury feel
        if claude_advice and ('luxury' in claude_advice.lower() or 'premium' in claude_advice.lower()):
            enhancer = ImageEnhance.Brightness(img_cropped)
            img_cropped = enhancer.enhance(1.05)
            enhancer = ImageEnhance.Contrast(img_cropped)
            img_cropped = enhancer.enhance(1.1)
        
        # Paste image
        x_position = (PAGE_WIDTH - img_cropped.width) // 2
        detail_page.paste(img_cropped, (x_position, TOP_MARGIN))
        
        # 2. Add Claude's advice text (if available)
        if claude_advice:
            text_img = create_text_block(claude_advice, CONTENT_WIDTH)
            text_x = (PAGE_WIDTH - text_img.width) // 2
            text_y = TOP_MARGIN + IMAGE_HEIGHT + 50
            
            if text_img.mode == 'RGBA':
                detail_page.paste(text_img, (text_x, text_y), text_img)
            else:
                detail_page.paste(text_img, (text_x, text_y))
        
        # 3. Add subtle page indicator
        draw = ImageDraw.Draw(detail_page)
        page_text = f"- {image_number} -"
        try:
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            small_font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), page_text, font=small_font)
        text_width = bbox[2] - bbox[0]
        draw.text((PAGE_WIDTH//2 - text_width//2, TOTAL_HEIGHT - 50), 
                 page_text, fill=(200, 200, 200), font=small_font)
        
        # Save to base64
        buffer = io.BytesIO()
        detail_page.save(buffer, format='JPEG', quality=95, optimize=True)
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Remove padding for Make.com
        result_base64 = result_base64.rstrip('=')
        
        return {
            "output": {
                "detail_page": result_base64,
                "page_number": image_number,
                "file_name": file_name,
                "dimensions": {
                    "width": PAGE_WIDTH,
                    "height": TOTAL_HEIGHT
                },
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
                "error_type": type(e).__name__
            }
        }

# RunPod handler registration
runpod.serverless.start({"handler": handler})
