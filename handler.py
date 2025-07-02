import runpod
import base64
import requests
from io import BytesIO
from PIL import Image
import json
import os
import re
from datetime import datetime

# Webhook URL - Google Apps Script Web App URL
WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbyi38xfpk-s66l6MJvfhGBmJjdv-FiYnh7NvtbrO1-IHGgoJ1BQd7NHXEuSvLu9Tggnlw/exec"

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

def get_image_url_or_base64(input_data):
    """Get image URL or convert to base64 data URL"""
    try:
        # Check for URL first
        image_url = (input_data.get('image_url') or 
                    input_data.get('imageUrl') or 
                    input_data.get('url') or 
                    input_data.get('webContentLink') or
                    input_data.get('image') or '')
        
        if image_url:
            print(f"Found image URL: {image_url}")
            # For Google Drive URLs, we need to get the direct download link
            if 'drive.google.com' in image_url or 'docs.google.com' in image_url:
                file_id = extract_file_id_from_url(image_url)
                if file_id:
                    # Return the direct download URL
                    return f'https://drive.google.com/uc?export=download&id={file_id}'
            return image_url
        
        # Check for base64
        image_base64 = (input_data.get('image_base64') or 
                       input_data.get('base64') or 
                       input_data.get('image_data') or 
                       input_data.get('enhanced_image') or '')
        
        if image_base64:
            print(f"Using base64 data, length: {len(image_base64)}")
            if not image_base64.startswith('data:'):
                # Add proper data URL prefix
                image_base64 = f'data:image/png;base64,{image_base64}'
            return image_base64
        
        raise ValueError("No image URL or base64 data provided")
        
    except Exception as e:
        print(f"Error getting image: {e}")
        raise

def create_html_template_individual(image_data, image_number):
    """Create HTML template for individual images (1-2)"""
    
    image_url = get_image_url_or_base64(image_data)
    
    if image_number == 1:
        # Image 1 with twinkring logo
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Twinkring Detail Page 1</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Cormorant+Garamond:wght@400;600&family=Nanum+Myeongjo:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Cormorant Garamond', 'Playfair Display', 'Nanum Myeongjo', serif;
            background: #FFFFFF;
        }}
        .detail-container {{
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            background: #FFFFFF;
        }}
        .image-section {{
            position: relative;
            width: 100%;
            padding: 50px 0;
        }}
        .main-image {{
            width: 100%;
            max-width: 1100px;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .twinkring-logo {{
            position: absolute;
            top: 15%;
            left: 50%;
            transform: translateX(-50%);
            font-family: 'Playfair Display', 'Cormorant Garamond', serif;
            font-size: 120px;
            font-weight: 400;
            color: #141414;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            letter-spacing: -2px;
        }}
        .page-indicator {{
            text-align: center;
            color: #C8C8C8;
            font-size: 16px;
            padding: 20px 0;
            font-family: 'Nanum Myeongjo', serif;
        }}
        @media (max-width: 768px) {{
            .twinkring-logo {{
                font-size: 60px;
            }}
        }}
    </style>
</head>
<body>
    <div class="detail-container">
        <div class="image-section">
            <img src="{image_url}" alt="Twinkring Jewelry" class="main-image">
            <div class="twinkring-logo">twinkring</div>
        </div>
        <div class="page-indicator">- 1 -</div>
    </div>
</body>
</html>"""
        
    elif image_number == 2:
        # Image 2 with HWIYEON section
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Twinkring Detail Page 2</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Cormorant+Garamond:wght@400;600&family=Nanum+Myeongjo:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Cormorant Garamond', 'Playfair Display', 'Nanum Myeongjo', serif;
            background: #FFFFFF;
        }}
        .detail-container {{
            width: 100%;
            max-width: 1000px;
            margin: 0 auto;
            background: #FFFFFF;
        }}
        .hwiyeon-section {{
            background: #F5F5F5;
            padding: 80px 40px;
            text-align: center;
            margin: 50px 0;
        }}
        .hwiyeon-title {{
            font-family: 'Playfair Display', 'Cormorant Garamond', serif;
            font-size: 72px;
            font-weight: 400;
            color: #282828;
            margin: 0 0 40px 0;
            letter-spacing: 2px;
        }}
        .hwiyeon-subtitle {{
            font-family: 'Nanum Myeongjo', serif;
            font-size: 24px;
            color: #505050;
            margin: 0 0 60px 0;
            line-height: 1.6;
        }}
        .hwiyeon-body {{
            font-family: 'Nanum Myeongjo', serif;
            font-size: 20px;
            color: #646464;
            line-height: 2;
            margin: 0 auto;
            max-width: 600px;
        }}
        .hwiyeon-body p {{
            margin: 20px 0;
        }}
        .image-section {{
            padding: 0 50px;
        }}
        .main-image {{
            width: 100%;
            max-width: 900px;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .page-indicator {{
            text-align: center;
            color: #C8C8C8;
            font-size: 16px;
            padding: 20px 0;
            font-family: 'Nanum Myeongjo', serif;
        }}
    </style>
</head>
<body>
    <div class="detail-container">
        <div class="hwiyeon-section">
            <h1 class="hwiyeon-title">HWIYEON</h1>
            <p class="hwiyeon-subtitle">당신의 나, 시금 이 빛 위에서 같은 시간에 닿아 있음을</p>
            <div class="hwiyeon-body">
                <p>우리는 서로 다른 길 위에서 만났고,<br>
                다른 속도로 걸어왔지만 결국 같은 종심을 향하고 있죠.</p>
                <p>곁자리 둣 나란히 이어진 둣,<br>
                서로를 향한 마음이 교차하는 순간을 다자인으로 담았습니다.</p>
            </div>
        </div>
        <div class="image-section">
            <img src="{image_url}" alt="Twinkring Jewelry" class="main-image">
        </div>
        <div class="page-indicator">- 2 -</div>
    </div>
</body>
</html>"""
    
    return html

def create_html_template_combined(images_data, include_md_talk, include_design_point, include_color_options, 
                                md_talk_content="", design_content="", route_number=0):
    """Create HTML template for combined images (3-9)"""
    
    # Get image URLs
    image_urls = []
    for img_data in images_data:
        url = get_image_url_or_base64(img_data)
        image_urls.append(url)
    
    # Start HTML
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Twinkring Detail Page {route_number}</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Cormorant+Garamond:wght@400;600&family=Nanum+Myeongjo:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Cormorant Garamond', 'Playfair Display', 'Nanum Myeongjo', serif;
            background: #FFFFFF;
        }}
        .detail-container {{
            width: 100%;
            max-width: 860px;
            margin: 0 auto;
            background: #FFFFFF;
            padding: 100px 0;
        }}
        .content-section {{
            background: #FFFFFF;
            padding: 50px 30px;
            text-align: center;
            margin-bottom: 50px;
        }}
        .section-title {{
            font-family: 'Playfair Display', 'Cormorant Garamond', serif;
            font-size: 48px;
            font-weight: 400;
            color: #282828;
            margin: 0 0 40px 0;
            letter-spacing: 1px;
        }}
        .section-body {{
            font-family: 'Nanum Myeongjo', serif;
            font-size: 24px;
            color: #3C3C3C;
            line-height: 1.8;
            margin: 0 auto;
            max-width: 700px;
        }}
        .image-wrapper {{
            margin: 0 0 120px 0;
        }}
        .product-image {{
            width: 100%;
            max-width: 760px;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .page-indicator {{
            text-align: center;
            color: #C8C8C8;
            font-size: 16px;
            margin-top: 50px;
            font-family: 'Nanum Myeongjo', serif;
        }}
        
        /* COLOR Section Styles */
        .color-section {{
            background: #FFFFFF;
            padding: 80px 30px;
            text-align: center;
            position: relative;
            margin-top: -400px;
            z-index: 10;
        }}
        .color-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 80px 60px;
            max-width: 600px;
            margin: 60px auto 0;
        }}
        .color-item {{
            text-align: center;
        }}
        .color-image {{
            width: 100%;
            max-width: 240px;
            height: auto;
            margin: 0 auto 20px;
            display: block;
        }}
        .color-label {{
            font-family: 'Cormorant Garamond', serif;
            font-size: 22px;
            color: #505050;
            font-weight: 400;
        }}
    </style>
</head>
<body>
    <div class="detail-container">"""
    
    # Add MD TALK section if needed (for images 3-4)
    if include_md_talk:
        md_lines = md_talk_content.strip().split('\n') if md_talk_content else [
            "MD TALK",
            "",
            "신부의 부케처럼 풍성하고,",
            "드레스처럼 우아한 분위기를 담은 커플링이에요.",
            "결혼이라는 가장 빛나는 순간을",
            "손끝에 남기고 싶은 분들께 추천드립니다:)"
        ]
        
        html += f"""
        <div class="content-section">
            <h2 class="section-title">{md_lines[0]}</h2>
            <div class="section-body">
                <p>{'<br>'.join(md_lines[2:])}</p>
            </div>
        </div>"""
    
    # Add DESIGN POINT section if needed (for images 5-6)
    if include_design_point:
        design_lines = design_content.strip().split('\n') if design_content else [
            "DESIGN POINT",
            "",
            "중앙의 꼬임 텍스처가 따뜻한 연결감을 표현하고,",
            "여자 단품은 파베 세팅과 메인 스톤의 화려한 반짝임을,",
            "남자 단품은 하나의 메인 스톤으로 단단한 중심을 상징합니다."
        ]
        
        html += f"""
        <div class="content-section">
            <h2 class="section-title">{design_lines[0]}</h2>
            <div class="section-body">
                <p>{'<br>'.join(design_lines[2:])}</p>
            </div>
        </div>"""
    
    # Add product images
    for i, url in enumerate(image_urls):
        html += f"""
        <div class="image-wrapper">
            <img src="{url}" alt="Product Image {i+1}" class="product-image">
        </div>"""
    
    # Add COLOR section if needed (for images 7-8-9)
    if include_color_options:
        # Get the 9th image URL for color display
        ninth_image_url = ""
        if len(images_data) >= 3:
            # Check if the last image is the 9th one
            last_img = images_data[-1]
            if '_009' in last_img.get('file_name', '') or 'thumb' in last_img.get('file_name', '').lower():
                ninth_image_url = image_urls[-1]
        
        html += f"""
        <div class="color-section">
            <h2 class="section-title">COLOR</h2>
            <div class="color-grid">
                <div class="color-item">
                    <img src="{ninth_image_url}" alt="Yellow Gold" class="color-image" 
                         style="filter: sepia(100%) saturate(200%) hue-rotate(15deg) brightness(1.2);">
                    <p class="color-label">yellow gold</p>
                </div>
                <div class="color-item">
                    <img src="{ninth_image_url}" alt="Rose Gold" class="color-image"
                         style="filter: sepia(100%) saturate(150%) hue-rotate(340deg) brightness(1.1);">
                    <p class="color-label">rose gold</p>
                </div>
                <div class="color-item">
                    <img src="{ninth_image_url}" alt="White Gold" class="color-image"
                         style="filter: brightness(1.2) contrast(0.9);">
                    <p class="color-label">white gold</p>
                </div>
                <div class="color-item">
                    <img src="{ninth_image_url}" alt="Antique White" class="color-image"
                         style="filter: sepia(20%) brightness(1.1);">
                    <p class="color-label">antique white</p>
                </div>
            </div>
        </div>"""
    
    # Add page indicator
    page_text = "- Details 3-4 -" if include_md_talk else "- Details 5-6 -" if include_design_point else "- Details 7-8-9 -"
    html += f"""
        <div class="page-indicator">{page_text}</div>
    </div>
</body>
</html>"""
    
    return html

def send_to_webhook(html_content, handler_type, file_name, route_number=0, metadata={}):
    """Send results to Google Apps Script webhook"""
    try:
        if not WEBHOOK_URL:
            print("WARNING: Webhook URL not configured, skipping webhook send")
            return None
            
        # Convert HTML to base64 for transmission
        html_base64 = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
        html_base64_no_padding = html_base64.rstrip('=')
        
        webhook_data = {
            "handler_type": handler_type,
            "file_name": file_name,
            "route_number": route_number,
            "runpod_result": {
                "output": {
                    "output": {
                        **metadata,
                        "html_content": html_content,  # Plain HTML
                        "html_base64": html_base64_no_padding  # Base64 encoded
                    }
                }
            }
        }
        
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
    """Create jewelry detail page HTML templates"""
    try:
        print(f"=== V96 HTML Template Detail Page Handler Started ===")
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
            # Combined processing
            print(f"Processing combined images: {len(input_data['images'])} images")
            
            # Check ALL file names to determine route
            all_files = [img.get('file_name', '') for img in input_data['images']]
            print(f"All file names: {all_files}")
            
            # Determine route based on input data
            route_number = input_data.get('route_number', 0)
            
            # Check if this is thumbnail group (route 5)
            is_thumbnail_group = False
            if route_number == 5 or len(input_data['images']) == 3:
                for fname in all_files:
                    if 'thumb' in fname.lower() or '_007' in fname or '_008' in fname or '_009' in fname:
                        is_thumbnail_group = True
                        break
                if is_thumbnail_group:
                    print("Detected as thumbnail group (images 7-8-9)")
            
            # Set flags based on route number or file analysis
            if route_number == 5 or is_thumbnail_group:
                print("Detected as route 5 (images 7-8-9) - Will add COLOR section")
                include_md = False
                include_colors = True
                include_design_point = False
            elif route_number == 4:
                print("Detected as route 4 (images 5-6) - Will add DESIGN POINT")
                include_md = False
                include_colors = False
                include_design_point = True
            elif route_number == 3:
                print("Detected as route 3 (images 3-4) - Will add MD TALK section")
                include_md = True
                include_colors = False
                include_design_point = False
            else:
                include_colors = input_data.get('include_color_options', False)
                include_md = input_data.get('include_md_talk', True)
                include_design_point = input_data.get('include_design_point', False)
            
            md_talk_content = input_data.get('html_section_content', '')
            design_content = input_data.get('design_content', '')
            
            # Create HTML template
            html_content = create_html_template_combined(
                input_data['images'],
                include_md,
                include_design_point,
                include_colors,
                md_talk_content,
                design_content,
                route_number
            )
            
            # Convert to base64 for Make.com
            html_base64 = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            html_base64_no_padding = html_base64.rstrip('=')
            
            # Prepare metadata
            metadata = {
                "page_type": "combined_" + ("3_4" if include_md else "5_6" if include_design_point else "7_8_9" if include_colors else "unknown"),
                "page_number": 0,
                "image_count": len(input_data['images']),
                "has_md_talk": include_md,
                "has_color_options": include_colors,
                "has_design_point": include_design_point,
                "is_thumbnail_group": is_thumbnail_group,
                "format": "html_template",
                "status": "success",
                "version": "V96"
            }
            
            # Send to webhook if configured
            webhook_result = send_to_webhook(
                html_content,
                "detail",
                f"combined_{route_number}.html",
                route_number,
                metadata
            )
            
            if webhook_result:
                metadata["webhook_result"] = webhook_result
            
            print(f"Successfully created combined HTML template")
            
            return {
                "output": {
                    "html_content": html_content,
                    "html_base64": html_base64_no_padding,
                    **metadata
                }
            }
        
        # Individual image processing (for images 1 and 2)
        image_number = int(input_data.get('image_number', 1))
        file_name = input_data.get('file_name', 'unknown.jpg')
        
        # Auto-detect image number from filename
        if '_001' in file_name:
            image_number = 1
        elif '_002' in file_name:
            image_number = 2
        
        print(f"Processing INDIVIDUAL image: {file_name} (Image #{image_number})")
        
        # Handle single image URL in 'image' field
        if 'image' in input_data and not input_data.get('image_url'):
            input_data['image_url'] = input_data['image']
            print(f"Using 'image' field as image_url: {input_data['image_url']}")
        
        # Create HTML template
        html_content = create_html_template_individual(input_data, image_number)
        
        # Convert to base64 for Make.com
        html_base64 = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
        html_base64_no_padding = html_base64.rstrip('=')
        
        # Prepare metadata
        metadata = {
            "page_number": image_number,
            "page_type": "individual",
            "file_name": file_name,
            "has_logo": image_number == 1,
            "has_hwiyeon": image_number == 2,
            "format": "html_template",
            "status": "success",
            "version": "V96"
        }
        
        # Send to webhook if configured
        webhook_result = send_to_webhook(
            html_content,
            "detail",
            f"page_{image_number}.html",
            image_number,
            metadata
        )
        
        if webhook_result:
            metadata["webhook_result"] = webhook_result
        
        print(f"Successfully created INDIVIDUAL HTML template")
        print(f"Has logo: {image_number == 1}, Has HWIYEON: {image_number == 2}")
        
        return {
            "output": {
                "html_content": html_content,
                "html_base64": html_base64_no_padding,
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
                "version": "V96"
            }
        }

# RunPod handler registration
runpod.serverless.start({"handler": handler})
