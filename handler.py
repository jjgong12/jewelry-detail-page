import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import logging
import string
import requests
import replicate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# CUBIC DETAIL ENHANCEMENT HANDLER V2
# VERSION: Cubic-Sparkle-V2-All-Corrections
# 모든 보정 작업 통합 버전
################################

VERSION = "Cubic-Sparkle-V2-All-Corrections"

# ===== GLOBAL INITIALIZATION =====
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
REPLICATE_CLIENT = None

if REPLICATE_API_TOKEN:
    try:
        REPLICATE_CLIENT = replicate.Client(api_token=REPLICATE_API_TOKEN)
        logger.info("✅ Replicate client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Replicate: {e}")

def decode_base64_fast(base64_str: str) -> bytes:
    """Fast base64 decode with padding handling"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
        
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        base64_str = ''.join(base64_str.split())
        
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            return decoded
        except Exception:
            no_pad = base64_str.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def image_to_base64(image):
    """Convert image to base64 with padding for Google Script compatibility"""
    buffered = BytesIO()
    
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA for transparency")
        image = image.convert('RGBA')
    
    logger.info("💎 Saving RGBA image as PNG with compression level 3")
    image.save(buffered, format='PNG', compress_level=3, optimize=True)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str  # WITH padding

def find_input_data(data):
    """Extract input image data from various formats"""
    if isinstance(data, str) and len(data) > 50:
        return data
    
    if isinstance(data, dict):
        priority_keys = ['image', 'enhanced_image', 'thumbnail', 'image_base64', 'base64', 'img']
        
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        for key in ['input', 'data', 'output']:
            if key in data:
                if isinstance(data[key], str) and len(data[key]) > 50:
                    return data[key]
                elif isinstance(data[key], dict):
                    result = find_input_data(data[key])
                    if result:
                        return result
    
    return None

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type - AC, AB, or other"""
    if not filename:
        return "other"
    
    filename_lower = filename.lower()
    
    if 'ac_' in filename_lower:
        return "ac_pattern"
    elif 'ab_' in filename_lower:
        return "ab_pattern"
    else:
        return "other"

def auto_white_balance_fast(image: Image.Image) -> Image.Image:
    """Fast white balance - preserving transparency"""
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA for white balance")
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_img = Image.merge('RGB', (r, g, b))
    
    img_array = np.array(rgb_img, dtype=np.float32)
    
    sampled = img_array[::15, ::15]
    gray_mask = (
        (np.abs(sampled[:,:,0] - sampled[:,:,1]) < 15) & 
        (np.abs(sampled[:,:,1] - sampled[:,:,2]) < 15) &
        (sampled[:,:,0] > 180)
    )
    
    if np.sum(gray_mask) > 10:
        r_avg = np.mean(sampled[gray_mask, 0])
        g_avg = np.mean(sampled[gray_mask, 1])
        b_avg = np.mean(sampled[gray_mask, 2])
        
        gray_avg = (r_avg + g_avg + b_avg) / 3
        
        img_array[:,:,0] *= (gray_avg / r_avg) if r_avg > 0 else 1
        img_array[:,:,1] *= (gray_avg / g_avg) if g_avg > 0 else 1
        img_array[:,:,2] *= (gray_avg / b_avg) if b_avg > 0 else 1
    
    rgb_balanced = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    r2, g2, b2 = rgb_balanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    if result.mode != 'RGBA':
        logger.error("❌ WARNING: White balance result is not RGBA!")
        result = result.convert('RGBA')
    
    return result

def apply_pattern_enhancement_transparent(image: Image.Image, pattern_type: str) -> Image.Image:
    """Apply pattern enhancement while preserving transparency - AC 20%, AB 16%, Other 5%"""
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA in pattern enhancement")
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    img_array = np.array(rgb_image, dtype=np.float32)
    
    if pattern_type == "ac_pattern":
        logger.info("🔍 AC Pattern - Applying 20% white overlay with brightness 1.03")
        white_overlay = 0.20
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.03)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
        logger.info("✅ AC Pattern enhancement applied with 20% white overlay")
    
    elif pattern_type == "ab_pattern":
        logger.info("🔍 AB Pattern - Applying 16% white overlay with brightness 1.03")
        white_overlay = 0.16
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        
        img_array[:,:,0] *= 0.96
        img_array[:,:,1] *= 0.98
        img_array[:,:,2] *= 1.02
        
        cool_overlay = np.array([240, 248, 255], dtype=np.float32)
        img_array = img_array * 0.95 + cool_overlay * 0.05
        
        img_array = np.clip(img_array, 0, 255)
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.88)
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.03)
        
        logger.info("✅ AB Pattern enhancement applied with 16% white overlay")
        
    else:
        logger.info("🔍 Other Pattern - Applying 5% white overlay with brightness 1.12")
        white_overlay = 0.05
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.12)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.99)
        
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.5)
        
        logger.info("✅ Other Pattern enhancement applied with 5% white overlay and brightness 1.12")
    
    contrast = ImageEnhance.Contrast(rgb_image)
    rgb_image = contrast.enhance(1.1)
    
    sharpness = ImageEnhance.Sharpness(rgb_image)
    rgb_image = sharpness.enhance(1.8)
    
    r2, g2, b2 = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
    
    logger.info(f"✅ Enhancement applied with contrast 1.1 and updated brightness. Mode: {enhanced_image.mode}")
    
    if enhanced_image.mode != 'RGBA':
        logger.error("❌ WARNING: Enhanced image is not RGBA!")
        enhanced_image = enhanced_image.convert('RGBA')
    
    return enhanced_image

def apply_swinir_enhancement(image: Image.Image) -> Image.Image:
    """Apply SwinIR enhancement while preserving transparency"""
    if not REPLICATE_CLIENT:
        logger.warning("SwinIR skipped - no Replicate client")
        return image
    
    try:
        logger.info("🎨 Applying SwinIR enhancement with transparency support")
        
        if image.mode != 'RGBA':
            logger.warning(f"⚠️ Converting {image.mode} to RGBA for SwinIR")
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        buffered = BytesIO()
        rgb_image.save(buffered, format="PNG", optimize=True, compress_level=3)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        output = REPLICATE_CLIENT.run(
            "jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396546b6c9c8f9d47a021a",
            input={
                "image": img_data_url,
                "task_type": "Real-World Image Super-Resolution",
                "scale": 1,
                "noise_level": 10,
                "jpeg_quality": 50
            }
        )
        
        if output:
            if isinstance(output, str):
                response = requests.get(output)
                enhanced_image = Image.open(BytesIO(response.content))
            else:
                enhanced_image = Image.open(BytesIO(base64.b64decode(output)))
            
            r2, g2, b2 = enhanced_image.split()
            result = Image.merge('RGBA', (r2, g2, b2, a))
            
            logger.info("✅ SwinIR enhancement successful with transparency")
            
            if result.mode != 'RGBA':
                logger.error("❌ WARNING: SwinIR result is not RGBA!")
                result = result.convert('RGBA')
            
            return result
            
    except Exception as e:
        logger.warning(f"SwinIR error: {str(e)}")
        
    return image

def detect_ring_structure(image):
    """Advanced ring detection using multiple techniques"""
    logger.info("🔍 Starting advanced ring structure detection...")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    gray = np.array(image.convert('L'))
    h, w = gray.shape
    
    edges_canny = cv2.Canny(gray, 50, 150)
    edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
    edges_sobel = (edges_sobel / edges_sobel.max() * 255).astype(np.uint8)
    
    combined_edges = edges_canny | (edges_sobel > 50)
    
    contours, _ = cv2.findContours(combined_edges.astype(np.uint8), 
                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    ring_candidates = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 100 or area > h * w * 0.8:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, (width, height), angle) = ellipse
            
            aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
            
            if circularity > 0.3 or aspect_ratio > 0.5:
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                kernel = np.ones((5,5), np.uint8)
                eroded = cv2.erode(mask, kernel, iterations=2)
                
                inner_contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
                
                for inner in inner_contours:
                    inner_area = cv2.contourArea(inner)
                    if inner_area > area * 0.1:
                        ring_candidates.append({
                            'outer_contour': contour,
                            'inner_contour': inner,
                            'center': center,
                            'size': (width, height),
                            'angle': angle,
                            'circularity': circularity,
                            'aspect_ratio': aspect_ratio,
                            'area': area,
                            'inner_area': inner_area
                        })
    
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=50, param2=30, minRadius=10, maxRadius=int(min(h, w)/2))
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            mask = np.zeros(gray.shape, np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            cv2.circle(mask, (x, y), max(1, r//3), 0, -1)
            
            overlap = cv2.bitwise_and(combined_edges, mask)
            if np.sum(overlap) > r * 2 * np.pi * 0.3:
                ring_candidates.append({
                    'type': 'circle',
                    'center': (x, y),
                    'radius': r,
                    'inner_radius': r//3
                })
    
    logger.info(f"✅ Found {len(ring_candidates)} ring candidates")
    return ring_candidates

def ensure_ring_holes_transparent_ultra_v4_ring_aware(image: Image.Image) -> Image.Image:
    """ULTRA PRECISE V4 RING-AWARE HOLE DETECTION - SIMPLIFIED VERSION"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 ULTRA PRECISE V4 RING-AWARE Hole Detection")
    
    ring_candidates = detect_ring_structure(image)
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    
    h, w = alpha_array.shape
    
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    
    holes_mask = np.zeros_like(alpha_array, dtype=np.float32)
    
    # Process ring interiors
    for ring in ring_candidates:
        if 'center' in ring:
            ring_interior_mask = np.zeros_like(alpha_array, dtype=np.uint8)
            
            if 'radius' in ring:
                cv2.circle(ring_interior_mask, tuple(map(int, ring['center'])), 
                          int(ring.get('inner_radius', ring['radius'] * 0.6)), 255, -1)
            elif 'inner_contour' in ring:
                cv2.drawContours(ring_interior_mask, [ring['inner_contour']], -1, 255, -1)
            
            if np.any(ring_interior_mask > 0):
                interior_brightness = np.mean(gray[ring_interior_mask > 0])
                interior_v = np.mean(v_channel[ring_interior_mask > 0])
                interior_saturation = np.mean(s_channel[ring_interior_mask > 0])
                
                if (interior_brightness > 220 or interior_v > 225) and interior_saturation < 30:
                    holes_mask[ring_interior_mask > 0] = 255
                    logger.info("✅ Ring interior identified as hole")
    
    # Basic hole detection
    very_bright_v = v_channel > 248
    very_bright_l = l_channel > 243
    very_bright_gray = gray > 243
    
    mean_saturation = np.mean(s_channel[alpha_array > 128])
    saturation_threshold = min(20, mean_saturation * 0.3)
    very_low_saturation = s_channel < saturation_threshold
    
    alpha_holes = alpha_array < 20
    
    potential_holes = ((very_bright_v | very_bright_l | very_bright_gray) & 
                      very_low_saturation) | alpha_holes
    
    # Clean up noise
    kernel_size = max(3, min(7, int(np.sqrt(h * w) / 100)))
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    potential_holes = cv2.morphologyEx(potential_holes.astype(np.uint8), cv2.MORPH_OPEN, kernel_clean)
    potential_holes = cv2.morphologyEx(potential_holes, cv2.MORPH_CLOSE, kernel_clean)
    
    # Apply holes
    if np.any(holes_mask > 0) or np.any(potential_holes > 0):
        combined_holes = holes_mask | potential_holes
        alpha_array[combined_holes > 0] = 0
        logger.info("✅ Ring holes applied")
    
    a_new = Image.fromarray(alpha_array)
    result = Image.merge('RGBA', (r, g, b, a_new))
    
    if result.mode != 'RGBA':
        logger.error("❌ WARNING: Result is not RGBA!")
        result = result.convert('RGBA')
    
    return result

def detect_cubic_regions(image: Image.Image, sensitivity=1.0):
    """큐빅/보석 영역 감지"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    alpha_array = np.array(image.split()[3], dtype=np.uint8)
    
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    
    h, s, v = cv2.split(hsv)
    l, a_chan, b_chan = cv2.split(lab)
    
    white_cubic = (
        (l > 240 * sensitivity) & 
        (s < 30) & 
        (alpha_array > 200)
    )
    
    color_cubic = (
        (l > 200 * sensitivity) & 
        (s > 100) & 
        (v > 200 * sensitivity) &
        (alpha_array > 200)
    )
    
    highlights = (
        (l > 250) & 
        (v > 250) &
        (alpha_array > 200)
    )
    
    cubic_mask = white_cubic | color_cubic | highlights
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cubic_mask = cv2.morphologyEx(cubic_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cubic_mask = cv2.morphologyEx(cubic_mask, cv2.MORPH_CLOSE, kernel)
    
    return cubic_mask.astype(bool), white_cubic, color_cubic, highlights

def enhance_cubic_sparkle(image: Image.Image, intensity=1.0) -> Image.Image:
    """큐빅의 반짝임과 디테일 강화"""
    logger.info("💎 Starting cubic detail enhancement...")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
    alpha_array = np.array(a, dtype=np.uint8)
    
    cubic_mask, white_cubic, color_cubic, highlights = detect_cubic_regions(image, intensity)
    
    cubic_count = np.sum(cubic_mask)
    logger.info(f"✨ Detected {cubic_count} cubic pixels")
    
    if cubic_count == 0:
        logger.info("No cubic regions detected, returning original")
        return image
    
    # Edge enhancement
    logger.info("🔷 Enhancing cubic edges...")
    gray = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 100, 200)
    edges3 = cv2.Canny(gray, 150, 250)
    
    all_edges = edges1 | edges2 | edges3
    cubic_edges = all_edges & cubic_mask
    
    edge_dilated = cv2.dilate(cubic_edges.astype(np.uint8), np.ones((3,3)), iterations=1)
    
    # Highlight boosting
    logger.info("✨ Boosting highlights...")
    lab = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    
    bright_mask = (l_channel > 240) & cubic_mask
    if np.any(bright_mask):
        boost_factor = 1.1 * intensity
        rgb_array[bright_mask] = np.minimum(rgb_array[bright_mask] * boost_factor, 255)
    
    # Specular reflections
    logger.info("💫 Adding specular reflections...")
    
    num_labels, labels = cv2.connectedComponents(cubic_mask.astype(np.uint8))
    
    for i in range(1, min(num_labels, 500)):
        cubic_region = (labels == i)
        region_size = np.sum(cubic_region)
        
        if region_size < 10 or region_size > 10000:
            continue
        
        coords = np.where(cubic_region)
        if len(coords[0]) == 0:
            continue
            
        center_y = int(np.mean(coords[0]))
        center_x = int(np.mean(coords[1]))
        
        region_brightness = l_channel[cubic_region]
        if len(region_brightness) == 0:
            continue
            
        max_bright_idx = np.argmax(region_brightness)
        max_y = coords[0][max_bright_idx]
        max_x = coords[1][max_bright_idx]
        
        sparkle_radius = max(3, int(np.sqrt(region_size) * 0.3))
        
        for dy in range(-sparkle_radius, sparkle_radius + 1):
            for dx in range(-sparkle_radius, sparkle_radius + 1):
                y, x = max_y + dy, max_x + dx
                
                if 0 <= y < rgb_array.shape[0] and 0 <= x < rgb_array.shape[1]:
                    dist = np.sqrt(dy**2 + dx**2)
                    if dist <= sparkle_radius:
                        sparkle_intensity = (1 - (dist / sparkle_radius)) * intensity * 0.5
                        
                        current_color = rgb_array[y, x]
                        brightness_boost = (255 - current_color) * sparkle_intensity
                        rgb_array[y, x] = np.minimum(current_color + brightness_boost, 255)
    
    # Selective sharpening
    logger.info("🔪 Sharpening cubic areas...")
    if np.any(cubic_mask):
        blurred = cv2.GaussianBlur(rgb_array, (5, 5), 1.0)
        sharpened = rgb_array + (rgb_array - blurred) * (1.5 * intensity)
        
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                np.clip(sharpened[:,:,c], 0, 255),
                rgb_array[:,:,c]
            )
    
    # Color enhancement for color cubics
    logger.info("🌈 Enhancing color cubics...")
    if np.any(color_cubic):
        hsv = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_float = hsv.astype(np.float32)
        
        saturation_boost = 1.3 * intensity
        hsv_float[:,:,1] = np.where(
            color_cubic,
            np.minimum(hsv_float[:,:,1] * saturation_boost, 255),
            hsv_float[:,:,1]
        )
        
        value_boost = 1.05 * intensity
        hsv_float[:,:,2] = np.where(
            color_cubic,
            np.minimum(hsv_float[:,:,2] * value_boost, 255),
            hsv_float[:,:,2]
        )
        
        rgb_array = cv2.cvtColor(hsv_float.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
    # Edge highlighting
    logger.info("✨ Highlighting edges...")
    if np.any(edge_dilated):
        edge_highlight_strength = 0.3 * intensity
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                edge_dilated,
                np.minimum(rgb_array[:,:,c] + (255 - rgb_array[:,:,c]) * edge_highlight_strength, 255),
                rgb_array[:,:,c]
            )
    
    # Final adjustments
    logger.info("🎨 Final adjustments...")
    
    if np.any(cubic_mask):
        mean_val = np.mean(rgb_array[cubic_mask])
        contrast_factor = 1.1 * intensity
        
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                np.clip((rgb_array[:,:,c] - mean_val) * contrast_factor + mean_val, 0, 255),
                rgb_array[:,:,c]
            )
    
    rgb_enhanced = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
    r2, g2, b2 = rgb_enhanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    sharpness = ImageEnhance.Sharpness(result)
    result = sharpness.enhance(1.0 + (0.2 * intensity))
    
    logger.info("✅ Cubic enhancement complete!")
    
    return result

def handler(event):
    """RunPod handler for all corrections + cubic detail enhancement"""
    try:
        logger.info(f"=== Cubic Detail Enhancement V2 {VERSION} Started ===")
        logger.info("🎯 모든 보정 작업 통합 버전")
        logger.info("✨ Processing Order:")
        logger.info("  1. White Balance")
        logger.info("  2. Pattern Enhancement (AC/AB/Other)")
        logger.info("  3. SwinIR Quality Enhancement (Optional)")
        logger.info("  4. Ring Hole Detection")
        logger.info("  5. Cubic Detail Enhancement")
        
        # 입력 데이터 추출
        image_data_str = find_input_data(event)
        
        if not image_data_str and isinstance(event, dict):
            if 'thumbnail' in event:
                image_data_str = event['thumbnail']
            elif 'output' in event and isinstance(event['output'], dict):
                if 'thumbnail' in event['output']:
                    image_data_str = event['output']['thumbnail']
        
        if not image_data_str:
            raise ValueError("No input image data found")
        
        # 파라미터 추출
        filename = event.get('filename', '')
        intensity = float(event.get('intensity', 1.0))
        intensity = max(0.1, min(2.0, intensity))
        apply_swinir = event.get('apply_swinir', True)
        apply_pattern = event.get('pattern_enhancement', True)
        
        logger.info(f"Parameters: intensity={intensity}, swinir={apply_swinir}, pattern={apply_pattern}")
        
        # 이미지 디코딩
        image_bytes = decode_base64_fast(image_data_str)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGBA':
            logger.info(f"Converting {image.mode} to RGBA")
            image = image.convert('RGBA')
        
        original_size = image.size
        logger.info(f"Input image size: {original_size}")
        
        # 1. White Balance
        logger.info("⚖️ Step 1: Applying white balance")
        image = auto_white_balance_fast(image)
        
        # 2. Pattern Enhancement
        if apply_pattern:
            logger.info("🎨 Step 2: Applying pattern enhancement")
            pattern_type = detect_pattern_type(filename)
            
            detected_type = {
                "ac_pattern": "무도금화이트(0.20)",
                "ab_pattern": "무도금화이트-쿨톤(0.16)",
                "other": "기타색상(0.05)"
            }.get(pattern_type, "기타색상")
            
            logger.info(f"Detected pattern: {pattern_type} - {detected_type}")
            image = apply_pattern_enhancement_transparent(image, pattern_type)
        else:
            pattern_type = "none"
            detected_type = "보정없음"
        
        # 3. SwinIR Enhancement
        if apply_swinir:
            logger.info("🚀 Step 3: Applying SwinIR enhancement")
            image = apply_swinir_enhancement(image)
        
        # 4. Ring Hole Detection
        logger.info("🔍 Step 4: Applying ring hole detection")
        image = ensure_ring_holes_transparent_ultra_v4_ring_aware(image)
        
        # 5. Cubic Detail Enhancement
        logger.info("💎 Step 5: Applying cubic detail enhancement")
        enhanced_image = enhance_cubic_sparkle(image, intensity)
        
        # Base64로 인코딩
        output_base64 = image_to_base64(enhanced_image)
        
        # 통계 정보
        cubic_mask, _, _, _ = detect_cubic_regions(image)
        cubic_pixel_count = np.sum(cubic_mask)
        cubic_percentage = (cubic_pixel_count / (image.size[0] * image.size[1])) * 100
        
        return {
            "output": {
                "enhanced_image": output_base64,
                "thumbnail": output_base64,
                "enhanced_image_with_prefix": f"data:image/png;base64,{output_base64}",
                "size": list(enhanced_image.size),
                "version": VERSION,
                "status": "success",
                "format": "PNG",
                "mode": "RGBA",
                "pattern_type": pattern_type,
                "detected_type": detected_type,
                "intensity": intensity,
                "cubic_statistics": {
                    "cubic_pixels": int(cubic_pixel_count),
                    "cubic_percentage": round(cubic_percentage, 2),
                    "has_cubics": cubic_pixel_count > 0
                },
                "corrections_applied": [
                    "white_balance",
                    "pattern_enhancement" if apply_pattern else "pattern_skipped",
                    "swinir" if apply_swinir else "swinir_skipped",
                    "ring_hole_detection",
                    "cubic_detail_enhancement"
                ],
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "compatible_with": ["enhancement_handler", "thumbnail_handler"],
                "input_accepted": ["enhanced_image", "thumbnail", "image"],
                "processing_order": "1.WB → 2.Pattern → 3.SwinIR → 4.RingHoles → 5.CubicDetail"
            }
        }
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
