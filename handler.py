import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import logging
import string
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# CUBIC DETAIL ENHANCEMENT HANDLER V18-RING-REFINED
# VERSION: Cubic-Sparkle-V18-C099-B117-WO10-RingRefined
# Updated: Refined ring hole detection for better quality
################################

VERSION = "Cubic-Sparkle-V18-C099-B117-WO10-RingRefined"

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
    """Convert image to base64 with padding - ALWAYS include padding"""
    buffered = BytesIO()
    
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA for transparency")
        image = image.convert('RGBA')
    
    logger.info("💎 Saving RGBA image as PNG with compression level 3")
    image.save(buffered, format='PNG', compress_level=3, optimize=True)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str

def find_input_data_robust(data, path="", depth=0, max_depth=10):
    """More robust input data extraction with detailed logging"""
    if depth > max_depth:
        return None
    
    logger.info(f"🔍 Searching at depth {depth}, path: {path}")
    
    if isinstance(data, str):
        str_len = len(data)
        logger.info(f"  Found string at {path} with length {str_len}")
        
        if str_len > 20:
            sample = data[:100].strip()
            if sample and all(c in string.ascii_letters + string.digits + '+/=' for c in sample[:50]):
                logger.info(f"✅ Found potential base64 data at {path} (length: {str_len})")
                return data
        elif str_len > 0:
            logger.info(f"  String too short at {path}: {str_len} chars")
    
    if isinstance(data, dict):
        logger.info(f"  Dict at {path} with keys: {list(data.keys())}")
        
        base64_candidates = []
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if key in ['enhanced_image', 'image', 'base64', 'image_base64'] and isinstance(value, str) and len(value) > 1000:
                continue
                
            if isinstance(value, str) and len(value) > 20:
                logger.info(f"  Checking string at {current_path}: length={len(value)}")
                
                sample = value[:100].strip()
                if sample and all(c in string.ascii_letters + string.digits + '+/=' for c in sample[:50]):
                    base64_candidates.append((current_path, value, len(value)))
            
            if isinstance(value, (dict, list)):
                result = find_input_data_robust(value, current_path, depth + 1, max_depth)
                if result:
                    return result
        
        if base64_candidates:
            base64_candidates.sort(key=lambda x: x[2], reverse=True)
            best_path, best_value, best_len = base64_candidates[0]
            logger.info(f"✅ Selected best base64 candidate at {best_path} (length: {best_len})")
            return best_value
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]"
            result = find_input_data_robust(item, current_path, depth + 1, max_depth)
            if result:
                return result
    
    return None

def detect_pattern_type(filename: str, default_pattern: str = "ab_pattern") -> str:
    """Detect pattern type - AC, AB, or other
    Now with relaxed detection (no underscore required) and default AB pattern
    """
    if not filename:
        logger.info(f"⚠️ No filename provided, using default pattern: {default_pattern}")
        return default_pattern
    
    filename_lower = filename.lower()
    logger.info(f"🔍 Checking pattern for filename: {filename}")
    
    if 'ac' in filename_lower:
        logger.info("✅ Detected AC pattern (무도금화이트)")
        return "ac_pattern"
    elif 'ab' in filename_lower:
        logger.info("✅ Detected AB pattern (무도금화이트-쿨톤)")
        return "ab_pattern"
    else:
        logger.info("✅ Detected other pattern (기타색상)")
        return "other"

def auto_white_balance_fast(image: Image.Image) -> Image.Image:
    """Fast white balance - preserving transparency"""
    if image.mode != 'RGBA':
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
    
    return result

def enhance_cubic_detail_for_pattern(image: Image.Image, pattern_type: str) -> Image.Image:
    """REFINED cubic detail for AC and AB patterns - REDUCED to prevent artifacts"""
    if pattern_type not in ["ac_pattern", "ab_pattern"]:
        return image
    
    logger.info(f"💎 Refined cubic detail enhancement for {pattern_type}")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    edges = rgb_image.filter(ImageFilter.EDGE_ENHANCE)
    unsharp = rgb_image.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
    
    img_array = np.array(rgb_image, dtype=np.float32)
    edges_array = np.array(edges, dtype=np.float32)
    unsharp_array = np.array(unsharp, dtype=np.float32)
    
    v_channel = np.max(img_array, axis=2)
    bright_mask = v_channel > 235
    very_bright_mask = v_channel > 245
    
    if pattern_type == "ac_pattern":
        edge_blend = 0.25
        unsharp_blend = 0.15
    else:  # ab_pattern
        edge_blend = 0.2
        unsharp_blend = 0.12
    
    for c in range(3):
        img_array[:,:,c] = np.where(
            bright_mask,
            img_array[:,:,c] * (1 - edge_blend) + edges_array[:,:,c] * edge_blend,
            img_array[:,:,c]
        )
        
        img_array[:,:,c] = np.where(
            very_bright_mask,
            img_array[:,:,c] * (1 - unsharp_blend) + unsharp_array[:,:,c] * unsharp_blend,
            img_array[:,:,c]
        )
    
    if pattern_type == "ac_pattern":
        highlight_boost = 1.08
    else:  # ab_pattern
        highlight_boost = 1.06
    
    highlight_mask = v_channel > 245
    img_array[highlight_mask] = np.minimum(img_array[highlight_mask] * highlight_boost, 255)
    
    rgb_enhanced = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    sharpness = ImageEnhance.Sharpness(rgb_enhanced)
    rgb_enhanced = sharpness.enhance(1.2)
    
    r2, g2, b2 = rgb_enhanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    logger.info("✅ Refined cubic detail enhancement applied")
    
    return result

def apply_pattern_enhancement_transparent(image: Image.Image, pattern_type: str) -> Image.Image:
    """Apply pattern enhancement while preserving transparency - V18 with Contrast 0.99, Brightness 1.17, White Overlay 10%"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    img_array = np.array(rgb_image, dtype=np.float32)
    
    if pattern_type == "ac_pattern":
        logger.info("🔍 AC Pattern (무도금화이트) - Applying 10% white overlay with brightness 1.17, contrast 0.99")
        white_overlay = 0.10
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        rgb_image = Image.merge('RGB', rgb_image.split())
        temp_rgba = Image.merge('RGBA', (*rgb_image.split(), a))
        temp_rgba = enhance_cubic_detail_for_pattern(temp_rgba, pattern_type)
        r_temp, g_temp, b_temp, _ = temp_rgba.split()
        rgb_image = Image.merge('RGB', (r_temp, g_temp, b_temp))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.17)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(0.99)
        
    elif pattern_type == "ab_pattern":
        logger.info("🔍 AB Pattern (무도금화이트-쿨톤) - Applying 10% white overlay with brightness 1.17, contrast 0.99")
        white_overlay = 0.10
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        
        img_array[:,:,0] *= 0.96
        img_array[:,:,1] *= 0.98
        img_array[:,:,2] *= 1.02
        
        cool_overlay = np.array([240, 248, 255], dtype=np.float32)
        img_array = img_array * 0.95 + cool_overlay * 0.05
        
        img_array = np.clip(img_array, 0, 255)
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        temp_rgba = Image.merge('RGBA', (*rgb_image.split(), a))
        temp_rgba = enhance_cubic_detail_for_pattern(temp_rgba, pattern_type)
        r_temp, g_temp, b_temp, _ = temp_rgba.split()
        rgb_image = Image.merge('RGB', (r_temp, g_temp, b_temp))
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.88)
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.17)
        
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(0.99)
        
    else:
        logger.info("🔍 Other Pattern (기타색상) - Applying 2% white overlay with brightness 1.05, contrast 1.06")
        white_overlay = 0.02
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.05)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.99)
        
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(1.06)
        
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.2)
    
    sharpness = ImageEnhance.Sharpness(rgb_image)
    if pattern_type in ["ac_pattern", "ab_pattern"]:
        rgb_image = sharpness.enhance(1.4)
    else:
        rgb_image = sharpness.enhance(1.3)
    
    r2, g2, b2 = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
    
    return enhanced_image

def apply_swinir_enhancement(image: Image.Image) -> Image.Image:
    """Apply SwinIR enhancement - delayed loading"""
    try:
        logger.info("🎨 Applying SwinIR enhancement for cubic detail")
        
        import replicate
        
        api_token = os.environ.get('REPLICATE_API_TOKEN')
        if not api_token:
            logger.warning("No Replicate API token")
            return image
        
        client = replicate.Client(api_token=api_token)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        buffered = BytesIO()
        rgb_image.save(buffered, format="PNG", optimize=True, compress_level=3)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        output = client.run(
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
            
            logger.info("✅ SwinIR enhancement successful")
            return result
            
    except Exception as e:
        logger.warning(f"SwinIR error: {str(e)}")
        
    return image

def ensure_ring_holes_transparent_refined(image: Image.Image) -> Image.Image:
    """Refined ring hole detection - more precise and quality-preserving"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 Refined ring hole detection for jewelry")
    
    r, g, b, a = image.split()
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    alpha_array = np.array(a, dtype=np.uint8)
    
    # Convert to grayscale for analysis
    gray = np.mean(rgb_array, axis=2)
    
    # Method 1: Detect specific gray background colors (narrower range)
    # More conservative range to avoid false positives
    gray_background_mask = (
        (gray > 190) & (gray < 210) &  # Narrower gray value range
        (np.abs(rgb_array[:,:,0] - rgb_array[:,:,1]) < 10) &  # Stricter R≈G
        (np.abs(rgb_array[:,:,1] - rgb_array[:,:,2]) < 10) &  # Stricter G≈B
        (alpha_array > 250)  # Only fully opaque areas
    )
    
    # Method 2: HSV-based detection for very bright, desaturated areas
    hsv = image.convert('HSV')
    h, s, v = hsv.split()
    s_array = np.array(s)
    v_array = np.array(v)
    
    # More conservative detection
    bright_desaturated_mask = (
        (v_array > 245) &  # Very bright only
        (s_array < 20) &   # Very low saturation only
        (alpha_array > 250)  # Fully opaque only
    )
    
    # Combine detection methods
    potential_holes = gray_background_mask | bright_desaturated_mask
    
    # Apply morphological operations to clean up - more conservative
    holes_image = Image.fromarray((potential_holes * 255).astype(np.uint8))
    
    # Smaller operations to preserve quality
    holes_image = holes_image.filter(ImageFilter.MinFilter(1))  # Minimal erosion
    holes_image = holes_image.filter(ImageFilter.MaxFilter(1))  # Minimal dilation
    
    holes_mask = np.array(holes_image) > 128
    
    # Special handling for ring centers - more precise
    h, w = alpha_array.shape
    center_y, center_x = h // 2, w // 2
    
    # Check if image center region should be transparent
    # Smaller region for more precise detection
    center_region_size = min(h, w) // 8  # Smaller than before
    center_region = gray[
        max(0, center_y - center_region_size):min(h, center_y + center_region_size),
        max(0, center_x - center_region_size):min(w, center_x + center_region_size)
    ]
    
    if center_region.size > 0:
        center_mean = np.mean(center_region)
        center_std = np.std(center_region)
        
        # More conservative: only if very uniform and grayish
        if 195 < center_mean < 205 and center_std < 5:  # Tighter range and uniformity check
            y_indices, x_indices = np.ogrid[:h, :w]
            center_mask = ((x_indices - center_x)**2 + (y_indices - center_y)**2) < center_region_size**2
            holes_mask = holes_mask | center_mask
    
    # Apply the holes mask to alpha channel with some edge smoothing
    # Create a smoother transition
    from scipy.ndimage import gaussian_filter
    try:
        # Smooth the mask edges slightly for better quality
        holes_mask_smooth = gaussian_filter(holes_mask.astype(float), sigma=0.5) > 0.5
        alpha_array[holes_mask_smooth] = 0
    except ImportError:
        # Fallback without scipy
        alpha_array[holes_mask] = 0
    
    # Final cleanup - remove very small isolated transparent areas
    # This helps preserve quality by avoiding scattered transparent pixels
    a_new = Image.fromarray(alpha_array)
    
    # Optional: Apply a very light median filter to clean up noise
    # Only if the mask has significant changes
    if np.sum(holes_mask) > 100:  # Only if we actually made changes
        a_new = a_new.filter(ImageFilter.MedianFilter(1))
    
    result = Image.merge('RGBA', (r, g, b, a_new))
    
    logger.info(f"✅ Refined ring holes applied - {np.sum(holes_mask)} pixels made transparent")
    return result

def detect_cubic_regions_enhanced(image: Image.Image, sensitivity=1.0):
    """Enhanced cubic detection with STRICTER thresholds to prevent over-detection"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    alpha_array = np.array(image.split()[3], dtype=np.uint8)
    
    hsv = image.convert('HSV')
    h, s, v = hsv.split()
    h_array = np.array(h)
    s_array = np.array(s)
    v_array = np.array(v)
    
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges.convert('L'))
    
    white_cubic = (
        (v_array > 245 * sensitivity) &
        (s_array < 25) &
        (alpha_array > 200)
    )
    
    color_cubic = (
        (v_array > 210 * sensitivity) &
        (s_array > 120) &
        (alpha_array > 200)
    )
    
    edge_cubic = (edges_array > 120) & (v_array > 230) & (alpha_array > 200)
    
    highlights = (
        (v_array > 250) &
        (s_array < 40) &
        (alpha_array > 200)
    )
    
    cubic_mask = white_cubic | color_cubic | edge_cubic | highlights
    
    cubic_image = Image.fromarray((cubic_mask * 255).astype(np.uint8))
    cubic_image = cubic_image.filter(ImageFilter.MinFilter(3))
    cubic_image = cubic_image.filter(ImageFilter.MaxFilter(3))
    cubic_mask = np.array(cubic_image) > 128
    
    return cubic_mask.astype(bool), white_cubic, color_cubic, highlights

def enhance_cubic_sparkle_with_swinir(image: Image.Image, intensity=1.0) -> Image.Image:
    """REFINED cubic sparkle for SwinIR - BALANCED approach"""
    logger.info("💎 Refined cubic detail processing for SwinIR...")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
    
    cubic_mask, white_cubic, color_cubic, highlights = detect_cubic_regions_enhanced(image, intensity)
    
    cubic_count = np.sum(cubic_mask)
    logger.info(f"✨ Detected {cubic_count} cubic pixels")
    
    if cubic_count == 0:
        logger.info("No cubic regions detected, skipping enhancement")
        return image
    
    edges = image.filter(ImageFilter.EDGE_ENHANCE)
    edges_array = np.array(edges.convert('RGB'), dtype=np.float32)
    
    unsharp = image.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
    unsharp_array = np.array(unsharp.convert('RGB'), dtype=np.float32)
    
    for c in range(3):
        rgb_array[:,:,c] = np.where(
            cubic_mask,
            rgb_array[:,:,c] * 0.75 + edges_array[:,:,c] * 0.25,
            rgb_array[:,:,c]
        )
        
        rgb_array[:,:,c] = np.where(
            highlights,
            rgb_array[:,:,c] * 0.85 + unsharp_array[:,:,c] * 0.15,
            rgb_array[:,:,c]
        )
    
    if np.any(cubic_mask):
        mean_val = np.mean(rgb_array[cubic_mask])
        contrast_factor = 1.15 * intensity
        
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                np.clip((rgb_array[:,:,c] - mean_val) * contrast_factor + mean_val, 0, 255),
                rgb_array[:,:,c]
            )
    
    if np.any(highlights):
        boost_factor = 1.1 * intensity
        rgb_array[highlights] = np.minimum(rgb_array[highlights] * boost_factor, 255)
    
    if np.any(color_cubic):
        rgb_temp = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
        hsv_temp = rgb_temp.convert('HSV')
        h_temp, s_temp, v_temp = hsv_temp.split()
        
        s_array = np.array(s_temp, dtype=np.float32)
        v_array = np.array(v_temp, dtype=np.float32)
        
        s_array = np.where(
            color_cubic,
            np.minimum(s_array * (1.3 * intensity), 255),
            s_array
        )
        
        v_array = np.where(
            color_cubic,
            np.minimum(v_array * 1.03, 255),
            v_array
        )
        
        hsv_enhanced = Image.merge('HSV', (
            h_temp,
            Image.fromarray(s_array.astype(np.uint8)),
            Image.fromarray(v_array.astype(np.uint8))
        ))
        rgb_array = np.array(hsv_enhanced.convert('RGB'), dtype=np.float32)
    
    rgb_enhanced = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
    r2, g2, b2 = rgb_enhanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    sharpness = ImageEnhance.Sharpness(result)
    result = sharpness.enhance(1.0 + (0.2 * intensity))
    
    logger.info("✅ Refined cubic pre-enhancement complete")
    
    return result

def handler(event):
    """RunPod handler function - V18 with Refined Ring Detection"""
    logger.info(f"=== Cubic Detail Enhancement {VERSION} Started ===")
    logger.info(f"Handler received event type: {type(event)}")
    
    logger.info("📋 Complete event structure:")
    logger.info(json.dumps(event, indent=2)[:1000] + "...")
    
    try:
        job_input = None
        
        if isinstance(event, dict):
            if 'input' in event:
                job_input = event['input']
                logger.info("Found input in event['input']")
            else:
                job_input = event
                logger.info("Using event directly as input")
        else:
            job_input = event
            logger.info("Event is not a dict, using as is")
        
        return process_cubic_enhancement(job_input)
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        import traceback
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION,
                "traceback": traceback.format_exc()
            }
        }

def process_cubic_enhancement(job):
    """Process cubic detail enhancement with refined ring hole detection"""
    try:
        logger.info("🚀 Fast loading version with Refined Ring Detection")
        logger.info("💎 SwinIR for refined detail enhancement")
        logger.info(f"Job input type: {type(job)}")
        
        if isinstance(job, dict):
            logger.info(f"Job keys: {list(job.keys())[:10]}")
            for key, value in job.items():
                if isinstance(value, str):
                    logger.info(f"  {key}: string, length={len(value)}")
                    if len(value) < 100:
                        logger.info(f"    Value: {value}")
                elif isinstance(value, dict):
                    logger.info(f"  {key}: dict, keys={list(value.keys())}")
                elif isinstance(value, list):
                    logger.info(f"  {key}: list, length={len(value)}")
                else:
                    logger.info(f"  {key}: {type(value)}")
        
        image_data_str = find_input_data_robust(job)
        
        if not image_data_str:
            error_details = []
            error_details.append("No valid image data found in input.")
            
            if isinstance(job, dict):
                error_details.append(f"Available keys: {list(job.keys())}")
                
                for key, value in job.items():
                    if isinstance(value, str):
                        if len(value) == 0:
                            error_details.append(f"- '{key}' is empty string")
                        elif len(value) < 20:
                            error_details.append(f"- '{key}' too short ({len(value)} chars)")
                        else:
                            sample = value[:50].strip()
                            if not all(c in string.ascii_letters + string.digits + '+/=' for c in sample):
                                error_details.append(f"- '{key}' doesn't look like base64")
            
            error_details.append("\nExpected fields: 'enhancement', 'enhanced_image', 'image', 'base64', etc.")
            error_details.append("Image data should be base64 encoded string.")
            error_details.append("\nMake.com note: Check that the field mapping is correct.")
            error_details.append("The image field might be: {{4.data.output.output.enhanced_image}}")
            
            raise ValueError("\n".join(error_details))
        
        params = job if isinstance(job, dict) else {}
        filename = params.get('filename', '')
        intensity = float(params.get('intensity', 1.0))
        intensity = max(0.1, min(2.0, intensity))
        apply_swinir = params.get('apply_swinir', True)
        apply_pattern = params.get('pattern_enhancement', True)
        
        default_pattern = params.get('default_pattern', 'ab_pattern')
        
        logger.info(f"Parameters: filename={filename or 'None (using default)'}, intensity={intensity}, swinir={apply_swinir}, pattern={apply_pattern}")
        
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
        
        # 2. Pattern Enhancement with default support
        if apply_pattern:
            logger.info("🎨 Step 2: Applying pattern enhancement with refined cubic detail")
            pattern_type = detect_pattern_type(filename, default_pattern=default_pattern)
            detected_type = {
                "ac_pattern": "무도금화이트(0.10)",
                "ab_pattern": "무도금화이트-쿨톤(0.10)",
                "other": "기타색상(0.02)"
            }.get(pattern_type, "기타색상")
            
            logger.info(f"Detected pattern: {pattern_type} - {detected_type}")
            image = apply_pattern_enhancement_transparent(image, pattern_type)
        else:
            pattern_type = "none"
            detected_type = "보정없음"
        
        # 3. Refined Ring Hole Detection
        logger.info("🔍 Step 3: Refined ring hole detection for jewelry")
        image = ensure_ring_holes_transparent_refined(image)
        
        # 4. Refined Cubic Pre-enhancement
        logger.info("💎 Step 4: Refined cubic pre-enhancement")
        image = enhance_cubic_sparkle_with_swinir(image, intensity)
        
        # 5. SwinIR Enhancement
        if apply_swinir:
            logger.info("🚀 Step 5: Applying SwinIR for refined detail")
            enhanced_image = apply_swinir_enhancement(image)
        else:
            enhanced_image = image
        
        output_base64 = image_to_base64(enhanced_image)
        
        cubic_mask, _, _, _ = detect_cubic_regions_enhanced(image)
        cubic_pixel_count = int(np.sum(cubic_mask))
        cubic_percentage = (cubic_pixel_count / (image.size[0] * image.size[1])) * 100
        
        has_cubics = bool(cubic_pixel_count > 0)
        
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
                "filename_received": filename if filename else "None (using default)",
                "cubic_statistics": {
                    "cubic_pixels": cubic_pixel_count,
                    "cubic_percentage": round(cubic_percentage, 2),
                    "has_cubics": has_cubics
                },
                "corrections_applied": [
                    "white_balance",
                    "pattern_enhancement" if apply_pattern else "pattern_skipped",
                    "ring_hole_detection_refined",
                    "cubic_pre_enhancement_refined",
                    "swinir_detail" if apply_swinir else "swinir_skipped"
                ],
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "performance": "optimized_ring_detection_refined",
                "processing_order": "1.WB → 2.Pattern → 3.RingHoles(Refined) → 4.RefinedCubicPrep → 5.SwinIR",
                "v18_refined_improvements": [
                    "More conservative gray detection (190-210 range)",
                    "Stricter color uniformity checks",
                    "Smaller morphological operations",
                    "Added uniformity check for center region",
                    "Optional gaussian smoothing for edges",
                    "Median filter for noise reduction",
                    "Preserved image quality while detecting holes"
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in process_cubic_enhancement: {str(e)}")
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
