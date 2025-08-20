import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import logging
import string
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# CUBIC DETAIL ENHANCEMENT HANDLER V29-BALANCED
# VERSION: Cubic-Sparkle-V29-Balanced-Other-Color
# Updated: Balanced OTHER pattern for natural metallic colors
################################

VERSION = "Cubic-Sparkle-V29-Balanced-Other-Color"

def decode_base64_fast(base64_str: str) -> bytes:
    """Fast base64 decode with padding handling"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string - too short or empty")
        
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        # Remove whitespace
        base64_str = ''.join(base64_str.split())
        
        # Filter to valid base64 characters only
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        # Try decoding with current padding
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            return decoded
        except Exception:
            # Add proper padding if needed
            no_pad = base64_str.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        raise ValueError(f"Invalid base64 data: {str(e)}")

def image_to_base64(image):
    """Convert image to base64 with padding - ALWAYS include padding for Google Script/Make.com"""
    buffered = BytesIO()
    
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA for transparency")
        image = image.convert('RGBA')
    
    logger.info("💎 Saving RGBA image as PNG with compression level 3")
    image.save(buffered, format='PNG', compress_level=3, optimize=True)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # Always return WITH padding
    return base64_str

def find_input_data_robust(data, path="", depth=0, max_depth=10):
    """More robust input data extraction with detailed logging - FIXED for thumbnail"""
    if depth > max_depth:
        return None
    
    logger.info(f"🔍 Searching at depth {depth}, path: {path}")
    
    if isinstance(data, str):
        str_len = len(data)
        logger.info(f"  Found string at {path} with length {str_len}")
        
        if str_len > 100:  # Increased minimum for valid base64
            sample = data[:100].strip()
            if sample and all(c in string.ascii_letters + string.digits + '+/=' for c in sample[:50]):
                logger.info(f"✅ Found potential base64 data at {path} (length: {str_len})")
                return data
        elif str_len > 0:
            logger.info(f"  String too short at {path}: {str_len} chars")
    
    if isinstance(data, dict):
        logger.info(f"  Dict at {path} with keys: {list(data.keys())}")
        
        base64_candidates = []
        
        # Check if this is a thumbnail-only request
        is_thumbnail_request = 'thumbnail' in data and 'enhanced_image' not in data and 'image' not in data
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # FIXED: Don't skip thumbnail if it's the main image data
            if key == 'enhanced_image' and isinstance(value, str) and len(value) > 1000:
                # Skip enhanced_image to avoid loops
                continue
            elif key == 'thumbnail' and isinstance(value, str) and len(value) > 1000:
                # Only skip thumbnail if we have other image sources
                if not is_thumbnail_request:
                    logger.info(f"  Found thumbnail but looking for other sources first")
                    # Don't skip, but give it lower priority
                else:
                    logger.info(f"  This is a thumbnail request - will process thumbnail")
                
            if isinstance(value, str) and len(value) > 100:
                logger.info(f"  Checking string at {current_path}: length={len(value)}")
                
                sample = value[:100].strip()
                if sample and all(c in string.ascii_letters + string.digits + '+/=' for c in sample[:50]):
                    # Prioritize non-thumbnail keys unless it's a thumbnail-only request
                    priority = 0 if key == 'thumbnail' and not is_thumbnail_request else 1
                    base64_candidates.append((current_path, value, len(value), priority))
            
            if isinstance(value, (dict, list)):
                result = find_input_data_robust(value, current_path, depth + 1, max_depth)
                if result:
                    return result
        
        if base64_candidates:
            # Sort by priority first, then by length (longest first)
            base64_candidates.sort(key=lambda x: (x[3], x[2]), reverse=True)
            best_path, best_value, best_len, _ = base64_candidates[0]
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
    """Detect pattern type - AC, AB, or other"""
    if not filename:
        logger.info(f"⚠️ No filename provided, using default pattern: {default_pattern}")
        return default_pattern
    
    filename_lower = filename.lower()
    logger.info(f"🔍 Checking pattern for filename: {filename}")
    
    if 'ac_' in filename_lower or '_ac' in filename_lower:
        logger.info("✅ Detected AC pattern (무도금화이트)")
        return "ac_pattern"
    elif 'ab_' in filename_lower or '_ab' in filename_lower:
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
    
    # Sample image for performance
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

def gradual_cubic_detail_pass(image: Image.Image, pattern_type: str, pass_num: int) -> Image.Image:
    """Enhanced cubic detail - INCREASED values for more detail"""
    if pattern_type not in ["ac_pattern", "ab_pattern"]:
        return image
    
    logger.info(f"  💠 Pass {pass_num}: Enhanced cubic detail")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    edges = rgb_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    unsharp = rgb_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    img_array = np.array(rgb_image, dtype=np.float32)
    edges_array = np.array(edges, dtype=np.float32)
    unsharp_array = np.array(unsharp, dtype=np.float32)
    
    v_channel = np.max(img_array, axis=2)
    bright_mask = v_channel > 230
    very_bright_mask = v_channel > 245
    
    if pattern_type == "ac_pattern":
        edge_blend = 0.25 * (1.0 - pass_num * 0.05)
        unsharp_blend = 0.15 * (1.0 - pass_num * 0.05)
    else:  # ab_pattern
        edge_blend = 0.20 * (1.0 - pass_num * 0.05)
        unsharp_blend = 0.12 * (1.0 - pass_num * 0.05)
    
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
        highlight_boost = 1.08 + (0.02 * (3 - pass_num))
    else:
        highlight_boost = 1.06 + (0.015 * (3 - pass_num))
    
    highlight_mask = v_channel > 245
    img_array[highlight_mask] = np.minimum(img_array[highlight_mask] * highlight_boost, 255)
    
    rgb_enhanced = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    
    sharpness = ImageEnhance.Sharpness(rgb_enhanced)
    rgb_enhanced = sharpness.enhance(1.0 + (0.15 * (1.0 - pass_num * 0.05)))
    
    r2, g2, b2 = rgb_enhanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    return result

def apply_pattern_enhancement_gradual(image: Image.Image, pattern_type: str) -> Image.Image:
    """Apply pattern enhancement with BALANCED OTHER pattern for natural colors"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    img_array = np.array(rgb_image, dtype=np.float32)
    
    if pattern_type == "ac_pattern":
        logger.info("🔍 AC Pattern (무도금화이트) - Enhanced detail enhancement")
        
        logger.info("  Pass 1/4: Base white overlay (10%)")
        white_overlay = 0.10
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        temp_rgba = Image.merge('RGBA', (*rgb_image.split(), a))
        
        for pass_num in range(1, 4):
            logger.info(f"  Pass {pass_num+1}/4: Enhanced cubic detail")
            temp_rgba = gradual_cubic_detail_pass(temp_rgba, pattern_type, pass_num)
        
        r_temp, g_temp, b_temp, _ = temp_rgba.split()
        rgb_image = Image.merge('RGB', (r_temp, g_temp, b_temp))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.17)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(1.05)
        
    elif pattern_type == "ab_pattern":
        logger.info("🔍 AB Pattern (무도금화이트-쿨톤) - Enhanced detail enhancement")
        
        logger.info("  Pass 1/4: Base white overlay and cool tone")
        white_overlay = 0.10
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        
        # Cool tone adjustment
        img_array[:,:,0] *= 0.96  # Reduce red
        img_array[:,:,1] *= 0.98  # Slightly reduce green
        img_array[:,:,2] *= 1.02  # Increase blue
        
        # Cool overlay
        cool_overlay = np.array([240, 248, 255], dtype=np.float32)
        img_array = img_array * 0.95 + cool_overlay * 0.05
        
        img_array = np.clip(img_array, 0, 255)
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        temp_rgba = Image.merge('RGBA', (*rgb_image.split(), a))
        
        for pass_num in range(1, 4):
            logger.info(f"  Pass {pass_num+1}/4: Enhanced cubic detail")
            temp_rgba = gradual_cubic_detail_pass(temp_rgba, pattern_type, pass_num)
        
        r_temp, g_temp, b_temp, _ = temp_rgba.split()
        rgb_image = Image.merge('RGB', (r_temp, g_temp, b_temp))
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.88)
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.17)
        
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(1.05)
        
    else:  # other pattern - BALANCED for natural metallic colors
        logger.info("🔍 Other Pattern (기타색상) - Balanced Natural Enhancement")
        
        # NO WHITE OVERLAY - Keep original color tone
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        # MODERATE COLOR SATURATION - 적절한 색상 포화도
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(1.35)  # 2.20 -> 1.35로 대폭 감소 (자연스러운 색상)
        logger.info("  ✅ Applied moderate color saturation 1.35 for natural metallic colors")
        
        # BALANCED BRIGHTNESS - 균형잡힌 밝기
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(0.95)  # 0.72 -> 0.95로 증가 (원본과 유사한 밝기)
        logger.info("  ✅ Applied balanced brightness 0.95 for natural tones")
        
        # MODERATE CONTRAST - 적절한 대비
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(1.20)  # 1.55 -> 1.20으로 감소 (자연스러운 대비)
        logger.info("  ✅ Applied moderate contrast 1.20 for balanced depth")
        
        # GENTLE HSV COLOR BOOST - HSV 색상 공간에서 부드러운 조정
        hsv_image = rgb_image.convert('HSV')
        h, s, v = hsv_image.split()
        
        # Saturation 부드러운 부스트
        s_array = np.array(s, dtype=np.float32)
        s_array = np.minimum(s_array * 1.15, 255)  # 1.45 -> 1.15로 감소
        s = Image.fromarray(s_array.astype(np.uint8))
        
        # Value(명도) 미세 조정
        v_array = np.array(v, dtype=np.float32)
        v_array = v_array * 0.98  # 0.88 -> 0.98로 증가 (거의 원본 유지)
        v = Image.fromarray(v_array.astype(np.uint8))
        
        hsv_enhanced = Image.merge('HSV', (h, s, v))
        rgb_image = hsv_enhanced.convert('RGB')
        logger.info("  ✅ Applied gentle HSV boost - Saturation x1.15, Value x0.98")
        
        # Additional color channel manipulation for metallic tones
        rgb_array = np.array(rgb_image, dtype=np.float32)
        
        # Gentle boost for warm metallic colors (yellow/rose gold)
        warm_mask = (rgb_array[:,:,0] > rgb_array[:,:,2]) | (rgb_array[:,:,1] > rgb_array[:,:,2])
        if np.any(warm_mask):
            # Subtle enhancement for gold tones
            rgb_array[:,:,0] = np.where(warm_mask, np.minimum(rgb_array[:,:,0] * 1.05, 255), rgb_array[:,:,0])
            rgb_array[:,:,1] = np.where(warm_mask, np.minimum(rgb_array[:,:,1] * 1.03, 255), rgb_array[:,:,1])
            logger.info("  ✅ Applied subtle warm color boost for metallic tones")
        
        rgb_image = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
        
        # MODERATE SHARPNESS for clear but natural details
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.25)  # 1.45 -> 1.25로 감소
        logger.info("  ✅ Applied moderate sharpness 1.25 for natural clarity")
        
        # SUBTLE EDGE ENHANCEMENT for OTHER pattern
        edges = rgb_image.filter(ImageFilter.EDGE_ENHANCE)  # EDGE_ENHANCE_MORE -> EDGE_ENHANCE로 변경
        rgb_array = np.array(rgb_image, dtype=np.float32)
        edges_array = np.array(edges, dtype=np.float32)
        
        # 부드러운 엣지 블렌딩
        edge_blend = 0.08  # 0.20 -> 0.08로 대폭 감소
        for c in range(3):
            rgb_array[:,:,c] = rgb_array[:,:,c] * (1 - edge_blend) + edges_array[:,:,c] * edge_blend
        
        rgb_image = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
        logger.info("  ✅ Applied subtle edge enhancement blend 8% for smooth details")
        
        # Skip final color boost - already balanced
        logger.info("  ✅ Skipped final color boost - already well balanced")
    
    # Final sharpness adjustment
    sharpness = ImageEnhance.Sharpness(rgb_image)
    if pattern_type in ["ac_pattern", "ab_pattern"]:
        rgb_image = sharpness.enhance(1.4)
    else:
        rgb_image = sharpness.enhance(1.10)  # 1.20 -> 1.10 for OTHER (gentler)
    
    r2, g2, b2 = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
    
    return enhanced_image

def simple_morphology_operation(mask, operation='close', iterations=1):
    """Simple morphology operations using PIL only - NO scipy"""
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    
    for _ in range(iterations):
        if operation == 'close':
            # Dilation then erosion
            mask_image = mask_image.filter(ImageFilter.MaxFilter(3))
            mask_image = mask_image.filter(ImageFilter.MinFilter(3))
        elif operation == 'open':
            # Erosion then dilation
            mask_image = mask_image.filter(ImageFilter.MinFilter(3))
            mask_image = mask_image.filter(ImageFilter.MaxFilter(3))
        elif operation == 'dilate':
            mask_image = mask_image.filter(ImageFilter.MaxFilter(3))
        elif operation == 'erode':
            mask_image = mask_image.filter(ImageFilter.MinFilter(3))
    
    return np.array(mask_image) > 128

def check_pixel_continuity(gray, x, y, window=5):
    """Check if a pixel is part of a continuous uniform region"""
    h, w = gray.shape
    
    # Get window bounds
    x_min = max(0, x - window)
    x_max = min(w, x + window + 1)
    y_min = max(0, y - window)
    y_max = min(h, y + window + 1)
    
    # Extract window
    window_pixels = gray[y_min:y_max, x_min:x_max]
    
    if window_pixels.size == 0:
        return False
    
    # Check uniformity
    std_dev = np.std(window_pixels)
    mean_val = np.mean(window_pixels)
    
    # Uniform if low standard deviation and similar to center pixel
    is_uniform = (std_dev < 3) and (abs(gray[y, x] - mean_val) < 2)
    
    return is_uniform

def verify_morphological_pattern(mask, x, y, pattern_size=7):
    """Verify if the region forms a circular/elliptical pattern (likely a hole)"""
    h, w = mask.shape
    
    # Get region bounds
    half_size = pattern_size // 2
    x_min = max(0, x - half_size)
    x_max = min(w, x + half_size + 1)
    y_min = max(0, y - half_size)
    y_max = min(h, y + half_size + 1)
    
    # Extract region
    region = mask[y_min:y_max, x_min:x_max]
    
    if region.size < 9:  # Too small to analyze
        return False
    
    # Count true pixels
    true_count = np.sum(region)
    total_count = region.size
    
    # A hole should have most pixels marked as background
    fill_ratio = true_count / total_count if total_count > 0 else 0
    
    # Holes typically have fill ratio > 0.6 (mostly filled)
    return fill_ratio > 0.6

def ensure_ring_holes_transparent_advanced(image: Image.Image) -> Image.Image:
    """Advanced ring hole detection with multi-stage verification and highlight preservation"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 Advanced ring hole detection with highlight preservation")
    
    r, g, b, a = image.split()
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    alpha_array = np.array(a, dtype=np.uint8)
    
    # Convert to grayscale
    gray = np.mean(rgb_array, axis=2).astype(np.uint8)
    
    # Get image dimensions
    h, w = alpha_array.shape
    center_y, center_x = h // 2, w // 2
    
    # Create radial distance map
    y_indices, x_indices = np.meshgrid(range(h), range(w), indexing='ij')
    distance_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    # Define ring regions with differentiated processing
    max_radius = min(h, w) // 2
    inner_ring_radius = max_radius * 0.35  # Inner 35% - very conservative
    middle_ring_radius = max_radius * 0.60  # Middle ring
    outer_ring_radius = max_radius * 0.85  # Outer ring
    
    # Create region masks
    inner_ring_mask = distance_from_center < inner_ring_radius
    middle_ring_mask = (distance_from_center >= inner_ring_radius) & (distance_from_center < middle_ring_radius)
    outer_ring_mask = (distance_from_center >= middle_ring_radius) & (distance_from_center <= outer_ring_radius)
    beyond_ring_mask = distance_from_center > outer_ring_radius
    
    logger.info(f"  Region distribution - Inner: {np.sum(inner_ring_mask)}, Middle: {np.sum(middle_ring_mask)}, Outer: {np.sum(outer_ring_mask)}")
    
    # Stage 1: Color-based detection with region-specific thresholds
    background_candidates = np.zeros_like(gray, dtype=bool)
    
    # Define exact background color (RGB 200,200,200)
    exact_background = (
        (rgb_array[:,:,0] >= 198) & (rgb_array[:,:,0] <= 202) &
        (rgb_array[:,:,1] >= 198) & (rgb_array[:,:,1] <= 202) &
        (rgb_array[:,:,2] >= 198) & (rgb_array[:,:,2] <= 202) &
        (alpha_array > 250)
    )
    
    # Inner ring: ONLY exact background color
    if np.any(inner_ring_mask):
        inner_candidates = inner_ring_mask & exact_background
        background_candidates |= inner_candidates
        logger.info(f"    Inner ring: {np.sum(inner_candidates)} exact background pixels")
    
    # Middle ring: Slightly relaxed but still conservative
    if np.any(middle_ring_mask):
        middle_candidates = (
            middle_ring_mask &
            (gray >= 195) & (gray <= 205) &  # Slightly wider range
            (np.abs(rgb_array[:,:,0].astype(np.float32) - rgb_array[:,:,1].astype(np.float32)) < 3) &
            (np.abs(rgb_array[:,:,1].astype(np.float32) - rgb_array[:,:,2].astype(np.float32)) < 3) &
            (alpha_array > 250)
        )
        background_candidates |= middle_candidates
        logger.info(f"    Middle ring: {np.sum(middle_candidates)} background candidates")
    
    # Outer ring: More aggressive detection
    if np.any(outer_ring_mask):
        outer_candidates = (
            outer_ring_mask &
            (gray > 240) &  # Bright areas
            (np.abs(rgb_array[:,:,0].astype(np.float32) - rgb_array[:,:,1].astype(np.float32)) < 8) &
            (np.abs(rgb_array[:,:,1].astype(np.float32) - rgb_array[:,:,2].astype(np.float32)) < 8) &
            (alpha_array > 200)
        )
        background_candidates |= outer_candidates
        logger.info(f"    Outer ring: {np.sum(outer_candidates)} background candidates")
    
    # Beyond ring: Most aggressive
    if np.any(beyond_ring_mask):
        beyond_candidates = (
            beyond_ring_mask &
            (gray > 235) &
            (np.abs(rgb_array[:,:,0].astype(np.float32) - rgb_array[:,:,1].astype(np.float32)) < 10) &
            (np.abs(rgb_array[:,:,1].astype(np.float32) - rgb_array[:,:,2].astype(np.float32)) < 10)
        )
        background_candidates |= beyond_candidates
        logger.info(f"    Beyond ring: {np.sum(beyond_candidates)} background candidates")
    
    # Stage 2: Continuity verification
    logger.info("  Stage 2: Verifying pixel continuity...")
    continuity_verified = np.zeros_like(background_candidates, dtype=bool)
    
    # Only check continuity for pixels that passed Stage 1
    candidate_coords = np.where(background_candidates)
    sample_rate = 5  # Check every 5th pixel for performance
    
    for idx in range(0, len(candidate_coords[0]), sample_rate):
        y, x = candidate_coords[0][idx], candidate_coords[1][idx]
        
        # Different window sizes for different regions
        if inner_ring_mask[y, x]:
            window_size = 7  # Larger window for inner ring (more strict)
        elif middle_ring_mask[y, x]:
            window_size = 5
        else:
            window_size = 3  # Smaller window for outer regions
        
        if check_pixel_continuity(gray, x, y, window=window_size):
            # Mark this pixel and nearby pixels as verified
            y_min = max(0, y - 1)
            y_max = min(h, y + 2)
            x_min = max(0, x - 1)
            x_max = min(w, x + 2)
            continuity_verified[y_min:y_max, x_min:x_max] = background_candidates[y_min:y_max, x_min:x_max]
    
    logger.info(f"    Continuity verified: {np.sum(continuity_verified)} pixels")
    
    # Stage 3: Morphological pattern verification
    logger.info("  Stage 3: Verifying morphological patterns...")
    final_mask = np.zeros_like(continuity_verified, dtype=bool)
    
    # Check if regions form hole-like patterns
    verified_coords = np.where(continuity_verified)
    sample_rate = 10  # Check every 10th pixel for performance
    
    for idx in range(0, len(verified_coords[0]), sample_rate):
        y, x = verified_coords[0][idx], verified_coords[1][idx]
        
        # Different pattern sizes for different regions
        if inner_ring_mask[y, x]:
            pattern_size = 9  # Larger pattern check for inner ring
        elif middle_ring_mask[y, x]:
            pattern_size = 7
        else:
            pattern_size = 5
        
        if verify_morphological_pattern(continuity_verified, x, y, pattern_size=pattern_size):
            # Mark region as final
            half_size = pattern_size // 2
            y_min = max(0, y - half_size)
            y_max = min(h, y + half_size + 1)
            x_min = max(0, x - half_size)
            x_max = min(w, x + half_size + 1)
            final_mask[y_min:y_max, x_min:x_max] = continuity_verified[y_min:y_max, x_min:x_max]
    
    logger.info(f"    Morphological verification: {np.sum(final_mask)} pixels")
    
    # Highlight preservation: Protect highlights even if they passed other tests
    highlight_mask = (
        (gray > 245) &  # Very bright
        (np.max(rgb_array, axis=2) - np.min(rgb_array, axis=2) > 10)  # Has color variation (not pure gray)
    )
    
    # Edge detection for highlight patterns
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges.convert('L'))
    edge_highlights = (edges_array > 50) & (gray > 230)
    
    # Gradient detection for highlights
    gray_image = Image.fromarray(gray)
    gray_blurred = gray_image.filter(ImageFilter.GaussianBlur(radius=3))
    gray_blurred_array = np.array(gray_blurred)
    gradient_mask = np.abs(gray.astype(np.float32) - gray_blurred_array.astype(np.float32)) > 15
    
    # Combined highlight protection
    protect_mask = (
        (inner_ring_mask & (highlight_mask | edge_highlights | gradient_mask)) |  # Protect all inner highlights
        (middle_ring_mask & highlight_mask & gradient_mask) |  # Protect strong middle highlights
        (outer_ring_mask & highlight_mask & edge_highlights & gradient_mask)  # Only protect very strong outer highlights
    )
    
    # Remove protected areas from final mask
    final_mask = final_mask & ~protect_mask
    
    logger.info(f"  Protected {np.sum(protect_mask)} highlight pixels")
    
    # Clean up using morphology
    final_mask = simple_morphology_operation(final_mask, 'open', iterations=1)
    final_mask = simple_morphology_operation(final_mask, 'close', iterations=1)
    
    # Apply to alpha channel with smooth transitions
    new_alpha = alpha_array.copy()
    
    if np.any(final_mask):
        # Create smooth transition
        mask_image = Image.fromarray((final_mask * 255).astype(np.uint8))
        
        # Different blur radii for different regions
        mask_smooth = np.zeros_like(final_mask, dtype=np.float32)
        
        # Inner ring: very subtle transition
        inner_final = final_mask & inner_ring_mask
        if np.any(inner_final):
            inner_image = Image.fromarray((inner_final * 255).astype(np.uint8))
            inner_blurred = inner_image.filter(ImageFilter.GaussianBlur(radius=0.5))
            mask_smooth += np.array(inner_blurred) / 255.0
        
        # Middle ring: moderate transition
        middle_final = final_mask & middle_ring_mask
        if np.any(middle_final):
            middle_image = Image.fromarray((middle_final * 255).astype(np.uint8))
            middle_blurred = middle_image.filter(ImageFilter.GaussianBlur(radius=1.0))
            mask_smooth += np.array(middle_blurred) / 255.0
        
        # Outer ring: smoother transition
        outer_final = final_mask & (outer_ring_mask | beyond_ring_mask)
        if np.any(outer_final):
            outer_image = Image.fromarray((outer_final * 255).astype(np.uint8))
            outer_blurred = outer_image.filter(ImageFilter.GaussianBlur(radius=1.5))
            mask_smooth += np.array(outer_blurred) / 255.0
        
        # Apply with gradient
        mask_smooth = np.clip(mask_smooth, 0, 1)
        new_alpha = (new_alpha * (1 - mask_smooth)).astype(np.uint8)
    
    # Final smoothing
    a_new = Image.fromarray(new_alpha)
    
    if np.sum(final_mask) > 100:
        a_new = a_new.filter(ImageFilter.MedianFilter(3))
        a_new = a_new.filter(ImageFilter.GaussianBlur(radius=0.3))
    
    result = Image.merge('RGBA', (r, g, b, a_new))
    
    removed_count = np.sum(final_mask)
    protected_count = np.sum(protect_mask)
    logger.info(f"✅ Advanced detection complete - {removed_count} pixels made transparent, {protected_count} highlights preserved")
    
    return result

def detect_cubic_regions_enhanced(image: Image.Image, sensitivity=1.0):
    """Enhanced cubic detection - balanced for better detail preservation"""
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
    
    # Detect different types of cubic regions
    white_cubic = (
        (v_array > 245 * sensitivity) &
        (s_array < 25) &
        (alpha_array > 200)
    )
    
    color_cubic = (
        (v_array > 210 * sensitivity) &
        (s_array > 100) &
        (alpha_array > 200)
    )
    
    edge_cubic = (edges_array > 100) & (v_array > 220) & (alpha_array > 200)
    
    highlights = (
        (v_array > 250) &
        (s_array < 35) &
        (alpha_array > 200)
    )
    
    cubic_mask = white_cubic | color_cubic | edge_cubic | highlights
    
    # Clean up the mask
    cubic_image = Image.fromarray((cubic_mask * 255).astype(np.uint8))
    cubic_image = cubic_image.filter(ImageFilter.MinFilter(3))
    cubic_image = cubic_image.filter(ImageFilter.MaxFilter(3))
    cubic_mask = np.array(cubic_image) > 128
    
    return cubic_mask.astype(bool), white_cubic, color_cubic, highlights

def enhance_cubic_sparkle_gradual(image: Image.Image, intensity=1.0, num_passes=3) -> Image.Image:
    """Enhanced cubic sparkle with stronger detail preservation"""
    logger.info(f"💎 Enhanced cubic detail processing ({num_passes} passes)...")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    result = image
    
    for pass_num in range(num_passes):
        logger.info(f"  💫 Enhanced cubic pass {pass_num + 1}/{num_passes}")
        
        r, g, b, a = result.split()
        rgb_array = np.array(result.convert('RGB'), dtype=np.float32)
        
        cubic_mask, white_cubic, color_cubic, highlights = detect_cubic_regions_enhanced(result, intensity)
        
        cubic_count = np.sum(cubic_mask)
        
        if cubic_count == 0:
            logger.info(f"    No cubic regions detected in pass {pass_num + 1}")
            continue
        
        logger.info(f"    Detected {cubic_count} cubic pixels")
        
        edges = result.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges_array = np.array(edges.convert('RGB'), dtype=np.float32)
        
        unsharp = result.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
        unsharp_array = np.array(unsharp.convert('RGB'), dtype=np.float32)
        
        pass_factor = 1.0 - (pass_num * 0.1)
        
        # Apply enhancements
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                rgb_array[:,:,c] * (1 - 0.25 * pass_factor) + edges_array[:,:,c] * (0.25 * pass_factor),
                rgb_array[:,:,c]
            )
            
            rgb_array[:,:,c] = np.where(
                highlights,
                rgb_array[:,:,c] * (1 - 0.15 * pass_factor) + unsharp_array[:,:,c] * (0.15 * pass_factor),
                rgb_array[:,:,c]
            )
        
        # Contrast adjustment
        if np.any(cubic_mask):
            mean_val = np.mean(rgb_array[cubic_mask])
            contrast_factor = 1.0 + (0.15 * intensity * pass_factor)
            
            for c in range(3):
                rgb_array[:,:,c] = np.where(
                    cubic_mask,
                    np.clip((rgb_array[:,:,c] - mean_val) * contrast_factor + mean_val, 0, 255),
                    rgb_array[:,:,c]
                )
        
        # Highlight boost
        if np.any(highlights):
            boost_factor = 1.0 + (0.08 * intensity * pass_factor)
            rgb_array[highlights] = np.minimum(rgb_array[highlights] * boost_factor, 255)
        
        # Color saturation for colored cubics
        if np.any(color_cubic):
            rgb_temp = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
            hsv_temp = rgb_temp.convert('HSV')
            h_temp, s_temp, v_temp = hsv_temp.split()
            
            s_array = np.array(s_temp, dtype=np.float32)
            v_array = np.array(v_temp, dtype=np.float32)
            
            s_array = np.where(
                color_cubic,
                np.minimum(s_array * (1.0 + 0.25 * intensity * pass_factor), 255),
                s_array
            )
            
            v_array = np.where(
                color_cubic,
                np.minimum(v_array * (1.0 + 0.05 * pass_factor), 255),
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
        
        # Final sharpness
        sharpness = ImageEnhance.Sharpness(result)
        result = sharpness.enhance(1.0 + (0.15 * intensity * pass_factor))
    
    logger.info("✅ Enhanced cubic processing complete")
    
    return result

def handler(event):
    """RunPod handler function - V29 Balanced"""
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
    """Process cubic detail enhancement with advanced ring detection"""
    try:
        logger.info("🚀 RunPod V29 - BALANCED OTHER Pattern Colors")
        logger.info("💎 Multi-stage verification for accurate hole detection")
        logger.info("🌈 OTHER PATTERN: Balanced Natural Enhancement for metallic colors")
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
        
        # Find image data
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
                        elif len(value) < 100:
                            error_details.append(f"- '{key}' too short ({len(value)} chars)")
                        else:
                            sample = value[:50].strip()
                            if not all(c in string.ascii_letters + string.digits + '+/=' for c in sample):
                                error_details.append(f"- '{key}' doesn't look like base64")
            
            error_details.append("\nExpected fields: 'image', 'base64', 'enhanced_image', 'thumbnail', etc.")
            error_details.append("Image data should be base64 encoded string (minimum 100 chars).")
            error_details.append("\nMake.com note: Check that the field mapping is correct.")
            error_details.append("The image field might be: {{4.data.output.output.enhanced_image}} or {{4.data.output.output.thumbnail}}")
            
            raise ValueError("\n".join(error_details))
        
        # Extract parameters
        params = job if isinstance(job, dict) else {}
        filename = params.get('filename', '')
        intensity = float(params.get('intensity', 1.2))
        intensity = max(0.1, min(2.0, intensity))
        apply_pattern = params.get('pattern_enhancement', True)
        
        default_pattern = params.get('default_pattern', 'ab_pattern')
        
        logger.info(f"Parameters: filename={filename or 'None (using default)'}, intensity={intensity}, pattern={apply_pattern}")
        
        # Decode image
        image_bytes = decode_base64_fast(image_data_str)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGBA':
            logger.info(f"Converting {image.mode} to RGBA")
            image = image.convert('RGBA')
        
        original_size = image.size
        logger.info(f"Input image size: {original_size}")
        
        # Step 1: White Balance
        logger.info("⚖️ Step 1/5: Applying white balance")
        image = auto_white_balance_fast(image)
        
        # Step 2: Pattern Enhancement with balanced colors
        if apply_pattern:
            logger.info("🎨 Step 2/5: Balanced pattern enhancement for natural colors")
            pattern_type = detect_pattern_type(filename, default_pattern=default_pattern)
            detected_type = {
                "ac_pattern": "무도금화이트(0.10)",
                "ab_pattern": "무도금화이트-쿨톤(0.10)",
                "other": "기타색상(균형) - Color 1.35, Brightness 0.95, Contrast 1.20, HSV x1.15/0.98, Edge 8%"
            }.get(pattern_type, "기타색상")
            
            logger.info(f"Detected pattern: {pattern_type} - {detected_type}")
            image = apply_pattern_enhancement_gradual(image, pattern_type)
        else:
            pattern_type = "none"
            detected_type = "보정없음"
        
        # Step 3: First cubic enhancement
        logger.info("💎 Step 3/5: Enhanced cubic pre-processing")
        image = enhance_cubic_sparkle_gradual(image, intensity * 1.0, num_passes=3)
        
        # Step 4: Advanced ring hole detection with multi-stage verification
        logger.info("🔍 Step 4/5: Advanced ring processing with highlight preservation")
        image = ensure_ring_holes_transparent_advanced(image)
        
        # Step 5: Second cubic enhancement (replaces SwinIR)
        logger.info("💎 Step 5/5: Enhanced cubic final processing (no external API)")
        enhanced_image = enhance_cubic_sparkle_gradual(image, intensity * 0.8, num_passes=2)
        
        # Convert to base64 with padding
        output_base64 = image_to_base64(enhanced_image)
        
        # Calculate statistics
        cubic_mask, _, _, _ = detect_cubic_regions_enhanced(image)
        cubic_pixel_count = int(np.sum(cubic_mask))
        cubic_percentage = (cubic_pixel_count / (image.size[0] * image.size[1])) * 100
        
        has_cubics = bool(cubic_pixel_count > 0)  # Convert numpy bool to Python bool
        
        # Return proper structure for Make.com
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
                    "pattern_enhancement_balanced" if apply_pattern else "pattern_skipped",
                    "cubic_enhancement_strong",
                    "ring_hole_detection_advanced",
                    "cubic_enhancement_final"
                ],
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "performance": "runpod_compatible_no_external_api",
                "processing_order": "1.WB → 2.Pattern(Balanced) → 3.Cubic1(Strong) → 4.RingHoles(Advanced) → 5.Cubic2(Strong)",
                "v29_balanced_changes": [
                    "OTHER PATTERN: Moderate color saturation 1.35 (natural metallic)",
                    "OTHER PATTERN: Balanced brightness 0.95 (preserves original)",
                    "OTHER PATTERN: Moderate contrast 1.20 (natural depth)",
                    "OTHER PATTERN: Gentle HSV boost - Saturation x1.15, Value x0.98",
                    "OTHER PATTERN: Subtle warm color boost for metallic tones",
                    "OTHER PATTERN: Subtle edge enhancement 8%",
                    "OTHER PATTERN: Moderate sharpness 1.25",
                    "OTHER PATTERN: Gentler final sharpness 1.10",
                    "Result: Natural metallic colors with preserved texture"
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
                "traceback": traceback.format_exc(),
                "help": "Check that image data is base64 encoded and at least 100 characters long. For thumbnail: use 'thumbnail' key."
            }
        }

# RunPod handler
runpod.serverless.start({"handler": handler})
