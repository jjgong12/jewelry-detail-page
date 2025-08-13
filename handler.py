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
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# CUBIC DETAIL ENHANCEMENT HANDLER V21-HIGHLIGHT-PRESERVE
# VERSION: Cubic-Sparkle-V21-Enhanced-Highlight-Preserve
# Updated: Improved ring highlight preservation and natural processing
################################

VERSION = "Cubic-Sparkle-V21-Enhanced-Highlight-Preserve"

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
    """Detect pattern type - AC, AB, or other"""
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
    else:
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
    """Apply pattern enhancement with enhanced multi-pass cubic detail"""
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
        
        img_array[:,:,0] *= 0.96
        img_array[:,:,1] *= 0.98
        img_array[:,:,2] *= 1.02
        
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
        
    else:
        logger.info("🔍 Other Pattern (기타색상) - Standard enhancement")
        white_overlay = 0.02
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.05)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.99)
        
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(1.08)
        
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.3)
    
    sharpness = ImageEnhance.Sharpness(rgb_image)
    if pattern_type in ["ac_pattern", "ab_pattern"]:
        rgb_image = sharpness.enhance(1.4)
    else:
        rgb_image = sharpness.enhance(1.25)
    
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

def ensure_ring_holes_transparent_advanced(image: Image.Image) -> Image.Image:
    """Advanced ring hole detection - preserves metallic highlights naturally"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 Advanced ring hole detection with natural highlight preservation")
    
    r, g, b, a = image.split()
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    alpha_array = np.array(a, dtype=np.uint8)
    
    # Convert to different color spaces for analysis
    gray = np.mean(rgb_array, axis=2).astype(np.uint8)
    hsv = image.convert('HSV')
    h, s, v = hsv.split()
    h_array = np.array(h)
    s_array = np.array(s)
    v_array = np.array(v)
    
    # Step 1: Detect true background (non-metallic gray areas)
    logger.info("  Step 1: Detecting true background areas")
    
    # True background has very specific characteristics
    background_mask = (
        (gray > 195) & (gray < 205) &  # Narrow gray range
        (np.abs(rgb_array[:,:,0] - rgb_array[:,:,1]) < 8) &  # Very neutral
        (np.abs(rgb_array[:,:,1] - rgb_array[:,:,2]) < 8) &
        (s_array < 10) &  # Almost no saturation
        (alpha_array > 250)
    )
    
    # Step 2: Gradient analysis to distinguish highlights from holes
    logger.info("  Step 2: Gradient analysis for highlight detection")
    
    # Calculate gradients
    grad_x = np.gradient(gray.astype(np.float32), axis=1)
    grad_y = np.gradient(gray.astype(np.float32), axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Metallic highlights have gradients, holes don't
    has_gradient = gradient_magnitude > 5
    
    # Step 3: Detect metallic highlights (should be preserved)
    logger.info("  Step 3: Identifying metallic highlights to preserve")
    
    # Metallic highlights characteristics
    metallic_highlight = (
        (v_array > 240) &  # Very bright
        (s_array < 30) &  # Low saturation but not zero
        has_gradient &  # Has gradient transitions
        (gradient_magnitude < 50)  # Not an edge
    )
    
    # Step 4: Detect ring edges to understand structure
    logger.info("  Step 4: Edge detection for ring structure")
    
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges.convert('L'))
    strong_edges = edges_array > 100
    
    # Dilate edges to create edge proximity mask
    from scipy.ndimage import binary_dilation
    edge_proximity = binary_dilation(strong_edges, iterations=3)
    
    # Step 5: Continuity analysis
    logger.info("  Step 5: Continuity analysis for hole detection")
    
    # True holes are continuous regions without texture
    potential_holes = (
        (v_array > 245) &  # Very bright
        (s_array < 15) &  # Very low saturation
        (~has_gradient) &  # No gradient (flat area)
        (~edge_proximity) &  # Not near edges
        (alpha_array > 250)
    )
    
    # Label connected components
    from scipy.ndimage import label, binary_opening, binary_closing
    
    # Clean up the mask
    potential_holes = binary_opening(potential_holes, iterations=1)
    potential_holes = binary_closing(potential_holes, iterations=1)
    
    labeled_holes, num_features = label(potential_holes)
    
    # Step 6: Analyze each potential hole region
    logger.info("  Step 6: Analyzing individual regions")
    
    final_holes_mask = np.zeros_like(potential_holes, dtype=bool)
    
    for i in range(1, num_features + 1):
        region = labeled_holes == i
        region_size = np.sum(region)
        
        if region_size < 20:  # Too small, probably noise
            continue
        
        # Check if region is truly uniform (characteristic of holes)
        region_gray_values = gray[region]
        region_std = np.std(region_gray_values)
        
        # Check color uniformity
        region_r = rgb_array[:,:,0][region]
        region_g = rgb_array[:,:,1][region]
        region_b = rgb_array[:,:,2][region]
        
        color_variation = np.std(region_r) + np.std(region_g) + np.std(region_b)
        
        # True holes have very uniform color
        if region_std < 5 and color_variation < 15:
            # Additional check: is it surrounded by metal?
            region_dilated = binary_dilation(region, iterations=5)
            surrounding = region_dilated & ~region
            
            if np.any(surrounding):
                surrounding_gray = gray[surrounding]
                # If surrounded by darker metal, it's likely a hole
                if np.mean(surrounding_gray) < np.mean(region_gray_values) - 20:
                    final_holes_mask |= region
    
    # Step 7: Special handling for inner ring highlights
    logger.info("  Step 7: Processing inner ring area specially")
    
    h, w = alpha_array.shape
    center_y, center_x = h // 2, w // 2
    
    # Create radial distance map
    y_indices, x_indices = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    # Inner ring area
    inner_ring_radius = min(h, w) // 5
    inner_ring_mask = distance_from_center < inner_ring_radius
    
    # In inner ring, be more conservative about removing highlights
    inner_highlights = inner_ring_mask & metallic_highlight
    
    # Remove inner highlights from holes mask
    final_holes_mask = final_holes_mask & ~inner_highlights
    
    # Step 8: Apply transparency with smooth transitions
    logger.info("  Step 8: Applying smooth transparency transitions")
    
    # Create smooth alpha channel
    new_alpha = alpha_array.copy()
    
    # Apply background mask
    new_alpha[background_mask] = 0
    
    # Apply holes mask with feathering
    if np.any(final_holes_mask):
        # Create distance transform for smooth edges
        from scipy.ndimage import distance_transform_edt
        
        holes_float = final_holes_mask.astype(np.float32)
        # Small blur for smoother edges
        holes_image = Image.fromarray((holes_float * 255).astype(np.uint8))
        holes_blurred = holes_image.filter(ImageFilter.GaussianBlur(radius=1))
        holes_smooth = np.array(holes_blurred) / 255.0
        
        # Apply with gradient
        new_alpha = (new_alpha * (1 - holes_smooth * 0.95)).astype(np.uint8)
    
    # Step 9: Preserve metallic highlights explicitly
    logger.info("  Step 9: Ensuring metallic highlights are preserved")
    
    # Restore alpha for confirmed metallic highlights
    metallic_preserve = metallic_highlight & (gradient_magnitude > 3) & (gradient_magnitude < 30)
    new_alpha[metallic_preserve] = np.maximum(new_alpha[metallic_preserve], 240)
    
    # Final smoothing
    a_new = Image.fromarray(new_alpha)
    
    # Very light smoothing to blend edges
    if np.sum(final_holes_mask) > 50:
        a_new = a_new.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    result = Image.merge('RGBA', (r, g, b, a_new))
    
    holes_count = np.sum(final_holes_mask)
    highlights_preserved = np.sum(metallic_preserve)
    logger.info(f"✅ Hole detection complete - {holes_count} hole pixels, {highlights_preserved} highlight pixels preserved")
    
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
        
        if np.any(cubic_mask):
            mean_val = np.mean(rgb_array[cubic_mask])
            contrast_factor = 1.0 + (0.15 * intensity * pass_factor)
            
            for c in range(3):
                rgb_array[:,:,c] = np.where(
                    cubic_mask,
                    np.clip((rgb_array[:,:,c] - mean_val) * contrast_factor + mean_val, 0, 255),
                    rgb_array[:,:,c]
                )
        
        if np.any(highlights):
            boost_factor = 1.0 + (0.08 * intensity * pass_factor)
            rgb_array[highlights] = np.minimum(rgb_array[highlights] * boost_factor, 255)
        
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
        
        sharpness = ImageEnhance.Sharpness(result)
        result = sharpness.enhance(1.0 + (0.15 * intensity * pass_factor))
    
    logger.info("✅ Enhanced cubic processing complete")
    
    return result

def handler(event):
    """RunPod handler function - V21 with Natural Highlight Preservation"""
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
    """Process cubic detail enhancement with natural highlight preservation"""
    try:
        logger.info("🚀 Natural Highlight Preservation Version")
        logger.info("💎 Advanced ring processing with metallic highlight detection")
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
        intensity = float(params.get('intensity', 1.2))
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
        
        # Step 1: White Balance
        logger.info("⚖️ Step 1/6: Applying white balance")
        image = auto_white_balance_fast(image)
        
        # Step 2: Pattern Enhancement with enhanced detail
        if apply_pattern:
            logger.info("🎨 Step 2/6: Enhanced pattern detail")
            pattern_type = detect_pattern_type(filename, default_pattern=default_pattern)
            detected_type = {
                "ac_pattern": "무도금화이트(0.10)",
                "ab_pattern": "무도금화이트-쿨톤(0.10)",
                "other": "기타색상(0.02)"
            }.get(pattern_type, "기타색상")
            
            logger.info(f"Detected pattern: {pattern_type} - {detected_type}")
            image = apply_pattern_enhancement_gradual(image, pattern_type)
        else:
            pattern_type = "none"
            detected_type = "보정없음"
        
        # Step 3: First cubic enhancement
        logger.info("💎 Step 3/6: Enhanced cubic pre-processing")
        image = enhance_cubic_sparkle_gradual(image, intensity * 1.0, num_passes=3)
        
        # Step 4: Advanced ring hole detection with natural highlight preservation
        logger.info("🔍 Step 4/6: Advanced ring processing with highlight preservation")
        image = ensure_ring_holes_transparent_advanced(image)
        
        # Step 5: Second cubic enhancement
        logger.info("💎 Step 5/6: Enhanced cubic final processing")
        image = enhance_cubic_sparkle_gradual(image, intensity * 0.8, num_passes=2)
        
        # Step 6: SwinIR Enhancement
        if apply_swinir:
            logger.info("🚀 Step 6/6: Applying SwinIR for maximum detail")
            enhanced_image = apply_swinir_enhancement(image)
        else:
            logger.info("⏭️ Step 6/6: Skipping SwinIR (disabled)")
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
                    "pattern_enhancement_enhanced" if apply_pattern else "pattern_skipped",
                    "cubic_enhancement_strong",
                    "ring_hole_detection_natural",
                    "cubic_enhancement_final",
                    "swinir_detail_maximum" if apply_swinir else "swinir_skipped"
                ],
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "performance": "natural_highlight_preservation",
                "processing_order": "1.WB → 2.Pattern(Enhanced) → 3.Cubic1(Strong) → 4.RingHoles(Natural) → 5.Cubic2(Strong) → 6.SwinIR",
                "v21_improvements": [
                    "Advanced gradient analysis for highlight detection",
                    "Metallic highlight preservation system",
                    "Continuity analysis for true hole detection",
                    "Radial distance mapping for inner ring processing",
                    "Multi-stage region analysis with statistics",
                    "Smooth alpha transitions with feathering",
                    "Edge proximity detection for structure understanding",
                    "Color uniformity checks for hole validation",
                    "Preserved natural lighting on ring surfaces",
                    "scipy.ndimage integration for better morphology"
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
