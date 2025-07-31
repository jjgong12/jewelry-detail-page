import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import logging
import string
import requests
import json

# Optional imports with graceful fallback
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    logging.warning("Replicate not available - SwinIR will be skipped")

try:
    from scipy.ndimage import label, center_of_mass
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("Scipy not available - Advanced sparkle effects will be skipped")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# CUBIC DETAIL ENHANCEMENT HANDLER V3.3
# VERSION: Cubic-Sparkle-V3.3-BuildFix
# Fixed build issues and dependencies
################################

VERSION = "Cubic-Sparkle-V3.3-BuildFix"

def decode_base64_fast(base64_str: str) -> bytes:
    """Fast base64 decode with padding handling"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string - too short")
        
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        # Remove whitespace
        base64_str = ''.join(base64_str.split())
        
        # Filter valid base64 characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        # Try decoding with existing padding first
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            logger.info("✅ Base64 decoded successfully with existing padding")
            return decoded
        except Exception:
            # Add proper padding if needed
            no_pad = base64_str.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
            logger.info("✅ Base64 decoded successfully after adding padding")
            return decoded
            
    except Exception as e:
        logger.error(f"Base64 decode error: {str(e)}")
        logger.error(f"Base64 string length: {len(base64_str) if base64_str else 0}")
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
    
    # Ensure proper padding for Google Script
    logger.info(f"📏 Base64 length: {len(base64_str)}, ends with: {base64_str[-4:]}")
    
    return base64_str

def find_input_data(data):
    """Extract input image data from various formats"""
    logger.info(f"🔍 Finding input data in type: {type(data)}")
    
    # Direct string input
    if isinstance(data, str):
        if len(data) > 50:
            logger.info("✅ Found direct string input")
            return data
        else:
            logger.warning(f"String too short: {len(data)} chars")
            return None
    
    # Dictionary input
    if isinstance(data, dict):
        logger.info(f"📊 Input dictionary keys: {list(data.keys())}")
        
        # Priority keys for cubic detail (after enhancement/thumbnail)
        priority_keys = ['enhanced_image', 'thumbnail', 'image', 'image_base64', 'base64', 'img']
        
        # Direct key check
        for key in priority_keys:
            if key in data:
                value = data[key]
                if isinstance(value, str) and len(value) > 50:
                    logger.info(f"✅ Found image in key '{key}' (length: {len(value)})")
                    return value
                elif isinstance(value, str):
                    logger.warning(f"Key '{key}' has short string: {len(value)} chars")
        
        # Nested structures (from other handlers)
        nested_keys = ['output', 'data', 'result', 'response']
        for nested in nested_keys:
            if nested in data and isinstance(data[nested], dict):
                logger.info(f"🔍 Checking nested '{nested}'")
                for img_key in priority_keys:
                    if img_key in data[nested]:
                        value = data[nested][img_key]
                        if isinstance(value, str) and len(value) > 50:
                            logger.info(f"✅ Found image in {nested}.{img_key}")
                            return value
        
        # Make.com numbered keys
        for i in range(10):
            key = str(i)
            if key in data:
                logger.info(f"🔍 Checking numbered key '{key}'")
                value = data[key]
                if isinstance(value, str) and len(value) > 50:
                    logger.info(f"✅ Found image in key '{key}'")
                    return value
                elif isinstance(value, dict):
                    # Recursively check
                    result = find_input_data(value)
                    if result:
                        return result
        
        # Deep recursive search as last resort
        logger.info("🔍 Starting deep recursive search...")
        for key, value in data.items():
            if isinstance(value, dict):
                result = find_input_data(value)
                if result:
                    logger.info(f"✅ Found image recursively via key '{key}'")
                    return result
    
    logger.error("❌ No valid image data found")
    return None

def detect_pattern_type(filename: str) -> str:
    """Detect pattern type from filename"""
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
    """Fast white balance preserving transparency"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_img = Image.merge('RGB', (r, g, b))
    
    img_array = np.array(rgb_img, dtype=np.float32)
    
    # Sample every 15th pixel for speed
    sampled = img_array[::15, ::15]
    
    # Find near-gray pixels
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
        
        # Apply correction
        img_array[:,:,0] *= (gray_avg / r_avg) if r_avg > 0 else 1
        img_array[:,:,1] *= (gray_avg / g_avg) if g_avg > 0 else 1
        img_array[:,:,2] *= (gray_avg / b_avg) if b_avg > 0 else 1
    
    rgb_balanced = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    r2, g2, b2 = rgb_balanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    return result

def apply_pattern_enhancement_transparent(image: Image.Image, pattern_type: str) -> Image.Image:
    """Apply pattern-specific enhancement"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    img_array = np.array(rgb_image, dtype=np.float32)
    
    if pattern_type == "ac_pattern":
        # AC Pattern - 무도금화이트
        logger.info("🔍 AC Pattern - 무도금화이트 (20% white overlay)")
        white_overlay = 0.20
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.03)
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
    elif pattern_type == "ab_pattern":
        # AB Pattern - 무도금화이트 쿨톤
        logger.info("🔍 AB Pattern - 무도금화이트 쿨톤 (16% white overlay)")
        white_overlay = 0.16
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        
        # Cool tone adjustment
        img_array[:,:,0] *= 0.96  # Reduce red
        img_array[:,:,1] *= 0.98  # Slightly reduce green
        img_array[:,:,2] *= 1.02  # Boost blue
        
        # Cool overlay
        cool_overlay = np.array([240, 248, 255], dtype=np.float32)
        img_array = img_array * 0.95 + cool_overlay * 0.05
        
        img_array = np.clip(img_array, 0, 255)
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.88)
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.03)
        
    else:
        # Other patterns - 기타색상
        logger.info("🔍 Other Pattern - 기타색상 (5% white overlay)")
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
    
    # Common enhancements
    contrast = ImageEnhance.Contrast(rgb_image)
    rgb_image = contrast.enhance(1.1)
    
    sharpness = ImageEnhance.Sharpness(rgb_image)
    rgb_image = sharpness.enhance(1.8)
    
    # Merge back with alpha
    r2, g2, b2 = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
    
    return enhanced_image

def apply_swinir_enhancement(image: Image.Image) -> Image.Image:
    """Apply SwinIR enhancement if available"""
    if not REPLICATE_AVAILABLE:
        logger.warning("❌ Replicate not available, skipping SwinIR")
        return image
    
    try:
        logger.info("🎨 Applying SwinIR enhancement")
        
        api_token = os.environ.get('REPLICATE_API_TOKEN')
        if not api_token:
            logger.warning("❌ No REPLICATE_API_TOKEN found")
            return image
        
        client = replicate.Client(api_token=api_token)
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Separate alpha channel
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        # Convert to base64
        buffered = BytesIO()
        rgb_image.save(buffered, format="PNG", optimize=True, compress_level=3)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        # Run SwinIR
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
            
            # Merge back with alpha
            r2, g2, b2 = enhanced_image.split()
            result = Image.merge('RGBA', (r2, g2, b2, a))
            
            logger.info("✅ SwinIR enhancement successful")
            return result
            
    except Exception as e:
        logger.error(f"❌ SwinIR error: {str(e)}")
        
    return image

def ensure_ring_holes_transparent_simple(image: Image.Image) -> Image.Image:
    """Simple ring hole detection without OpenCV"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 Detecting ring holes")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    
    # Convert to HSV
    hsv = image.convert('HSV')
    h, s, v = hsv.split()
    s_array = np.array(s)
    v_array = np.array(v)
    
    # Ring holes are very bright with low saturation
    holes_mask = (v_array > 248) & (s_array < 20) & (alpha_array > 200)
    
    # Simple morphology using PIL
    holes_image = Image.fromarray((holes_mask * 255).astype(np.uint8))
    holes_image = holes_image.filter(ImageFilter.MinFilter(3))  # erode
    holes_image = holes_image.filter(ImageFilter.MaxFilter(3))  # dilate
    holes_mask = np.array(holes_image) > 128
    
    # Make holes transparent
    alpha_array[holes_mask] = 0
    
    a_new = Image.fromarray(alpha_array)
    result = Image.merge('RGBA', (r, g, b, a_new))
    
    hole_count = np.sum(holes_mask)
    logger.info(f"✅ Found {hole_count} ring hole pixels")
    
    return result

def detect_cubic_regions_enhanced(image: Image.Image, sensitivity=1.0):
    """Enhanced cubic detection"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    alpha_array = np.array(image.split()[3], dtype=np.uint8)
    
    # Convert to HSV
    hsv = image.convert('HSV')
    h, s, v = hsv.split()
    s_array = np.array(s)
    v_array = np.array(v)
    
    # Edge detection
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges.convert('L'))
    
    # Detect different types of cubics
    white_cubic = (
        (v_array > 240 * sensitivity) & 
        (s_array < 30) & 
        (alpha_array > 200)
    )
    
    color_cubic = (
        (v_array > 200 * sensitivity) & 
        (s_array > 100) & 
        (alpha_array > 200)
    )
    
    edge_cubic = (edges_array > 100) & (v_array > 220) & (alpha_array > 200)
    
    highlights = (
        (v_array > 250) & 
        (s_array < 50) &
        (alpha_array > 200)
    )
    
    # Combine all cubic masks
    cubic_mask = white_cubic | color_cubic | edge_cubic | highlights
    
    # Clean up using morphology
    cubic_image = Image.fromarray((cubic_mask * 255).astype(np.uint8))
    cubic_image = cubic_image.filter(ImageFilter.MinFilter(3))
    cubic_image = cubic_image.filter(ImageFilter.MaxFilter(3))
    cubic_mask = np.array(cubic_image) > 128
    
    return cubic_mask.astype(bool), white_cubic, color_cubic, highlights

def enhance_cubic_sparkle_with_swinir(image: Image.Image, intensity=1.0) -> Image.Image:
    """Pre-enhancement for SwinIR"""
    logger.info("💎 Enhancing cubic details...")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
    
    # Detect cubic regions
    cubic_mask, white_cubic, color_cubic, highlights = detect_cubic_regions_enhanced(image, intensity)
    
    cubic_count = np.sum(cubic_mask)
    logger.info(f"✨ Found {cubic_count} cubic pixels ({cubic_count/(image.size[0]*image.size[1])*100:.1f}%)")
    
    if cubic_count == 0:
        logger.info("No cubic regions found")
        return image
    
    # 1. Edge enhancement
    edges = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    edges_array = np.array(edges.convert('RGB'), dtype=np.float32)
    
    for c in range(3):
        rgb_array[:,:,c] = np.where(
            cubic_mask,
            rgb_array[:,:,c] * 0.7 + edges_array[:,:,c] * 0.3,
            rgb_array[:,:,c]
        )
    
    # 2. Contrast boost
    if np.any(cubic_mask):
        mean_val = np.mean(rgb_array[cubic_mask])
        contrast_factor = 1.2 * intensity
        
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                np.clip((rgb_array[:,:,c] - mean_val) * contrast_factor + mean_val, 0, 255),
                rgb_array[:,:,c]
            )
    
    # 3. Highlight boost
    if np.any(highlights):
        boost_factor = 1.15 * intensity
        rgb_array[highlights] = np.minimum(rgb_array[highlights] * boost_factor, 255)
    
    # 4. Color cubic saturation
    if np.any(color_cubic):
        rgb_temp = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
        hsv_temp = rgb_temp.convert('HSV')
        h_temp, s_temp, v_temp = hsv_temp.split()
        
        s_array = np.array(s_temp, dtype=np.float32)
        v_array = np.array(v_temp, dtype=np.float32)
        
        s_array = np.where(
            color_cubic,
            np.minimum(s_array * (1.4 * intensity), 255),
            s_array
        )
        
        v_array = np.where(
            color_cubic,
            np.minimum(v_array * 1.05, 255),
            v_array
        )
        
        hsv_enhanced = Image.merge('HSV', (
            h_temp,
            Image.fromarray(s_array.astype(np.uint8)),
            Image.fromarray(v_array.astype(np.uint8))
        ))
        rgb_array = np.array(hsv_enhanced.convert('RGB'), dtype=np.float32)
    
    # 5. Add sparkle points (if scipy available)
    if SCIPY_AVAILABLE:
        try:
            labeled_cubics, num_features = label(cubic_mask)
            
            for i in range(1, min(num_features + 1, 200)):
                region_mask = labeled_cubics == i
                if np.sum(region_mask) < 5:
                    continue
                
                region_brightness = np.mean(rgb_array, axis=2) * region_mask
                if np.any(region_brightness > 0):
                    max_pos = np.unravel_index(np.argmax(region_brightness), region_brightness.shape)
                    y, x = max_pos
                    
                    sparkle_radius = 3
                    for dy in range(-sparkle_radius, sparkle_radius + 1):
                        for dx in range(-sparkle_radius, sparkle_radius + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < rgb_array.shape[0] and 0 <= nx < rgb_array.shape[1]:
                                dist = np.sqrt(dy**2 + dx**2)
                                if dist <= sparkle_radius:
                                    sparkle_intensity = (1 - dist/sparkle_radius) * 0.3 * intensity
                                    rgb_array[ny, nx] = np.minimum(
                                        rgb_array[ny, nx] + (255 - rgb_array[ny, nx]) * sparkle_intensity,
                                        255
                                    )
        except Exception as e:
            logger.warning(f"Sparkle enhancement error: {e}")
    
    # Convert back to image
    rgb_enhanced = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
    r2, g2, b2 = rgb_enhanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    # Final sharpening
    sharpness = ImageEnhance.Sharpness(result)
    result = sharpness.enhance(1.0 + (0.3 * intensity))
    
    logger.info("✅ Cubic enhancement complete")
    
    return result

def handler(event):
    """RunPod handler function"""
    logger.info(f"=== Cubic Detail Enhancement {VERSION} Started ===")
    logger.info(f"📥 Event type: {type(event)}")
    
    try:
        # Parse event data
        job_input = None
        
        if isinstance(event, dict):
            logger.info(f"📋 Event keys: {list(event.keys())}")
            
            # Try different possible structures
            if 'input' in event:
                job_input = event['input']
                logger.info("✅ Found data in event['input']")
            elif 'job' in event:
                job_input = event['job']
                logger.info("✅ Found data in event['job']")
            elif 'data' in event:
                job_input = event['data']
                logger.info("✅ Found data in event['data']")
            else:
                # Direct event
                job_input = event
                logger.info("✅ Using event directly")
        else:
            logger.error(f"❌ Unexpected event type: {type(event)}")
            return {
                "output": {
                    "error": f"Invalid event type: {type(event)}",
                    "status": "failed",
                    "version": VERSION
                }
            }
        
        # Process the job
        return process_cubic_enhancement(job_input)
        
    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Traceback:\n{tb}")
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION,
                "traceback": tb
            }
        }

def process_cubic_enhancement(job):
    """Main processing function"""
    try:
        logger.info("🚀 Processing cubic detail enhancement")
        
        # Find input image
        image_data_str = find_input_data(job)
        
        if not image_data_str:
            logger.error("❌ No input image found")
            if isinstance(job, dict):
                logger.error(f"Available keys: {list(job.keys())}")
                # Log structure
                for key, value in list(job.items())[:5]:  # First 5 items
                    if isinstance(value, str):
                        logger.error(f"  {key}: string({len(value)} chars)")
                    elif isinstance(value, dict):
                        logger.error(f"  {key}: dict({list(value.keys())})")
                    else:
                        logger.error(f"  {key}: {type(value)}")
            
            raise ValueError("No valid image data found in input")
        
        # Extract parameters
        filename = ''
        intensity = 1.0
        apply_swinir = True
        apply_pattern = True
        
        if isinstance(job, dict):
            filename = job.get('filename', '')
            intensity = float(job.get('intensity', 1.0))
            apply_swinir = job.get('apply_swinir', True)
            apply_pattern = job.get('pattern_enhancement', True)
        
        intensity = max(0.1, min(2.0, intensity))
        
        logger.info(f"📋 Parameters:")
        logger.info(f"  - Filename: {filename}")
        logger.info(f"  - Intensity: {intensity}")
        logger.info(f"  - Apply SwinIR: {apply_swinir}")
        logger.info(f"  - Apply Pattern: {apply_pattern}")
        
        # Decode image
        image_bytes = decode_base64_fast(image_data_str)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGBA':
            logger.info(f"Converting {image.mode} to RGBA")
            image = image.convert('RGBA')
        
        original_size = image.size
        logger.info(f"📐 Image size: {original_size[0]}x{original_size[1]}")
        
        # Processing pipeline
        # 1. White Balance
        logger.info("⚖️ Step 1/5: White Balance")
        image = auto_white_balance_fast(image)
        
        # 2. Pattern Enhancement
        if apply_pattern:
            logger.info("🎨 Step 2/5: Pattern Enhancement")
            pattern_type = detect_pattern_type(filename)
            detected_type = {
                "ac_pattern": "무도금화이트(20%)",
                "ab_pattern": "무도금화이트-쿨톤(16%)",
                "other": "기타색상(5%)"
            }.get(pattern_type, "Unknown")
            
            logger.info(f"  Pattern: {pattern_type} ({detected_type})")
            image = apply_pattern_enhancement_transparent(image, pattern_type)
        else:
            logger.info("⏭️ Step 2/5: Pattern Enhancement (skipped)")
            pattern_type = "none"
            detected_type = "보정없음"
        
        # 3. Ring Hole Detection
        logger.info("🔍 Step 3/5: Ring Hole Detection")
        image = ensure_ring_holes_transparent_simple(image)
        
        # 4. Cubic Enhancement
        logger.info("💎 Step 4/5: Cubic Enhancement")
        image = enhance_cubic_sparkle_with_swinir(image, intensity)
        
        # 5. SwinIR
        if apply_swinir and REPLICATE_AVAILABLE:
            logger.info("🚀 Step 5/5: SwinIR Detail Enhancement")
            enhanced_image = apply_swinir_enhancement(image)
        else:
            if not REPLICATE_AVAILABLE:
                logger.info("⏭️ Step 5/5: SwinIR (not available)")
            else:
                logger.info("⏭️ Step 5/5: SwinIR (skipped)")
            enhanced_image = image
        
        # Convert to base64
        output_base64 = image_to_base64(enhanced_image)
        logger.info(f"📤 Output base64 length: {len(output_base64)}")
        
        # Calculate statistics
        cubic_mask, _, _, _ = detect_cubic_regions_enhanced(enhanced_image)
        cubic_pixel_count = int(np.sum(cubic_mask))
        total_pixels = enhanced_image.size[0] * enhanced_image.size[1]
        cubic_percentage = (cubic_pixel_count / total_pixels) * 100
        
        # Build response
        response = {
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
                    "cubic_pixels": cubic_pixel_count,
                    "cubic_percentage": round(cubic_percentage, 2),
                    "has_cubics": cubic_pixel_count > 0,
                    "total_pixels": total_pixels
                },
                "corrections_applied": [
                    "white_balance",
                    f"pattern_enhancement_{pattern_type}" if apply_pattern else "pattern_skipped",
                    "ring_hole_detection",
                    f"cubic_enhancement_intensity_{intensity}",
                    "swinir_detail" if (apply_swinir and REPLICATE_AVAILABLE) else "swinir_skipped"
                ],
                "dependencies": {
                    "replicate": REPLICATE_AVAILABLE,
                    "scipy": SCIPY_AVAILABLE
                },
                "base64_padding": "INCLUDED",
                "compression": "PNG_LEVEL_3"
            }
        }
        
        logger.info("✅ Processing complete!")
        logger.info(f"📊 Summary: {cubic_pixel_count} cubic pixels ({cubic_percentage:.1f}%)")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Processing error: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Traceback:\n{tb}")
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION,
                "traceback": tb,
                "dependencies": {
                    "replicate": REPLICATE_AVAILABLE,
                    "scipy": SCIPY_AVAILABLE
                }
            }
        }

# RunPod entry point
if __name__ == "__main__":
    logger.info(f"🚀 Starting RunPod handler v{VERSION}")
    logger.info(f"📦 Dependencies: Replicate={REPLICATE_AVAILABLE}, Scipy={SCIPY_AVAILABLE}")
    runpod.serverless.start({"handler": handler})
