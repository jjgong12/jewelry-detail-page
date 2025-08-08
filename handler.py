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
# CUBIC DETAIL ENHANCEMENT HANDLER V13
# VERSION: Cubic-Sparkle-V13-Balanced
# Fixed brightness and contrast issues
################################

VERSION = "Cubic-Sparkle-V13-Balanced"

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
    # ALWAYS include padding - no removal
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str

def find_input_data_robust(data, path="", depth=0, max_depth=10):
    """More robust input data extraction with detailed logging"""
    if depth > max_depth:
        return None
    
    logger.info(f"🔍 Searching at depth {depth}, path: {path}")
    
    # Direct string check - more lenient
    if isinstance(data, str):
        str_len = len(data)
        logger.info(f"  Found string at {path} with length {str_len}")
        
        # Check if it's likely base64 (even if short)
        if str_len > 20:  # Reduced from 50
            # Remove whitespace for checking
            sample = data[:100].strip()
            # Basic base64 character check
            if sample and all(c in string.ascii_letters + string.digits + '+/=' for c in sample[:50]):
                logger.info(f"✅ Found potential base64 data at {path} (length: {str_len})")
                return data
        elif str_len > 0:
            logger.info(f"  String too short at {path}: {str_len} chars")
    
    if isinstance(data, dict):
        logger.info(f"  Dict at {path} with keys: {list(data.keys())}")
        
        # Check ALL string values, not just priority keys
        base64_candidates = []
        
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # Skip image data
            if key in ['enhanced_image', 'image', 'base64', 'image_base64'] and isinstance(value, str) and len(value) > 1000:
                continue
                
            if isinstance(value, str) and len(value) > 20:
                # Log all substantial strings
                logger.info(f"  Checking string at {current_path}: length={len(value)}")
                
                # Check if it looks like base64
                sample = value[:100].strip()
                if sample and all(c in string.ascii_letters + string.digits + '+/=' for c in sample[:50]):
                    base64_candidates.append((current_path, value, len(value)))
            
            # Recursive search
            if isinstance(value, (dict, list)):
                result = find_input_data_robust(value, current_path, depth + 1, max_depth)
                if result:
                    return result
        
        # If we found base64 candidates, use the longest one
        if base64_candidates:
            # Sort by length, longest first
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

def apply_pattern_enhancement_transparent(image: Image.Image, pattern_type: str) -> Image.Image:
    """Apply pattern enhancement while preserving transparency - BALANCED VERSION"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_image = Image.merge('RGB', (r, g, b))
    
    img_array = np.array(rgb_image, dtype=np.float32)
    
    if pattern_type == "ac_pattern":
        # AC Pattern - Slightly reduced brightness
        logger.info("🔍 AC Pattern - Applying 18% white overlay with brightness 1.01")
        white_overlay = 0.18  # Reduced from 0.20
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.01)  # Reduced from 1.03
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.98)
        
    elif pattern_type == "ab_pattern":
        # AB Pattern - Slightly reduced brightness
        logger.info("🔍 AB Pattern - Applying 14% white overlay with brightness 1.01")
        white_overlay = 0.14  # Reduced from 0.16
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
        rgb_image = brightness.enhance(1.01)  # Reduced from 1.03
        
    else:
        # Other Pattern - SIGNIFICANTLY reduced white overlay and brightness
        logger.info("🔍 Other Pattern - Applying 2% white overlay with brightness 1.03")
        white_overlay = 0.02  # Greatly reduced from 0.05
        img_array = img_array * (1 - white_overlay) + 255 * white_overlay
        img_array = np.clip(img_array, 0, 255)
        
        rgb_image = Image.fromarray(img_array.astype(np.uint8))
        
        brightness = ImageEnhance.Brightness(rgb_image)
        rgb_image = brightness.enhance(1.03)  # Greatly reduced from 1.12
        
        color = ImageEnhance.Color(rgb_image)
        rgb_image = color.enhance(0.99)
        
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.5)
    
    # Common adjustments for all patterns
    contrast = ImageEnhance.Contrast(rgb_image)
    rgb_image = contrast.enhance(1.1)
    
    sharpness = ImageEnhance.Sharpness(rgb_image)
    rgb_image = sharpness.enhance(1.8)
    
    r2, g2, b2 = rgb_image.split()
    enhanced_image = Image.merge('RGBA', (r2, g2, b2, a))
    
    return enhanced_image

def apply_swinir_enhancement(image: Image.Image) -> Image.Image:
    """Apply SwinIR enhancement - delayed loading"""
    try:
        logger.info("🎨 Applying SwinIR enhancement for cubic detail")
        
        # Delayed loading
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

def ensure_ring_holes_transparent_simple(image: Image.Image) -> Image.Image:
    """Simple ring hole detection without OpenCV"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    logger.info("🔍 Simple ring hole detection")
    
    r, g, b, a = image.split()
    alpha_array = np.array(a, dtype=np.uint8)
    
    # Convert to HSV for better detection
    hsv = image.convert('HSV')
    h, s, v = hsv.split()
    s_array = np.array(s)
    v_array = np.array(v)
    
    # Ring holes are very bright and low saturation
    holes_mask = (v_array > 248) & (s_array < 20) & (alpha_array > 200)
    
    # Simple morphology using PIL
    holes_image = Image.fromarray((holes_mask * 255).astype(np.uint8))
    holes_image = holes_image.filter(ImageFilter.MinFilter(3))  # erode
    holes_image = holes_image.filter(ImageFilter.MaxFilter(3))  # dilate
    holes_mask = np.array(holes_image) > 128
    
    # Apply holes
    alpha_array[holes_mask] = 0
    
    a_new = Image.fromarray(alpha_array)
    result = Image.merge('RGBA', (r, g, b, a_new))
    
    logger.info("✅ Ring holes applied")
    return result

def detect_cubic_regions_enhanced(image: Image.Image, sensitivity=1.0):
    """Enhanced cubic detection for better SwinIR results"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    alpha_array = np.array(image.split()[3], dtype=np.uint8)
    
    # Convert to HSV
    hsv = image.convert('HSV')
    h, s, v = hsv.split()
    h_array = np.array(h)
    s_array = np.array(s)
    v_array = np.array(v)
    
    # Edge detection using PIL
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges_array = np.array(edges.convert('L'))
    
    # Enhanced cubic detection
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
    
    # Edge-based cubic detection
    edge_cubic = (edges_array > 100) & (v_array > 220) & (alpha_array > 200)
    
    highlights = (
        (v_array > 250) & 
        (s_array < 50) &
        (alpha_array > 200)
    )
    
    cubic_mask = white_cubic | color_cubic | edge_cubic | highlights
    
    # Clean up using PIL filters
    cubic_image = Image.fromarray((cubic_mask * 255).astype(np.uint8))
    cubic_image = cubic_image.filter(ImageFilter.MinFilter(3))
    cubic_image = cubic_image.filter(ImageFilter.MaxFilter(3))
    cubic_mask = np.array(cubic_image) > 128
    
    return cubic_mask.astype(bool), white_cubic, color_cubic, highlights

def enhance_cubic_sparkle_with_swinir(image: Image.Image, intensity=1.0) -> Image.Image:
    """Enhanced cubic sparkle optimized for SwinIR"""
    logger.info("💎 Enhanced cubic detail processing for SwinIR...")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
    
    cubic_mask, white_cubic, color_cubic, highlights = detect_cubic_regions_enhanced(image, intensity)
    
    cubic_count = np.sum(cubic_mask)
    logger.info(f"✨ Detected {cubic_count} cubic pixels")
    
    if cubic_count == 0:
        logger.info("No cubic regions detected, returning original")
        return image
    
    # Pre-enhancement for SwinIR
    # Boost cubic areas before SwinIR processing
    
    # 1. Edge enhancement using PIL
    edges = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    edges_array = np.array(edges.convert('RGB'), dtype=np.float32)
    
    # Apply edge enhancement selectively
    for c in range(3):
        rgb_array[:,:,c] = np.where(
            cubic_mask,
            rgb_array[:,:,c] * 0.7 + edges_array[:,:,c] * 0.3,
            rgb_array[:,:,c]
        )
    
    # 2. Contrast boost for cubics
    if np.any(cubic_mask):
        mean_val = np.mean(rgb_array[cubic_mask])
        contrast_factor = 1.2 * intensity
        
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                np.clip((rgb_array[:,:,c] - mean_val) * contrast_factor + mean_val, 0, 255),
                rgb_array[:,:,c]
            )
    
    # 3. Highlight enhancement
    if np.any(highlights):
        boost_factor = 1.15 * intensity
        rgb_array[highlights] = np.minimum(rgb_array[highlights] * boost_factor, 255)
    
    # 4. Color cubic saturation boost
    if np.any(color_cubic):
        # Convert to HSV for saturation adjustment
        rgb_temp = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
        hsv_temp = rgb_temp.convert('HSV')
        h_temp, s_temp, v_temp = hsv_temp.split()
        
        s_array = np.array(s_temp, dtype=np.float32)
        v_array = np.array(v_temp, dtype=np.float32)
        
        # Boost saturation
        s_array = np.where(
            color_cubic,
            np.minimum(s_array * (1.4 * intensity), 255),
            s_array
        )
        
        # Boost value slightly
        v_array = np.where(
            color_cubic,
            np.minimum(v_array * 1.05, 255),
            v_array
        )
        
        # Convert back
        hsv_enhanced = Image.merge('HSV', (
            h_temp,
            Image.fromarray(s_array.astype(np.uint8)),
            Image.fromarray(v_array.astype(np.uint8))
        ))
        rgb_array = np.array(hsv_enhanced.convert('RGB'), dtype=np.float32)
    
    # Convert back to image
    rgb_enhanced = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
    r2, g2, b2 = rgb_enhanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    # Final sharpening
    sharpness = ImageEnhance.Sharpness(result)
    result = sharpness.enhance(1.0 + (0.3 * intensity))
    
    logger.info("✅ Cubic pre-enhancement complete, ready for SwinIR!")
    
    return result

def handler(event):
    """RunPod handler function - V13 Balanced with better JSON serialization"""
    logger.info(f"=== Cubic Detail Enhancement {VERSION} Started ===")
    logger.info(f"Handler received event type: {type(event)}")
    
    # Log the complete structure for debugging
    logger.info("📋 Complete event structure:")
    logger.info(json.dumps(event, indent=2)[:1000] + "...")  # First 1000 chars
    
    try:
        # Extract the actual job data from RunPod event structure
        job_input = None
        
        # Try different possible input structures
        if isinstance(event, dict):
            # Standard RunPod structure: {"input": {...}}
            if 'input' in event:
                job_input = event['input']
                logger.info("Found input in event['input']")
            # Direct input structure
            else:
                job_input = event
                logger.info("Using event directly as input")
        else:
            job_input = event
            logger.info("Event is not a dict, using as is")
        
        # Process the job
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
    """Process cubic detail enhancement - V13 Balanced with better JSON handling"""
    try:
        logger.info("🚀 Fast loading version - No OpenCV")
        logger.info("💎 SwinIR for cubic detail enhancement")
        logger.info(f"Job input type: {type(job)}")
        
        # Log job structure in detail
        if isinstance(job, dict):
            logger.info(f"Job keys: {list(job.keys())[:10]}")
            # Log each key's value type and size
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
        
        # Extract input data using more robust function
        image_data_str = find_input_data_robust(job)
        
        if not image_data_str:
            # Build detailed error message
            error_details = []
            error_details.append("No valid image data found in input.")
            
            if isinstance(job, dict):
                error_details.append(f"Available keys: {list(job.keys())}")
                
                # Check each key for potential issues
                for key, value in job.items():
                    if isinstance(value, str):
                        if len(value) == 0:
                            error_details.append(f"- '{key}' is empty string")
                        elif len(value) < 20:
                            error_details.append(f"- '{key}' too short ({len(value)} chars)")
                        else:
                            # Check if it looks like base64
                            sample = value[:50].strip()
                            if not all(c in string.ascii_letters + string.digits + '+/=' for c in sample):
                                error_details.append(f"- '{key}' doesn't look like base64")
            
            error_details.append("\nExpected fields: 'enhancement', 'enhanced_image', 'image', 'base64', etc.")
            error_details.append("Image data should be base64 encoded string.")
            
            # Special note about Make.com
            error_details.append("\nMake.com note: Check that the field mapping is correct.")
            error_details.append("The image field might be: {{4.data.output.output.enhanced_image}}")
            
            raise ValueError("\n".join(error_details))
        
        # Extract parameters
        params = job if isinstance(job, dict) else {}
        filename = params.get('filename', '')
        intensity = float(params.get('intensity', 1.0))
        intensity = max(0.1, min(2.0, intensity))
        apply_swinir = params.get('apply_swinir', True)
        apply_pattern = params.get('pattern_enhancement', True)
        
        logger.info(f"Parameters: filename={filename}, intensity={intensity}, swinir={apply_swinir}, pattern={apply_pattern}")
        
        # Decode image
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
                "ac_pattern": "무도금화이트(0.18)",  # Updated value
                "ab_pattern": "무도금화이트-쿨톤(0.14)",  # Updated value
                "other": "기타색상(0.02)"  # Updated value
            }.get(pattern_type, "기타색상")
            
            logger.info(f"Detected pattern: {pattern_type} - {detected_type}")
            image = apply_pattern_enhancement_transparent(image, pattern_type)
        else:
            pattern_type = "none"
            detected_type = "보정없음"
        
        # 3. Ring Hole Detection (Simple version)
        logger.info("🔍 Step 3: Simple ring hole detection")
        image = ensure_ring_holes_transparent_simple(image)
        
        # 4. Cubic Pre-enhancement
        logger.info("💎 Step 4: Cubic pre-enhancement")
        image = enhance_cubic_sparkle_with_swinir(image, intensity)
        
        # 5. SwinIR Enhancement (for cubic detail)
        if apply_swinir:
            logger.info("🚀 Step 5: Applying SwinIR for cubic detail")
            enhanced_image = apply_swinir_enhancement(image)
        else:
            enhanced_image = image
        
        # Encode to base64
        output_base64 = image_to_base64(enhanced_image)
        
        # Statistics
        cubic_mask, _, _, _ = detect_cubic_regions_enhanced(image)
        cubic_pixel_count = int(np.sum(cubic_mask))  # Convert to Python int
        cubic_percentage = (cubic_pixel_count / (image.size[0] * image.size[1])) * 100
        
        # CRITICAL FIX: Convert numpy bool_ to Python bool
        has_cubics = bool(cubic_pixel_count > 0)  # Explicitly convert to Python bool
        
        # RunPod expects {"output": {...}} structure
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
                    "cubic_pixels": cubic_pixel_count,  # Already converted to Python int
                    "cubic_percentage": round(cubic_percentage, 2),
                    "has_cubics": has_cubics  # Now a Python bool, not numpy bool_
                },
                "corrections_applied": [
                    "white_balance",
                    "pattern_enhancement" if apply_pattern else "pattern_skipped",
                    "ring_hole_detection_simple",
                    "cubic_pre_enhancement",
                    "swinir_detail" if apply_swinir else "swinir_skipped"
                ],
                "base64_padding": "INCLUDED",  # Always included now
                "compression": "level_3",
                "performance": "optimized_no_cv2",
                "processing_order": "1.WB → 2.Pattern → 3.RingHoles(Simple) → 4.CubicPrep → 5.SwinIR",
                "v13_improvements": [
                    "Balanced brightness for all patterns",
                    "AC pattern: 0.18 overlay, 1.01 brightness",
                    "AB pattern: 0.14 overlay, 1.01 brightness",
                    "Other patterns: 0.02 overlay, 1.03 brightness (fixed)",
                    "Fixed excessive whitening in non-AC/AB patterns",
                    "Always include base64 padding for compatibility"
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
