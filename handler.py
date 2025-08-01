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
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

################################
# CUBIC DETAIL ENHANCEMENT HANDLER V3.6
# VERSION: Cubic-Sparkle-V3.6-RunPod-Fixed
# Fixed for RunPod input structure - FINAL
################################

VERSION = "Cubic-Sparkle-V3.6-RunPod-Fixed"

# Global flags for optional features
REPLICATE_AVAILABLE = False
SCIPY_AVAILABLE = False

# Try importing optional packages
try:
    import replicate
    REPLICATE_AVAILABLE = True
    logger.info("✅ Replicate module available")
except ImportError:
    logger.warning("⚠️ Replicate not available - SwinIR will be disabled")

try:
    from scipy.ndimage import label, center_of_mass
    SCIPY_AVAILABLE = True
    logger.info("✅ Scipy module available")
except ImportError:
    logger.warning("⚠️ Scipy not available - Advanced sparkle disabled")

def decode_base64_fast(base64_str: str) -> bytes:
    """Fast base64 decode with padding handling"""
    try:
        if not base64_str:
            raise ValueError("Empty base64 string")
            
        if len(base64_str) < 50:
            raise ValueError(f"Base64 string too short: {len(base64_str)} chars")
        
        # Remove data URL prefix
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        # Remove all whitespace
        base64_str = ''.join(base64_str.split())
        
        # Filter only valid base64 characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        filtered = ''.join(c for c in base64_str if c in valid_chars)
        
        if len(filtered) < 50:
            raise ValueError(f"Filtered base64 too short: {len(filtered)} chars")
        
        # Try with existing padding first
        try:
            decoded = base64.b64decode(filtered, validate=True)
            logger.info(f"✅ Decoded with existing padding (length: {len(decoded)})")
            return decoded
        except:
            # Fix padding if needed
            no_pad = filtered.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            decoded = base64.b64decode(padded, validate=True)
            logger.info(f"✅ Decoded after padding fix (length: {len(decoded)})")
            return decoded
            
    except Exception as e:
        logger.error(f"❌ Base64 decode failed: {str(e)}")
        raise ValueError(f"Base64 decode error: {str(e)}")

def image_to_base64(image: Image.Image) -> str:
    """Convert image to base64 with proper padding"""
    try:
        if image.mode != 'RGBA':
            logger.info(f"Converting {image.mode} to RGBA")
            image = image.convert('RGBA')
        
        buffered = BytesIO()
        image.save(buffered, format='PNG', compress_level=3, optimize=True)
        buffered.seek(0)
        
        base64_bytes = base64.b64encode(buffered.getvalue())
        base64_str = base64_bytes.decode('utf-8')
        
        logger.info(f"✅ Encoded to base64 (length: {len(base64_str)})")
        return base64_str
        
    except Exception as e:
        logger.error(f"❌ Image encoding failed: {str(e)}")
        raise

def find_input_data(data):
    """Find image data in various input formats"""
    logger.info(f"🔍 Searching for input data (type: {type(data)})")
    
    # Direct string
    if isinstance(data, str):
        if len(data) > 50:
            logger.info(f"✅ Found string input (length: {len(data)})")
            return data
        logger.warning(f"String too short: {len(data)}")
        return None
    
    # Dictionary
    if isinstance(data, dict):
        logger.info(f"📋 Dict keys: {list(data.keys())}")
        
        # Priority keys - exact order matters!
        image_keys = ['enhanced_image', 'thumbnail', 'image', 'image_base64', 'base64', 'img', 'input', 'data']
        
        # Direct check
        for key in image_keys:
            if key in data:
                value = data[key]
                if isinstance(value, str) and len(value) > 50:
                    logger.info(f"✅ Found in '{key}' (length: {len(value)})")
                    return value
        
        # Special handling for Make.com numbered keys (0, 1, 2, etc.)
        for i in range(10):
            key = str(i)
            if key in data:
                value = data[key]
                if isinstance(value, str) and len(value) > 50:
                    logger.info(f"✅ Found in numbered key '{key}' (length: {len(value)})")
                    return value
        
        # Check for base64 data in ANY key
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 1000:
                # Quick base64 check (first 100 chars)
                sample = value[:100].strip()
                if all(c in string.ascii_letters + string.digits + '+/=' for c in sample):
                    logger.info(f"✅ Found potential base64 in '{key}' (length: {len(value)})")
                    return value
        
        # Log all keys and their types for debugging
        logger.error("❌ No valid image data found in dictionary")
        logger.error("Available data:")
        for key, value in list(data.items())[:10]:  # First 10 items
            if isinstance(value, str):
                logger.error(f"  '{key}': string ({len(value)} chars)")
                if len(value) < 100:
                    logger.error(f"    Preview: {value[:50]}...")
            elif isinstance(value, dict):
                logger.error(f"  '{key}': dict with keys {list(value.keys())}")
            elif isinstance(value, list):
                logger.error(f"  '{key}': list with {len(value)} items")
            else:
                logger.error(f"  '{key}': {type(value)}")
    
    logger.error("❌ No image data found")
    return None

def detect_pattern_type(filename: str) -> str:
    """Detect color pattern from filename"""
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
    """Fast white balance correction"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        rgb_img = Image.merge('RGB', (r, g, b))
        
        # Convert to numpy
        img_array = np.array(rgb_img, dtype=np.float32)
        
        # Sample pixels for gray reference
        sampled = img_array[::15, ::15]
        
        # Find near-gray pixels
        gray_mask = (
            (np.abs(sampled[:,:,0] - sampled[:,:,1]) < 15) & 
            (np.abs(sampled[:,:,1] - sampled[:,:,2]) < 15) &
            (sampled[:,:,0] > 180)
        )
        
        if np.sum(gray_mask) > 10:
            # Calculate corrections
            r_avg = np.mean(sampled[gray_mask, 0])
            g_avg = np.mean(sampled[gray_mask, 1])
            b_avg = np.mean(sampled[gray_mask, 2])
            
            gray_avg = (r_avg + g_avg + b_avg) / 3
            
            # Apply corrections
            if r_avg > 0:
                img_array[:,:,0] *= (gray_avg / r_avg)
            if g_avg > 0:
                img_array[:,:,1] *= (gray_avg / g_avg)
            if b_avg > 0:
                img_array[:,:,2] *= (gray_avg / b_avg)
        
        # Convert back
        img_array = np.clip(img_array, 0, 255)
        rgb_balanced = Image.fromarray(img_array.astype(np.uint8))
        
        # Merge with alpha
        r2, g2, b2 = rgb_balanced.split()
        result = Image.merge('RGBA', (r2, g2, b2, a))
        
        return result
        
    except Exception as e:
        logger.error(f"White balance error: {e}")
        return image

def apply_pattern_enhancement_transparent(image: Image.Image, pattern_type: str) -> Image.Image:
    """Apply color pattern enhancement"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        img_array = np.array(rgb_image, dtype=np.float32)
        
        if pattern_type == "ac_pattern":
            # AC - 무도금화이트 (20% white)
            logger.info("🎨 AC Pattern - 무도금화이트")
            white_overlay = 0.20
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            img_array = np.clip(img_array, 0, 255)
            
            rgb_image = Image.fromarray(img_array.astype(np.uint8))
            
            # Adjustments
            brightness = ImageEnhance.Brightness(rgb_image)
            rgb_image = brightness.enhance(1.03)
            
            color = ImageEnhance.Color(rgb_image)
            rgb_image = color.enhance(0.98)
            
        elif pattern_type == "ab_pattern":
            # AB - 무도금화이트 쿨톤 (16% white)
            logger.info("🎨 AB Pattern - 무도금화이트 쿨톤")
            white_overlay = 0.16
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            
            # Cool tone
            img_array[:,:,0] *= 0.96
            img_array[:,:,1] *= 0.98
            img_array[:,:,2] *= 1.02
            
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
            # Other - 기타색상 (5% white)
            logger.info("🎨 Other Pattern - 기타색상")
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
        
        # Common adjustments
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(1.1)
        
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.8)
        
        # Merge back
        r2, g2, b2 = rgb_image.split()
        result = Image.merge('RGBA', (r2, g2, b2, a))
        
        return result
        
    except Exception as e:
        logger.error(f"Pattern enhancement error: {e}")
        return image

def ensure_ring_holes_transparent_simple(image: Image.Image) -> Image.Image:
    """Detect and make ring holes transparent"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        
        # Convert to HSV
        hsv = image.convert('HSV')
        h, s, v = hsv.split()
        
        # Arrays
        s_array = np.array(s)
        v_array = np.array(v)
        alpha_array = np.array(a, dtype=np.uint8)
        
        # Ring holes: very bright, low saturation
        holes_mask = (v_array > 248) & (s_array < 20) & (alpha_array > 200)
        
        # Simple morphology
        holes_image = Image.fromarray((holes_mask * 255).astype(np.uint8))
        holes_image = holes_image.filter(ImageFilter.MinFilter(3))
        holes_image = holes_image.filter(ImageFilter.MaxFilter(3))
        holes_mask = np.array(holes_image) > 128
        
        # Apply transparency
        alpha_array[holes_mask] = 0
        
        # Rebuild image
        a_new = Image.fromarray(alpha_array)
        result = Image.merge('RGBA', (r, g, b, a_new))
        
        hole_count = np.sum(holes_mask)
        logger.info(f"✅ Ring holes: {hole_count} pixels")
        
        return result
        
    except Exception as e:
        logger.error(f"Ring hole detection error: {e}")
        return image

def detect_cubic_regions_enhanced(image: Image.Image, sensitivity=1.0):
    """Detect cubic/crystal regions"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get alpha channel
        alpha_array = np.array(image.split()[3], dtype=np.uint8)
        
        # Convert to HSV
        hsv = image.convert('HSV')
        h, s, v = hsv.split()
        s_array = np.array(s)
        v_array = np.array(v)
        
        # Edge detection
        edges = image.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges.convert('L'))
        
        # Different cubic types
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
        
        # Combine
        cubic_mask = white_cubic | color_cubic | edge_cubic | highlights
        
        # Clean up
        cubic_image = Image.fromarray((cubic_mask * 255).astype(np.uint8))
        cubic_image = cubic_image.filter(ImageFilter.MinFilter(3))
        cubic_image = cubic_image.filter(ImageFilter.MaxFilter(3))
        cubic_mask = np.array(cubic_image) > 128
        
        return cubic_mask.astype(bool), white_cubic, color_cubic, highlights
        
    except Exception as e:
        logger.error(f"Cubic detection error: {e}")
        # Return empty masks
        shape = (image.size[1], image.size[0])
        empty = np.zeros(shape, dtype=bool)
        return empty, empty, empty, empty

def enhance_cubic_sparkle_simple(image: Image.Image, intensity=1.0) -> Image.Image:
    """Simple cubic enhancement without scipy"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Detect cubics
        cubic_mask, white_cubic, color_cubic, highlights = detect_cubic_regions_enhanced(image, intensity)
        
        cubic_count = np.sum(cubic_mask)
        total_pixels = image.size[0] * image.size[1]
        cubic_percent = (cubic_count / total_pixels) * 100
        
        logger.info(f"💎 Cubic pixels: {cubic_count} ({cubic_percent:.1f}%)")
        
        if cubic_count == 0:
            return image
        
        # Work with RGB
        r, g, b, a = image.split()
        rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
        
        # 1. Edge enhancement
        edges = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges_array = np.array(edges.convert('RGB'), dtype=np.float32)
        
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                rgb_array[:,:,c] * 0.7 + edges_array[:,:,c] * 0.3,
                rgb_array[:,:,c]
            )
        
        # 2. Contrast
        if np.any(cubic_mask):
            mean_val = np.mean(rgb_array[cubic_mask])
            contrast_factor = 1.2 * intensity
            
            for c in range(3):
                rgb_array[:,:,c] = np.where(
                    cubic_mask,
                    np.clip((rgb_array[:,:,c] - mean_val) * contrast_factor + mean_val, 0, 255),
                    rgb_array[:,:,c]
                )
        
        # 3. Highlights
        if np.any(highlights):
            boost = 1.15 * intensity
            for c in range(3):
                rgb_array[:,:,c] = np.where(
                    highlights,
                    np.minimum(rgb_array[:,:,c] * boost, 255),
                    rgb_array[:,:,c]
                )
        
        # Convert back
        rgb_enhanced = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
        r2, g2, b2 = rgb_enhanced.split()
        result = Image.merge('RGBA', (r2, g2, b2, a))
        
        # Sharpness
        sharpness = ImageEnhance.Sharpness(result)
        result = sharpness.enhance(1.0 + (0.3 * intensity))
        
        return result
        
    except Exception as e:
        logger.error(f"Cubic enhancement error: {e}")
        return image

def apply_swinir_enhancement(image: Image.Image) -> Image.Image:
    """Apply SwinIR if available"""
    if not REPLICATE_AVAILABLE:
        logger.info("⏭️ SwinIR skipped (replicate not available)")
        return image
    
    try:
        api_token = os.environ.get('REPLICATE_API_TOKEN')
        if not api_token:
            logger.warning("⏭️ SwinIR skipped (no API token)")
            return image
        
        logger.info("🚀 Applying SwinIR enhancement...")
        
        client = replicate.Client(api_token=api_token)
        
        # Prepare image
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        # To base64
        buffered = BytesIO()
        rgb_image.save(buffered, format="PNG", optimize=True)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_data_url = f"data:image/png;base64,{img_base64}"
        
        # Run model
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
            
            # Merge with alpha
            r2, g2, b2 = enhanced_image.split()
            result = Image.merge('RGBA', (r2, g2, b2, a))
            
            logger.info("✅ SwinIR completed")
            return result
            
    except Exception as e:
        logger.error(f"SwinIR error: {e}")
    
    return image

def process_cubic_enhancement(job_input):
    """Main processing function"""
    try:
        logger.info("="*50)
        logger.info("🚀 CUBIC DETAIL ENHANCEMENT START")
        logger.info("="*50)
        
        # Log input structure for debugging
        if isinstance(job_input, dict):
            logger.info(f"📋 Input is dict with keys: {list(job_input.keys())}")
        elif isinstance(job_input, str):
            logger.info(f"📄 Input is string with {len(job_input)} chars")
        else:
            logger.info(f"❓ Input type: {type(job_input)}")
        
        # Find image data
        image_data_str = find_input_data(job_input)
        
        if not image_data_str:
            error_msg = "No valid image data found"
            logger.error(f"❌ {error_msg}")
            
            # Detailed debug info
            debug_info = {
                "input_type": str(type(job_input)),
                "expected_keys": ["enhanced_image", "thumbnail", "image", "image_base64"],
                "received_keys": list(job_input.keys()) if isinstance(job_input, dict) else "Not a dict",
                "make_com_format": {
                    "input": {
                        "enhanced_image": "{{4.data.output.output.enhanced_image}}",
                        "filename": "optional",
                        "intensity": 1.0
                    }
                }
            }
            
            return {
                "output": {
                    "error": error_msg,
                    "status": "failed",
                    "version": VERSION,
                    "debug": debug_info
                }
            }
        
        # Parameters
        filename = ''
        intensity = 1.0
        apply_swinir = True
        apply_pattern = True
        
        if isinstance(job_input, dict):
            filename = job_input.get('filename', '')
            intensity = float(job_input.get('intensity', 1.0))
            apply_swinir = job_input.get('apply_swinir', True)
            apply_pattern = job_input.get('pattern_enhancement', True)
        
        intensity = max(0.1, min(2.0, intensity))
        
        logger.info("📋 PARAMETERS:")
        logger.info(f"  Filename: {filename}")
        logger.info(f"  Intensity: {intensity}")
        logger.info(f"  SwinIR: {apply_swinir}")
        logger.info(f"  Pattern: {apply_pattern}")
        
        # Decode image
        logger.info("🖼️ Decoding image...")
        image_bytes = decode_base64_fast(image_data_str)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        logger.info(f"📐 Image: {image.size[0]}x{image.size[1]} {image.mode}")
        
        # Processing pipeline
        logger.info("="*30)
        logger.info("🔧 PROCESSING PIPELINE")
        logger.info("="*30)
        
        # 1. White Balance
        logger.info("⚖️ [1/5] White Balance...")
        image = auto_white_balance_fast(image)
        
        # 2. Pattern
        if apply_pattern:
            logger.info("🎨 [2/5] Pattern Enhancement...")
            pattern_type = detect_pattern_type(filename)
            detected_type = {
                "ac_pattern": "무도금화이트(20%)",
                "ab_pattern": "무도금화이트-쿨톤(16%)",
                "other": "기타색상(5%)"
            }.get(pattern_type, "Unknown")
            
            logger.info(f"  Type: {pattern_type} - {detected_type}")
            image = apply_pattern_enhancement_transparent(image, pattern_type)
        else:
            logger.info("⏭️ [2/5] Pattern Enhancement (skipped)")
            pattern_type = "none"
            detected_type = "보정없음"
        
        # 3. Ring Holes
        logger.info("🔍 [3/5] Ring Hole Detection...")
        image = ensure_ring_holes_transparent_simple(image)
        
        # 4. Cubic Enhancement
        logger.info("💎 [4/5] Cubic Enhancement...")
        image = enhance_cubic_sparkle_simple(image, intensity)
        
        # 5. SwinIR
        if apply_swinir and REPLICATE_AVAILABLE:
            logger.info("🚀 [5/5] SwinIR Enhancement...")
            enhanced_image = apply_swinir_enhancement(image)
        else:
            logger.info("⏭️ [5/5] SwinIR Enhancement (skipped)")
            enhanced_image = image
        
        # Final encoding
        logger.info("📤 Encoding result...")
        output_base64 = image_to_base64(enhanced_image)
        
        # Statistics
        cubic_mask, _, _, _ = detect_cubic_regions_enhanced(enhanced_image)
        cubic_pixel_count = int(np.sum(cubic_mask))
        total_pixels = enhanced_image.size[0] * enhanced_image.size[1]
        cubic_percentage = (cubic_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Response
        logger.info("="*30)
        logger.info("✅ PROCESSING COMPLETE")
        logger.info(f"📊 Cubic pixels: {cubic_pixel_count} ({cubic_percentage:.1f}%)")
        logger.info(f"📐 Output size: {enhanced_image.size}")
        logger.info("="*30)
        
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
                    "cubic_pixels": cubic_pixel_count,
                    "cubic_percentage": round(cubic_percentage, 2),
                    "has_cubics": cubic_pixel_count > 0,
                    "total_pixels": total_pixels
                },
                "corrections_applied": [
                    "white_balance",
                    f"pattern_{pattern_type}" if apply_pattern else "pattern_skipped",
                    "ring_hole_detection",
                    f"cubic_enhancement_{intensity}",
                    "swinir" if (apply_swinir and REPLICATE_AVAILABLE) else "swinir_skipped"
                ],
                "base64_padding": "INCLUDED",
                "compression": "PNG_LEVEL_3"
            }
        }
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        
        logger.error("="*50)
        logger.error("❌ PROCESSING FAILED")
        logger.error(f"Error: {error_msg}")
        logger.error("Traceback:")
        logger.error(tb)
        logger.error("="*50)
        
        return {
            "output": {
                "error": error_msg,
                "status": "failed",
                "version": VERSION,
                "traceback": tb
            }
        }

def handler(event):
    """RunPod handler - main entry point"""
    try:
        logger.info(f"🎯 RunPod Handler v{VERSION}")
        logger.info(f"📦 Modules: Replicate={REPLICATE_AVAILABLE}, Scipy={SCIPY_AVAILABLE}")
        logger.info(f"📥 Event type: {type(event)}")
        
        # Critical: Log raw event for debugging
        if isinstance(event, dict):
            logger.info(f"📋 Event keys: {list(event.keys())}")
            if len(event) < 10:  # Small enough to log fully
                logger.info(f"📄 Full event: {event}")
        
        # RunPod ALWAYS sends data in {"input": {...}} format
        # This is the ONLY valid structure
        
        if not isinstance(event, dict):
            raise ValueError(f"Event must be a dict, got {type(event)}")
        
        if 'input' not in event:
            # This is THE error we keep hitting
            logger.error("❌ CRITICAL: No 'input' field in event!")
            logger.error(f"Available keys: {list(event.keys())}")
            logger.error("RunPod requires {'input': {...}} structure")
            
            # Log sample of each key for debugging
            for key in list(event.keys())[:3]:
                value = event[key]
                if isinstance(value, str):
                    logger.error(f"  {key}: string of {len(value)} chars")
                elif isinstance(value, dict):
                    logger.error(f"  {key}: dict with keys {list(value.keys())}")
                else:
                    logger.error(f"  {key}: {type(value)}")
            
            raise ValueError("Event must contain 'input' field. RunPod structure: {'input': {...}}")
        
        # Extract job_input from event['input']
        job_input = event['input']
        logger.info("✅ Successfully found event['input']")
        
        # Log job_input structure
        if isinstance(job_input, dict):
            logger.info(f"📋 job_input keys: {list(job_input.keys())}")
            # Check if we have image data
            for key in ['enhanced_image', 'image', 'thumbnail']:
                if key in job_input and isinstance(job_input[key], str):
                    logger.info(f"✅ Found '{key}' with {len(job_input[key])} chars")
        elif isinstance(job_input, str):
            logger.info(f"📄 job_input is string with {len(job_input)} chars")
        else:
            logger.info(f"❓ job_input type: {type(job_input)}")
        
        # Process
        return process_cubic_enhancement(job_input)
        
    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        
        logger.error(f"❌ Handler error: {error_msg}")
        logger.error(f"Traceback:\n{tb}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "failed", 
                "version": VERSION,
                "traceback": tb,
                "expected_structure": {"input": {"enhanced_image": "base64_string", "filename": "optional", "intensity": 1.0}}
            }
        }

# RunPod entry point
if __name__ == "__main__":
    logger.info(f"🚀 Starting Cubic Detail Enhancement v{VERSION}")
    logger.info(f"📦 Available modules: Replicate={REPLICATE_AVAILABLE}, Scipy={SCIPY_AVAILABLE}")
    logger.info("⚠️ CRITICAL: RunPod requires {'input': {...}} structure")
    runpod.serverless.start({"handler": handler})
