import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import string
import json
import traceback
import sys

# Minimal logging - RunPod doesn't like too much logging
def log(msg):
    print(msg, flush=True)

################################
# CUBIC DETAIL ENHANCEMENT HANDLER V7.1 - HEALTH CHECK FIX
# VERSION: Cubic-Sparkle-V7.1-HealthCheckFix
# Fixed RunPod health check and entry point
################################

VERSION = "Cubic-Sparkle-V7.1-HealthCheckFix"

def decode_base64_fast(base64_str: str) -> bytes:
    """Fast base64 decode with minimal logging"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
            
        # Remove data URL prefix
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        # Remove whitespace
        base64_str = ''.join(base64_str.split())
        
        # Filter valid characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        filtered = ''.join(c for c in base64_str if c in valid_chars)
        
        # Try with existing padding first
        try:
            return base64.b64decode(filtered, validate=True)
        except:
            # Fix padding if needed
            no_pad = filtered.rstrip('=')
            padding_needed = (4 - len(no_pad) % 4) % 4
            padded = no_pad + ('=' * padding_needed)
            return base64.b64decode(padded, validate=True)
            
    except Exception as e:
        raise ValueError(f"Base64 decode error: {str(e)}")

def image_to_base64(image: Image.Image) -> str:
    """Convert image to base64"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        buffered = BytesIO()
        image.save(buffered, format='PNG', compress_level=3, optimize=True)
        buffered.seek(0)
        
        base64_bytes = base64.b64encode(buffered.getvalue())
        return base64_bytes.decode('utf-8')
        
    except Exception as e:
        raise ValueError(f"Image encoding failed: {str(e)}")

def find_input_data(data):
    """Find image data in various input formats"""
    log(f"find_input_data: Looking for image in {type(data)}")
    
    # Direct string
    if isinstance(data, str) and len(data) > 50:
        log(f"Found direct string image data: {len(data)} chars")
        return data
    
    # Dictionary
    if isinstance(data, dict):
        log(f"Input is dict with keys: {list(data.keys())}")
        
        # Priority keys
        image_keys = ['enhanced_image', 'thumbnail', 'image', 'image_base64', 'base64', 'img', 'input', 'data']
        
        for key in image_keys:
            if key in data:
                value = data[key]
                if isinstance(value, str) and len(value) > 50:
                    log(f"Found image data in key '{key}': {len(value)} chars")
                    return value
                else:
                    log(f"Key '{key}' exists but not valid: type={type(value)}, len={len(str(value))}")
        
        # Check numbered keys
        for i in range(10):
            key = str(i)
            if key in data:
                value = data[key]
                if isinstance(value, str) and len(value) > 50:
                    log(f"Found image data in numbered key '{key}': {len(value)} chars")
                    return value
    
    log("❌ No valid image data found in any expected location")
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

def auto_white_balance_extreme(image: Image.Image) -> Image.Image:
    """EXTREME white balance correction - memory efficient"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        rgb_img = Image.merge('RGB', (r, g, b))
        
        # Work in-place when possible
        img_array = np.array(rgb_img, dtype=np.float32)
        
        # Gray world assumption
        for c in range(3):
            channel_mean = np.mean(img_array[:,:,c])
            if channel_mean > 0:
                img_array[:,:,c] *= (128 / channel_mean)
        
        # Sample-based correction
        sampled = img_array[::10, ::10]
        gray_mask = (
            (np.abs(sampled[:,:,0] - sampled[:,:,1]) < 20) & 
            (np.abs(sampled[:,:,1] - sampled[:,:,2]) < 20) &
            (sampled[:,:,0] > 150) & (sampled[:,:,0] < 250)
        )
        
        if np.sum(gray_mask) > 5:
            r_avg = np.mean(sampled[gray_mask, 0])
            g_avg = np.mean(sampled[gray_mask, 1])
            b_avg = np.mean(sampled[gray_mask, 2])
            
            gray_avg = (r_avg + g_avg + b_avg) / 3
            
            if r_avg > 0:
                img_array[:,:,0] *= min(1.5, gray_avg / r_avg)
            if g_avg > 0:
                img_array[:,:,1] *= min(1.5, gray_avg / g_avg)
            if b_avg > 0:
                img_array[:,:,2] *= min(1.5, gray_avg / b_avg)
        
        # Clip and convert
        np.clip(img_array, 0, 255, out=img_array)
        rgb_balanced = Image.fromarray(img_array.astype(np.uint8))
        
        # Fine-tune
        color = ImageEnhance.Color(rgb_balanced)
        rgb_balanced = color.enhance(1.1)
        
        # Merge with alpha
        r2, g2, b2 = rgb_balanced.split()
        return Image.merge('RGBA', (r2, g2, b2, a))
        
    except Exception:
        return image

def apply_pattern_enhancement_extreme(image: Image.Image, pattern_type: str, intensity: float = 1.0) -> Image.Image:
    """Pattern enhancement with REDUCED contrast"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        # Detail extraction
        detail_layer1 = rgb_image.filter(ImageFilter.DETAIL)
        detail_layer2 = rgb_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        img_array = np.array(rgb_image, dtype=np.float32)
        detail_array1 = np.array(detail_layer1, dtype=np.float32)
        detail_array2 = np.array(detail_layer2, dtype=np.float32)
        
        if pattern_type == "ac_pattern":
            # AC - Unplated white (25% white)
            white_overlay = 0.25 * intensity
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            
            # Detail mixing
            img_array = (
                img_array * 0.50 + 
                detail_array1 * 0.25 + 
                detail_array2 * 0.25
            )
            
            np.clip(img_array, 0, 255, out=img_array)
            rgb_image = Image.fromarray(img_array.astype(np.uint8))
            
            # Adjustments with REDUCED contrast
            brightness = ImageEnhance.Brightness(rgb_image)
            rgb_image = brightness.enhance(1.08)
            
            color = ImageEnhance.Color(rgb_image)
            rgb_image = color.enhance(0.92)
            
            # REDUCED contrast (was 1.25)
            contrast = ImageEnhance.Contrast(rgb_image)
            rgb_image = contrast.enhance(1.15)
            
            sharpness = ImageEnhance.Sharpness(rgb_image)
            rgb_image = sharpness.enhance(2.2)
            
        elif pattern_type == "ab_pattern":
            # AB - Unplated white cool tone (20% white)
            white_overlay = 0.20 * intensity
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            
            # Cool tone
            img_array[:,:,0] *= 0.92
            img_array[:,:,1] *= 0.96
            img_array[:,:,2] *= 1.05
            
            # Cool overlay
            cool_overlay = np.array([235, 245, 255], dtype=np.float32)
            img_array = img_array * 0.88 + cool_overlay * 0.12
            
            # Detail mixing
            img_array = (
                img_array * 0.50 + 
                detail_array1 * 0.25 + 
                detail_array2 * 0.25
            )
            
            np.clip(img_array, 0, 255, out=img_array)
            rgb_image = Image.fromarray(img_array.astype(np.uint8))
            
            color = ImageEnhance.Color(rgb_image)
            rgb_image = color.enhance(0.82)
            
            brightness = ImageEnhance.Brightness(rgb_image)
            rgb_image = brightness.enhance(1.06)
            
            # REDUCED contrast (was 1.22)
            contrast = ImageEnhance.Contrast(rgb_image)
            rgb_image = contrast.enhance(1.12)
            
            sharpness = ImageEnhance.Sharpness(rgb_image)
            rgb_image = sharpness.enhance(2.5)
            
        else:
            # Other - General colors (12% white)
            white_overlay = 0.12 * intensity
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            
            # Detail mixing
            img_array = (
                img_array * 0.50 + 
                detail_array1 * 0.25 + 
                detail_array2 * 0.25
            )
            
            np.clip(img_array, 0, 255, out=img_array)
            rgb_image = Image.fromarray(img_array.astype(np.uint8))
            
            brightness = ImageEnhance.Brightness(rgb_image)
            rgb_image = brightness.enhance(1.15)
            
            color = ImageEnhance.Color(rgb_image)
            rgb_image = color.enhance(1.08)
            
            # REDUCED contrast (was 1.3)
            contrast = ImageEnhance.Contrast(rgb_image)
            rgb_image = contrast.enhance(1.18)
            
            sharpness = ImageEnhance.Sharpness(rgb_image)
            rgb_image = sharpness.enhance(2.8)
        
        # Common final pass with REDUCED contrast
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(1.1)  # Was 1.2
        
        # Final sharpness
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(1.8)
        
        # Merge back
        r2, g2, b2 = rgb_image.split()
        return Image.merge('RGBA', (r2, g2, b2, a))
        
    except Exception:
        return image

def ensure_ring_holes_transparent_extreme(image: Image.Image) -> Image.Image:
    """Ring hole detection - memory efficient"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        
        # Arrays
        r_array = np.array(r, dtype=np.uint8)
        g_array = np.array(g, dtype=np.uint8)
        b_array = np.array(b, dtype=np.uint8)
        alpha_array = np.array(a, dtype=np.uint8)
        
        # Pure white detection
        pure_white = (r_array > 245) & (g_array > 245) & (b_array > 245)
        
        # Near-white detection
        near_white = (
            (r_array > 235) & (g_array > 235) & (b_array > 235) &
            (np.abs(r_array.astype(np.int16) - g_array.astype(np.int16)) < 10) &
            (np.abs(g_array.astype(np.int16) - b_array.astype(np.int16)) < 10)
        )
        
        # Combine masks
        holes_mask = pure_white | near_white
        
        # Apply transparency
        alpha_array[holes_mask] = 0
        
        # Simple feathering
        alpha_temp = Image.fromarray(alpha_array)
        alpha_temp = alpha_temp.filter(ImageFilter.GaussianBlur(1))
        alpha_array = np.array(alpha_temp)
        
        # Rebuild image
        a_new = Image.fromarray(alpha_array)
        return Image.merge('RGBA', (r, g, b, a_new))
        
    except Exception:
        return image

def detect_cubic_regions_extreme(image: Image.Image, sensitivity=1.5):
    """Simplified cubic detection"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get alpha
        alpha_array = np.array(image.split()[3], dtype=np.uint8)
        
        # Convert to HSV
        hsv = image.convert('HSV')
        h, s, v = hsv.split()
        s_array = np.array(s, dtype=np.uint8)
        v_array = np.array(v, dtype=np.uint8)
        
        # Edge detection
        edges = image.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges.convert('L'), dtype=np.uint8)
        
        # Cubic types
        white_cubic = (
            (v_array > 220) & 
            (s_array < 45) & 
            (alpha_array > 150)
        )
        
        color_cubic = (
            (v_array > 180) & 
            (s_array > 70) & 
            (alpha_array > 150)
        )
        
        edge_cubic = (
            (edges_array > 60) & 
            (v_array > 190) & 
            (alpha_array > 150)
        )
        
        highlights = (
            (v_array > 235) & 
            (s_array < 80) &
            (alpha_array > 150)
        )
        
        # Combine
        cubic_mask = white_cubic | color_cubic | edge_cubic | highlights
        
        return cubic_mask, white_cubic, color_cubic, highlights
        
    except Exception:
        shape = (image.size[1], image.size[0])
        empty = np.zeros(shape, dtype=bool)
        return empty, empty, empty, empty

def enhance_cubic_sparkle_extreme(image: Image.Image, intensity=1.5) -> Image.Image:
    """Cubic enhancement with REDUCED contrast"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Detect cubics
        cubic_mask, white_cubic, color_cubic, highlights = detect_cubic_regions_extreme(image, intensity)
        
        cubic_count = np.sum(cubic_mask)
        if cubic_count == 0:
            # Apply general enhancement
            sharpness = ImageEnhance.Sharpness(image)
            return sharpness.enhance(1.5 * intensity)
        
        # Work with RGB
        r, g, b, a = image.split()
        rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
        
        # Edge processing
        edges = image.filter(ImageFilter.FIND_EDGES)
        edges_enhanced = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges_array = np.array(edges_enhanced.convert('RGB'), dtype=np.float32)
        
        # Detail layer
        detail = image.filter(ImageFilter.DETAIL)
        detail_array = np.array(detail.convert('RGB'), dtype=np.float32)
        
        # Blend for cubic regions
        mask_3d = np.stack([cubic_mask] * 3, axis=2)
        rgb_array = np.where(
            mask_3d,
            rgb_array * 0.40 + edges_array * 0.30 + detail_array * 0.30,
            rgb_array
        )
        
        # Local contrast with REDUCED values
        if np.any(cubic_mask):
            for c in range(3):
                channel = rgb_array[:,:,c]
                
                # Process in blocks
                for y in range(0, channel.shape[0], 100):
                    for x in range(0, channel.shape[1], 100):
                        y_end = min(y + 100, channel.shape[0])
                        x_end = min(x + 100, channel.shape[1])
                        
                        local_mask = cubic_mask[y:y_end, x:x_end]
                        if np.any(local_mask):
                            local_data = channel[y:y_end, x:x_end]
                            local_mean = np.mean(local_data[local_mask])
                            
                            # REDUCED contrast factor (was 1.5)
                            contrast_factor = 1.2 * intensity
                            
                            channel[y:y_end, x:x_end] = np.where(
                                local_mask,
                                np.clip((local_data - local_mean) * contrast_factor + local_mean, 0, 255),
                                local_data
                            )
                
                rgb_array[:,:,c] = channel
        
        # Highlights with REDUCED boost
        if np.any(highlights):
            boost = 1.2 * intensity  # Was 1.3
            mask_3d = np.stack([highlights] * 3, axis=2)
            rgb_array = np.where(
                mask_3d,
                np.minimum(rgb_array * boost, 255),
                rgb_array
            )
        
        # Convert back
        np.clip(rgb_array, 0, 255, out=rgb_array)
        rgb_enhanced = Image.fromarray(rgb_array.astype(np.uint8))
        r2, g2, b2 = rgb_enhanced.split()
        result = Image.merge('RGBA', (r2, g2, b2, a))
        
        # Final sharpening with moderate values
        sharpness = ImageEnhance.Sharpness(result)
        result = sharpness.enhance(1.5 + (0.3 * intensity))
        
        # REDUCED final contrast (was 1.1)
        contrast = ImageEnhance.Contrast(result)
        result = contrast.enhance(1.05 + (0.05 * intensity))
        
        return result
        
    except Exception:
        return image

def apply_extreme_final_enhancement(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Final enhancement with REDUCED contrast"""
    try:
        # Clarity
        clarity = image.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
        
        # REDUCED contrast (was 1.05)
        contrast = ImageEnhance.Contrast(clarity)
        enhanced = contrast.enhance(1.03 * intensity)
        
        # Sharpness
        sharpness = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpness.enhance(1.15 * intensity)
        
        # Color
        color = ImageEnhance.Color(enhanced)
        enhanced = color.enhance(1.05 * intensity)
        
        return enhanced
        
    except Exception:
        return image

def apply_target_resize(image: Image.Image, target_width: int = None, target_height: int = None) -> Image.Image:
    """Resize image to target dimensions"""
    try:
        current_width, current_height = image.size
        
        # Both dimensions provided
        if target_width and target_height:
            resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
        # Only width
        elif target_width:
            ratio = target_width / current_width
            new_height = int(current_height * ratio)
            resized = image.resize((target_width, new_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
        # Only height
        elif target_height:
            ratio = target_height / current_height
            new_width = int(current_width * ratio)
            resized = image.resize((new_width, target_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
        else:
            return image
        
        # Sharpening after resize
        scale_factor = max(resized.width / current_width, resized.height / current_height)
        sharp_strength = 1.2 + (0.2 * min(scale_factor, 2))
        
        sharpness = ImageEnhance.Sharpness(resized)
        resized = sharpness.enhance(sharp_strength)
        
        return resized
        
    except Exception:
        return image

def process_cubic_enhancement(job_input):
    """Main processing function"""
    try:
        log("=== Starting process_cubic_enhancement ===")
        
        # Find image data
        log("Step 1: Finding image data...")
        image_data_str = find_input_data(job_input)
        
        if not image_data_str:
            log("ERROR: No valid image data found")
            return {
                "output": {
                    "error": "No valid image data found",
                    "status": "failed",
                    "version": VERSION,
                    "debug_info": {
                        "input_type": str(type(job_input)),
                        "input_keys": list(job_input.keys()) if isinstance(job_input, dict) else None,
                        "input_sample": str(job_input)[:100] if job_input else None
                    }
                }
            }
        
        log(f"Found image data: {len(image_data_str)} characters")
        
        # Parameters
        filename = ''
        intensity = 1.5
        apply_pattern = True
        target_width = None
        target_height = None
        
        if isinstance(job_input, dict):
            filename = job_input.get('filename', '')
            intensity = float(job_input.get('intensity', 1.5))
            apply_pattern = job_input.get('pattern_enhancement', True)
            target_width = job_input.get('target_width', None)
            target_height = job_input.get('target_height', None)
            log(f"Parameters: filename={filename}, intensity={intensity}, pattern={apply_pattern}")
        
        # Validate parameters
        intensity = max(0.5, min(3.0, intensity))
        
        if target_width:
            target_width = int(target_width)
            target_width = max(100, min(8192, target_width))
        if target_height:
            target_height = int(target_height)
            target_height = max(100, min(8192, target_height))
        
        # Decode image
        log("Step 2: Decoding base64 image...")
        image_bytes = decode_base64_fast(image_data_str)
        log(f"Decoded {len(image_bytes)} bytes")
        
        log("Step 3: Opening image with PIL...")
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGBA':
            log(f"Converting from {image.mode} to RGBA")
            image = image.convert('RGBA')
        
        original_size = (image.size[0], image.size[1])
        log(f"Image size: {original_size[0]}x{original_size[1]}")
        
        # Processing pipeline
        # 1. White Balance
        log("Step 4: Applying white balance...")
        image = auto_white_balance_extreme(image)
        
        # 2. Pattern Enhancement
        if apply_pattern:
            pattern_type = detect_pattern_type(filename)
            detected_type = {
                "ac_pattern": "Unplated White (25%)",
                "ab_pattern": "Unplated White-Cool (20%)",
                "other": "General Colors (12%)"
            }.get(pattern_type, "Unknown")
            
            log(f"Step 5: Applying pattern enhancement ({pattern_type})...")
            image = apply_pattern_enhancement_extreme(image, pattern_type, intensity)
        else:
            pattern_type = "none"
            detected_type = "No Correction"
            log("Step 5: Skipping pattern enhancement")
        
        # 3. Ring Holes
        log("Step 6: Detecting ring holes...")
        image = ensure_ring_holes_transparent_extreme(image)
        
        # 4. Cubic Enhancement
        log("Step 7: Enhancing cubic details...")
        image = enhance_cubic_sparkle_extreme(image, intensity)
        
        # 5. Resize if needed
        if target_width or target_height:
            log(f"Step 8: Resizing to {target_width}x{target_height}...")
            enhanced_image = apply_target_resize(image, target_width, target_height)
            enhanced_image = apply_extreme_final_enhancement(enhanced_image, intensity)
        else:
            enhanced_image = image
            log("Step 8: No resize needed")
        
        # 6. Final Enhancement
        log("Step 9: Applying final enhancement...")
        enhanced_image = apply_extreme_final_enhancement(enhanced_image, intensity)
        
        # Encode result
        log("Step 10: Encoding result to base64...")
        output_base64 = image_to_base64(enhanced_image)
        log(f"Encoded to {len(output_base64)} characters")
        
        # Statistics
        log("Step 11: Calculating statistics...")
        cubic_mask, _, _, _ = detect_cubic_regions_extreme(enhanced_image)
        cubic_pixel_count = int(np.sum(cubic_mask))
        total_pixels = enhanced_image.size[0] * enhanced_image.size[1]
        cubic_percentage = (cubic_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        
        log(f"✅ Processing complete! Cubic coverage: {cubic_percentage:.1f}%")
        
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
                "processing_mode": "OPTIMIZED",
                "cubic_statistics": {
                    "cubic_pixels": cubic_pixel_count,
                    "cubic_percentage": round(cubic_percentage, 2),
                    "has_cubics": cubic_pixel_count > 0,
                    "total_pixels": total_pixels
                },
                "corrections_applied": [
                    "extreme_white_balance",
                    f"pattern_{pattern_type}" if apply_pattern else "pattern_skipped",
                    "ring_hole_detection",
                    f"cubic_enhancement_{intensity}",
                    f"resize_to_{enhanced_image.width}x{enhanced_image.height}" if (target_width or target_height) else "no_resize",
                    "final_enhancement_reduced_contrast"
                ],
                "base64_padding": "INCLUDED",
                "compression": "PNG_LEVEL_3"
            }
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        log(f"❌ Processing error: {str(e)}")
        log(f"Traceback:\n{tb}")
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION,
                "traceback": tb
            }
        }

def handler(event):
    """RunPod handler - FIXED for health check and input structure"""
    try:
        log("="*50)
        log(f"🎯 HANDLER CALLED! - v{VERSION}")
        log(f"Event type: {type(event)}")
        
        # CRITICAL: Handle health checks properly
        # RunPod sends empty requests or None to check if worker is ready
        if event is None:
            log("✅ Health check: Event is None")
            return {
                "output": {
                    "status": "ready",
                    "version": VERSION,
                    "message": "Worker is ready to process requests"
                }
            }
        
        # If event is empty dict
        if isinstance(event, dict) and len(event) == 0:
            log("✅ Health check: Empty event dict")
            return {
                "output": {
                    "status": "ready",
                    "version": VERSION,
                    "message": "Worker is ready to process requests"
                }
            }
        
        # Check if event has 'input' field (RunPod standard)
        if isinstance(event, dict):
            log(f"Event keys: {list(event.keys())}")
            
            # Standard RunPod structure with 'input' field
            if 'input' in event:
                job_input = event['input']
                log("✅ Found 'input' key in event")
                
                # Check if input is None or empty (health check)
                if job_input is None or (isinstance(job_input, dict) and len(job_input) == 0):
                    log("✅ Health check: Empty or None input")
                    return {
                        "output": {
                            "status": "ready",
                            "version": VERSION,
                            "message": "Worker ready - please send request with image data"
                        }
                    }
            else:
                # No 'input' key - could be direct data or health check
                log("⚠️ No 'input' key found in event")
                
                # If there's other data, try to process it
                if len(event) > 0:
                    log("Attempting to process event directly")
                    job_input = event
                else:
                    # Empty event - health check
                    log("✅ Health check: No input key and empty event")
                    return {
                        "output": {
                            "status": "ready",
                            "version": VERSION,
                            "message": "Worker ready - please send request with 'input' field"
                        }
                    }
        else:
            # Event is not a dict - try to process it anyway
            log(f"⚠️ Event is not a dict: {type(event)}")
            job_input = event
        
        # At this point we should have valid job_input
        # Log what we're working with
        if isinstance(job_input, dict):
            log(f"job_input is dict with keys: {list(job_input.keys())}")
        elif isinstance(job_input, str):
            log(f"job_input is string with {len(job_input)} chars")
        else:
            log(f"job_input type: {type(job_input)}")
        
        # Process the image
        log("🚀 Starting image processing...")
        log("="*50)
        
        result = process_cubic_enhancement(job_input)
        
        log("="*50)
        log("✅ Handler completed successfully")
        log("="*50)
        
        return result
        
    except Exception as e:
        tb = traceback.format_exc()
        log("="*50)
        log(f"❌ HANDLER ERROR: {str(e)}")
        log(f"Traceback:\n{tb}")
        log("="*50)
        
        return {
            "output": {
                "error": str(e),
                "status": "failed",
                "version": VERSION,
                "traceback": tb
            }
        }

# RunPod entry point
if __name__ == "__main__":
    log(f"Starting Cubic Enhancement v{VERSION} - Health Check Fix")
    log("CRITICAL FIXES:")
    log("1. Proper health check handling for empty/None requests")
    log("2. Using dict format for handler registration")
    log("3. Always returning {'output': {...}} structure")
    
    try:
        # FIXED: Use dictionary format which is more standard
        log("Registering handler with RunPod...")
        runpod.serverless.start({"handler": handler})
        log("✅ Handler registered successfully")
    except Exception as e:
        log(f"❌ Failed to start RunPod handler: {str(e)}")
        log(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
