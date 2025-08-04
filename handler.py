import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import logging
import string
import json
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

################################
# CUBIC DETAIL ENHANCEMENT HANDLER V5.0 - EXTREME EDITION
# VERSION: Cubic-Sparkle-V5.0-Extreme-Lightweight
# Enhanced processing without external dependencies
################################

VERSION = "Cubic-Sparkle-V5.0-Extreme-Lightweight"

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
        
        # Priority keys
        image_keys = ['enhanced_image', 'thumbnail', 'image', 'image_base64', 'base64', 'img', 'input', 'data']
        
        # Direct check
        for key in image_keys:
            if key in data:
                value = data[key]
                if isinstance(value, str) and len(value) > 50:
                    logger.info(f"✅ Found in '{key}' (length: {len(value)})")
                    return value
        
        # Check numbered keys
        for i in range(10):
            key = str(i)
            if key in data:
                value = data[key]
                if isinstance(value, str) and len(value) > 50:
                    logger.info(f"✅ Found in numbered key '{key}' (length: {len(value)})")
                    return value
        
        # Check any key with base64-like content
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 1000:
                sample = value[:100].strip()
                if all(c in string.ascii_letters + string.digits + '+/=' for c in sample):
                    logger.info(f"✅ Found potential base64 in '{key}' (length: {len(value)})")
                    return value
    
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

def auto_white_balance_extreme(image: Image.Image) -> Image.Image:
    """EXTREME white balance correction with multi-pass algorithm"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        rgb_img = Image.merge('RGB', (r, g, b))
        
        # Convert to numpy
        img_array = np.array(rgb_img, dtype=np.float32)
        
        # PASS 1: Aggressive gray world assumption
        for c in range(3):
            channel_mean = np.mean(img_array[:,:,c])
            if channel_mean > 0:
                img_array[:,:,c] = img_array[:,:,c] * (128 / channel_mean)
        
        # PASS 2: Sample-based correction
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
            
            # More aggressive correction
            if r_avg > 0:
                img_array[:,:,0] *= min(1.5, gray_avg / r_avg)
            if g_avg > 0:
                img_array[:,:,1] *= min(1.5, gray_avg / g_avg)
            if b_avg > 0:
                img_array[:,:,2] *= min(1.5, gray_avg / b_avg)
        
        # PASS 3: Highlight preservation
        highlights = img_array > 240
        img_array = np.clip(img_array, 0, 255)
        
        # Restore some highlights
        img_array[highlights] = np.minimum(img_array[highlights] * 1.05, 255)
        
        # Convert back
        rgb_balanced = Image.fromarray(img_array.astype(np.uint8))
        
        # PASS 4: Fine-tune with PIL
        color = ImageEnhance.Color(rgb_balanced)
        rgb_balanced = color.enhance(1.1)
        
        # Merge with alpha
        r2, g2, b2 = rgb_balanced.split()
        result = Image.merge('RGBA', (r2, g2, b2, a))
        
        return result
        
    except Exception as e:
        logger.error(f"White balance error: {e}")
        return image

def apply_pattern_enhancement_extreme(image: Image.Image, pattern_type: str, intensity: float = 1.0) -> Image.Image:
    """EXTREME pattern enhancement with maximum detail preservation"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        r, g, b, a = image.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        
        # Multi-layer detail extraction
        detail_layer1 = rgb_image.filter(ImageFilter.DETAIL)
        detail_layer2 = rgb_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        detail_layer3 = rgb_image.filter(ImageFilter.SHARPEN)
        
        img_array = np.array(rgb_image, dtype=np.float32)
        detail_array1 = np.array(detail_layer1, dtype=np.float32)
        detail_array2 = np.array(detail_layer2, dtype=np.float32)
        detail_array3 = np.array(detail_layer3, dtype=np.float32)
        
        if pattern_type == "ac_pattern":
            # AC - Unplated white (25% white) - INCREASED
            logger.info("🎨 AC Pattern - Unplated White EXTREME")
            white_overlay = 0.25 * intensity
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            
            # EXTREME detail mixing
            img_array = (
                img_array * 0.40 + 
                detail_array1 * 0.25 + 
                detail_array2 * 0.20 + 
                detail_array3 * 0.15
            )
            
            img_array = np.clip(img_array, 0, 255)
            rgb_image = Image.fromarray(img_array.astype(np.uint8))
            
            # EXTREME adjustments
            brightness = ImageEnhance.Brightness(rgb_image)
            rgb_image = brightness.enhance(1.08)
            
            color = ImageEnhance.Color(rgb_image)
            rgb_image = color.enhance(0.92)
            
            contrast = ImageEnhance.Contrast(rgb_image)
            rgb_image = contrast.enhance(1.25)
            
            # EXTREME sharpness
            sharpness = ImageEnhance.Sharpness(rgb_image)
            rgb_image = sharpness.enhance(2.5)
            
        elif pattern_type == "ab_pattern":
            # AB - Unplated white cool tone (20% white) - INCREASED
            logger.info("🎨 AB Pattern - Unplated White Cool Tone EXTREME")
            white_overlay = 0.20 * intensity
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            
            # EXTREME cool tone
            img_array[:,:,0] *= 0.92
            img_array[:,:,1] *= 0.96
            img_array[:,:,2] *= 1.05
            
            # Strong cool overlay
            cool_overlay = np.array([235, 245, 255], dtype=np.float32)
            img_array = img_array * 0.88 + cool_overlay * 0.12
            
            # EXTREME detail mixing
            img_array = (
                img_array * 0.45 + 
                detail_array1 * 0.20 + 
                detail_array2 * 0.20 + 
                detail_array3 * 0.15
            )
            
            img_array = np.clip(img_array, 0, 255)
            rgb_image = Image.fromarray(img_array.astype(np.uint8))
            
            color = ImageEnhance.Color(rgb_image)
            rgb_image = color.enhance(0.82)
            
            brightness = ImageEnhance.Brightness(rgb_image)
            rgb_image = brightness.enhance(1.06)
            
            contrast = ImageEnhance.Contrast(rgb_image)
            rgb_image = contrast.enhance(1.22)
            
            # EXTREME cool tone sharpness
            sharpness = ImageEnhance.Sharpness(rgb_image)
            rgb_image = sharpness.enhance(2.8)
            
        else:
            # Other - General colors (12% white) - INCREASED
            logger.info("🎨 Other Pattern - General Colors EXTREME")
            white_overlay = 0.12 * intensity
            img_array = img_array * (1 - white_overlay) + 255 * white_overlay
            
            # EXTREME detail mixing for general colors
            img_array = (
                img_array * 0.35 + 
                detail_array1 * 0.25 + 
                detail_array2 * 0.25 + 
                detail_array3 * 0.15
            )
            
            img_array = np.clip(img_array, 0, 255)
            rgb_image = Image.fromarray(img_array.astype(np.uint8))
            
            brightness = ImageEnhance.Brightness(rgb_image)
            rgb_image = brightness.enhance(1.15)
            
            color = ImageEnhance.Color(rgb_image)
            rgb_image = color.enhance(1.08)
            
            contrast = ImageEnhance.Contrast(rgb_image)
            rgb_image = contrast.enhance(1.3)
            
            # EXTREME general sharpness
            sharpness = ImageEnhance.Sharpness(rgb_image)
            rgb_image = sharpness.enhance(3.0)
        
        # Common EXTREME final pass
        contrast = ImageEnhance.Contrast(rgb_image)
        rgb_image = contrast.enhance(1.2)
        
        # Ultra sharpness final pass
        sharpness = ImageEnhance.Sharpness(rgb_image)
        rgb_image = sharpness.enhance(2.0)
        
        # Merge back
        r2, g2, b2 = rgb_image.split()
        result = Image.merge('RGBA', (r2, g2, b2, a))
        
        return result
        
    except Exception as e:
        logger.error(f"Pattern enhancement error: {e}")
        return image

def ensure_ring_holes_transparent_extreme(image: Image.Image) -> Image.Image:
    """EXTREME ring hole detection with aggressive transparency"""
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
        
        # EXTREME hole detection - more aggressive
        holes_mask = (v_array > 240) & (s_array < 30) & (alpha_array > 180)
        
        # Pure white detection - more aggressive
        r_array = np.array(r)
        g_array = np.array(g)
        b_array = np.array(b)
        pure_white = (r_array > 245) & (g_array > 245) & (b_array > 245)
        
        # Near-white detection
        near_white = (
            (r_array > 235) & (g_array > 235) & (b_array > 235) &
            (np.abs(r_array.astype(float) - g_array.astype(float)) < 10) &
            (np.abs(g_array.astype(float) - b_array.astype(float)) < 10)
        )
        
        # Combine all masks
        holes_mask = holes_mask | pure_white | near_white
        
        # Aggressive morphology
        holes_image = Image.fromarray((holes_mask * 255).astype(np.uint8))
        holes_image = holes_image.filter(ImageFilter.MinFilter(5))
        holes_image = holes_image.filter(ImageFilter.MaxFilter(5))
        holes_mask = np.array(holes_image) > 128
        
        # Apply transparency with feathering
        alpha_array[holes_mask] = 0
        
        # Feather edges
        for i in range(2):
            alpha_temp = Image.fromarray(alpha_array)
            alpha_temp = alpha_temp.filter(ImageFilter.GaussianBlur(1))
            alpha_array = np.array(alpha_temp)
        
        # Rebuild image
        a_new = Image.fromarray(alpha_array)
        result = Image.merge('RGBA', (r, g, b, a_new))
        
        hole_count = np.sum(holes_mask)
        logger.info(f"✅ Ring holes (EXTREME): {hole_count} pixels")
        
        return result
        
    except Exception as e:
        logger.error(f"Ring hole detection error: {e}")
        return image

def detect_cubic_regions_extreme(image: Image.Image, sensitivity=1.5):
    """EXTREME cubic/crystal detection with maximum sensitivity"""
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
        
        # EXTREME edge detection - multi-pass
        edges1 = image.filter(ImageFilter.FIND_EDGES)
        edges2 = edges1.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges3 = edges2.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges_array = np.array(edges3.convert('L'))
        
        # EXTREME cubic types with lower thresholds
        white_cubic = (
            (v_array > 220 * sensitivity) & 
            (s_array < 45) & 
            (alpha_array > 150)
        )
        
        color_cubic = (
            (v_array > 180 * sensitivity) & 
            (s_array > 70) & 
            (alpha_array > 150)
        )
        
        # EXTREME edge cubic
        edge_cubic = (
            (edges_array > 60) & 
            (v_array > 190) & 
            (alpha_array > 150)
        )
        
        # EXTREME highlights
        highlights = (
            (v_array > 235) & 
            (s_array < 80) &
            (alpha_array > 150)
        )
        
        # EXTREME micro details
        micro_details = (
            (edges_array > 30) & 
            (v_array > 160) & 
            (s_array > 20) &
            (alpha_array > 150)
        )
        
        # Gradient detection for smooth transitions
        gradient_x = np.abs(np.gradient(v_array, axis=1))
        gradient_y = np.abs(np.gradient(v_array, axis=0))
        gradient_mask = ((gradient_x > 5) | (gradient_y > 5)) & (alpha_array > 150)
        
        # Combine all masks
        cubic_mask = white_cubic | color_cubic | edge_cubic | highlights | micro_details | gradient_mask
        
        # Less aggressive cleanup
        cubic_image = Image.fromarray((cubic_mask * 255).astype(np.uint8))
        cubic_image = cubic_image.filter(ImageFilter.MinFilter(1))
        cubic_image = cubic_image.filter(ImageFilter.MaxFilter(1))
        cubic_mask = np.array(cubic_image) > 128
        
        return cubic_mask.astype(bool), white_cubic, color_cubic, highlights
        
    except Exception as e:
        logger.error(f"Cubic detection error: {e}")
        shape = (image.size[1], image.size[0])
        empty = np.zeros(shape, dtype=bool)
        return empty, empty, empty, empty

def enhance_cubic_sparkle_extreme(image: Image.Image, intensity=1.5) -> Image.Image:
    """EXTREME cubic sparkle enhancement with maximum effect"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Detect cubics with higher sensitivity
        cubic_mask, white_cubic, color_cubic, highlights = detect_cubic_regions_extreme(image, intensity)
        
        cubic_count = np.sum(cubic_mask)
        total_pixels = image.size[0] * image.size[1]
        cubic_percent = (cubic_count / total_pixels) * 100
        
        logger.info(f"💎 EXTREME Cubic pixels: {cubic_count} ({cubic_percent:.1f}%)")
        
        if cubic_count == 0:
            # Even if no cubics detected, apply general enhancement
            sharpness = ImageEnhance.Sharpness(image)
            image = sharpness.enhance(1.5 * intensity)
            return image
        
        # Work with RGB
        r, g, b, a = image.split()
        rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
        
        # 1. EXTREME multi-layer edge processing
        edges1 = image.filter(ImageFilter.FIND_EDGES)
        edges2 = edges1.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edges3 = edges2.filter(ImageFilter.SHARPEN)
        
        edges_array1 = np.array(edges1.convert('RGB'), dtype=np.float32)
        edges_array2 = np.array(edges2.convert('RGB'), dtype=np.float32)
        edges_array3 = np.array(edges3.convert('RGB'), dtype=np.float32)
        
        # Detail layers
        detail1 = image.filter(ImageFilter.DETAIL)
        detail2 = detail1.filter(ImageFilter.SHARPEN)
        detail_array1 = np.array(detail1.convert('RGB'), dtype=np.float32)
        detail_array2 = np.array(detail2.convert('RGB'), dtype=np.float32)
        
        # EXTREME blending for cubic regions
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                (
                    rgb_array[:,:,c] * 0.20 + 
                    edges_array1[:,:,c] * 0.20 + 
                    edges_array2[:,:,c] * 0.20 + 
                    edges_array3[:,:,c] * 0.15 +
                    detail_array1[:,:,c] * 0.15 +
                    detail_array2[:,:,c] * 0.10
                ),
                rgb_array[:,:,c]
            )
        
        # 2. EXTREME local contrast
        if np.any(cubic_mask):
            for c in range(3):
                channel = rgb_array[:,:,c]
                
                # Local area processing
                for y in range(0, channel.shape[0], 50):
                    for x in range(0, channel.shape[1], 50):
                        y_end = min(y + 50, channel.shape[0])
                        x_end = min(x + 50, channel.shape[1])
                        
                        local_mask = cubic_mask[y:y_end, x:x_end]
                        if np.any(local_mask):
                            local_data = channel[y:y_end, x:x_end]
                            local_mean = np.mean(local_data[local_mask])
                            local_std = np.std(local_data[local_mask])
                            
                            # Adaptive contrast
                            contrast_factor = (1.5 + 0.2 * min(local_std / 30, 1.0)) * intensity
                            
                            channel[y:y_end, x:x_end] = np.where(
                                local_mask,
                                np.clip((local_data - local_mean) * contrast_factor + local_mean, 0, 255),
                                local_data
                            )
                
                rgb_array[:,:,c] = channel
        
        # 3. EXTREME highlights
        if np.any(highlights):
            # Primary boost
            boost = 1.3 * intensity
            for c in range(3):
                rgb_array[:,:,c] = np.where(
                    highlights,
                    np.minimum(rgb_array[:,:,c] * boost, 255),
                    rgb_array[:,:,c]
                )
            
            # Multi-level glow
            highlight_image = Image.fromarray((highlights * 255).astype(np.uint8))
            
            # Inner glow
            glow1 = highlight_image.filter(ImageFilter.GaussianBlur(1))
            glow_mask1 = np.array(glow1) > 150
            
            # Outer glow
            glow2 = highlight_image.filter(ImageFilter.GaussianBlur(3))
            glow_mask2 = np.array(glow2) > 80
            
            for c in range(3):
                rgb_array[:,:,c] = np.where(
                    glow_mask1 & ~highlights,
                    np.minimum(rgb_array[:,:,c] * (1.15 * intensity), 255),
                    rgb_array[:,:,c]
                )
                rgb_array[:,:,c] = np.where(
                    glow_mask2 & ~glow_mask1 & ~highlights,
                    np.minimum(rgb_array[:,:,c] * (1.08 * intensity), 255),
                    rgb_array[:,:,c]
                )
        
        # 4. EXTREME micro-detail enhancement
        if np.any(cubic_mask):
            # Multiple unsharp mask passes
            blur1 = np.array(image.filter(ImageFilter.GaussianBlur(0.5)).convert('RGB'), dtype=np.float32)
            blur2 = np.array(image.filter(ImageFilter.GaussianBlur(1.5)).convert('RGB'), dtype=np.float32)
            
            for c in range(3):
                # Fine details
                detail_mask1 = cubic_mask & (np.abs(rgb_array[:,:,c] - blur1[:,:,c]) > 3)
                rgb_array[:,:,c] = np.where(
                    detail_mask1,
                    np.clip(rgb_array[:,:,c] + (rgb_array[:,:,c] - blur1[:,:,c]) * 0.5 * intensity, 0, 255),
                    rgb_array[:,:,c]
                )
                
                # Medium details
                detail_mask2 = cubic_mask & (np.abs(rgb_array[:,:,c] - blur2[:,:,c]) > 5)
                rgb_array[:,:,c] = np.where(
                    detail_mask2,
                    np.clip(rgb_array[:,:,c] + (rgb_array[:,:,c] - blur2[:,:,c]) * 0.3 * intensity, 0, 255),
                    rgb_array[:,:,c]
                )
        
        # Convert back
        rgb_enhanced = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
        r2, g2, b2 = rgb_enhanced.split()
        result = Image.merge('RGBA', (r2, g2, b2, a))
        
        # EXTREME final sharpening cascade
        sharpness_passes = [
            (1.8 + (0.7 * intensity)),
            (1.5 + (0.5 * intensity)),
            (1.3 + (0.3 * intensity))
        ]
        
        for sharp_strength in sharpness_passes:
            sharpness = ImageEnhance.Sharpness(result)
            result = sharpness.enhance(sharp_strength)
        
        # Final contrast boost
        contrast = ImageEnhance.Contrast(result)
        result = contrast.enhance(1.1 + (0.1 * intensity))
        
        return result
        
    except Exception as e:
        logger.error(f"Cubic enhancement error: {e}")
        return image

def apply_extreme_final_enhancement(image: Image.Image, intensity: float = 1.0) -> Image.Image:
    """Apply final EXTREME enhancement pass"""
    try:
        # Clarity enhancement
        clarity = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # Structure enhancement
        contrast = ImageEnhance.Contrast(clarity)
        enhanced = contrast.enhance(1.05 * intensity)
        
        # Micro contrast
        sharpness = ImageEnhance.Sharpness(enhanced)
        enhanced = sharpness.enhance(1.2 * intensity)
        
        # Color vibrancy
        color = ImageEnhance.Color(enhanced)
        enhanced = color.enhance(1.05 * intensity)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Final enhancement error: {e}")
        return image

def apply_advanced_upscale(image: Image.Image, scale_factor: float) -> Image.Image:
    """Advanced upscaling with edge-preserving algorithm"""
    try:
        if scale_factor <= 1.0:
            return image
            
        logger.info(f"📏 Applying EXTREME {scale_factor}x upscale...")
        
        # Calculate new dimensions
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        # Step 1: Initial upscale with LANCZOS
        try:
            upscaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except AttributeError:
            upscaled = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Step 2: Edge-preserving enhancement
        edges = upscaled.filter(ImageFilter.FIND_EDGES)
        edges_enhanced = edges.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Step 3: Blend edges back
        r, g, b, a = upscaled.split()
        r_edge, g_edge, b_edge, _ = edges_enhanced.split()
        
        # Convert to arrays for blending
        r_array = np.array(r, dtype=np.float32)
        g_array = np.array(g, dtype=np.float32)
        b_array = np.array(b, dtype=np.float32)
        
        r_edge_array = np.array(r_edge, dtype=np.float32)
        g_edge_array = np.array(g_edge, dtype=np.float32)
        b_edge_array = np.array(b_edge, dtype=np.float32)
        
        # Adaptive blending based on edge strength
        blend_factor = 0.15 * min(scale_factor / 2, 1.5)
        
        r_array = r_array * (1 - blend_factor) + r_edge_array * blend_factor
        g_array = g_array * (1 - blend_factor) + g_edge_array * blend_factor
        b_array = b_array * (1 - blend_factor) + b_edge_array * blend_factor
        
        # Convert back
        r_new = Image.fromarray(np.clip(r_array, 0, 255).astype(np.uint8))
        g_new = Image.fromarray(np.clip(g_array, 0, 255).astype(np.uint8))
        b_new = Image.fromarray(np.clip(b_array, 0, 255).astype(np.uint8))
        
        upscaled = Image.merge('RGBA', (r_new, g_new, b_new, a))
        
        # Step 4: Progressive sharpening based on scale
        sharpness_passes = int(scale_factor)
        for i in range(sharpness_passes):
            strength = 1.1 + (0.1 * (i + 1) / sharpness_passes)
            sharpness = ImageEnhance.Sharpness(upscaled)
            upscaled = sharpness.enhance(strength)
        
        logger.info(f"✅ EXTREME upscaled to {new_width}x{new_height}")
        return upscaled
        
    except Exception as e:
        logger.error(f"Upscale error: {e}")
        return image

def apply_target_resize(image: Image.Image, target_width: int = None, target_height: int = None) -> Image.Image:
    """Resize image to specific dimensions while maintaining aspect ratio"""
    try:
        current_width, current_height = image.size
        
        # If both dimensions provided, resize to exact size
        if target_width and target_height:
            logger.info(f"📐 Resizing to exact {target_width}x{target_height}...")
            resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
        # If only width provided, maintain aspect ratio
        elif target_width:
            ratio = target_width / current_width
            new_height = int(current_height * ratio)
            logger.info(f"📐 Resizing to width {target_width} (height: {new_height})...")
            resized = image.resize((target_width, new_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
        # If only height provided, maintain aspect ratio
        elif target_height:
            ratio = target_height / current_height
            new_width = int(current_width * ratio)
            logger.info(f"📐 Resizing to height {target_height} (width: {new_width})...")
            resized = image.resize((new_width, target_height), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
        else:
            return image
        
        # Apply sharpening after resize
        scale_factor = max(resized.width / current_width, resized.height / current_height)
        if scale_factor > 1:
            sharp_strength = 1.2 + (0.3 * min(scale_factor, 3))
        else:
            sharp_strength = 1.1 + (0.2 * (1 - scale_factor))
        
        sharpness = ImageEnhance.Sharpness(resized)
        resized = sharpness.enhance(sharp_strength)
        
        logger.info(f"✅ Resized from {current_width}x{current_height} to {resized.width}x{resized.height}")
        return resized
        
    except Exception as e:
        logger.error(f"Resize error: {e}")
        return image

def process_cubic_enhancement(job_input):
    """Main processing function with EXTREME enhancements"""
    try:
        logger.info("="*50)
        logger.info("🚀 CUBIC DETAIL ENHANCEMENT EXTREME START")
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
                "version": VERSION
            }
            
            return {
                "output": {
                    "error": error_msg,
                    "status": "failed",
                    "version": VERSION,
                    "debug": debug_info
                }
            }
        
        # Parameters with EXTREME defaults
        filename = ''
        intensity = 1.5  # Increased default
        apply_pattern = True
        upscale_factor = 1
        target_width = None
        target_height = None
        
        if isinstance(job_input, dict):
            filename = job_input.get('filename', '')
            intensity = float(job_input.get('intensity', 1.5))
            apply_pattern = job_input.get('pattern_enhancement', True)
            upscale_factor = int(job_input.get('upscale_factor', 1))
            target_width = job_input.get('target_width', None)
            target_height = job_input.get('target_height', None)
        
        # Intensity range expanded
        intensity = max(0.5, min(3.0, intensity))
        upscale_factor = max(1, min(4, upscale_factor))
        
        # Validate target dimensions
        if target_width:
            target_width = int(target_width)
            target_width = max(100, min(8192, target_width))
        if target_height:
            target_height = int(target_height)
            target_height = max(100, min(8192, target_height))
        
        logger.info("📋 EXTREME PARAMETERS:")
        logger.info(f"  Filename: {filename}")
        logger.info(f"  Intensity: {intensity} (EXTREME)")
        logger.info(f"  Pattern: {apply_pattern}")
        if target_width or target_height:
            logger.info(f"  Target Size: {target_width or 'auto'}x{target_height or 'auto'}")
        else:
            logger.info(f"  Upscale: {upscale_factor}x")
        
        # Decode image
        logger.info("🖼️ Decoding image...")
        image_bytes = decode_base64_fast(image_data_str)
        image = Image.open(BytesIO(image_bytes))
        
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        logger.info(f"📐 Image: {image.size[0]}x{image.size[1]} {image.mode}")
        
        # Store original size for final resize
        original_size = (image.size[0], image.size[1])
        logger.info(f"📐 Original size stored: {original_size[0]}x{original_size[1]}")
        
        # EXTREME Processing pipeline
        logger.info("="*30)
        logger.info("🔧 EXTREME PROCESSING PIPELINE")
        logger.info("="*30)
        
        # 1. EXTREME White Balance
        logger.info("⚖️ [1/6] EXTREME White Balance...")
        image = auto_white_balance_extreme(image)
        
        # 2. EXTREME Pattern Enhancement
        if apply_pattern:
            logger.info("🎨 [2/6] EXTREME Pattern Enhancement...")
            pattern_type = detect_pattern_type(filename)
            detected_type = {
                "ac_pattern": "Unplated White EXTREME (25%)",
                "ab_pattern": "Unplated White-Cool EXTREME (20%)",
                "other": "General Colors EXTREME (12%)"
            }.get(pattern_type, "Unknown")
            
            logger.info(f"  Type: {pattern_type} - {detected_type}")
            image = apply_pattern_enhancement_extreme(image, pattern_type, intensity)
        else:
            logger.info("⏭️ [2/6] Pattern Enhancement (skipped)")
            pattern_type = "none"
            detected_type = "No Correction"
        
        # 3. EXTREME Ring Holes
        logger.info("🔍 [3/6] EXTREME Ring Hole Detection...")
        image = ensure_ring_holes_transparent_extreme(image)
        
        # 4. EXTREME Cubic Enhancement
        logger.info("💎 [4/6] EXTREME Cubic Enhancement...")
        image = enhance_cubic_sparkle_extreme(image, intensity)
        
        # 5. Upscaling or Resizing
        if target_width or target_height:
            logger.info("📐 [5/6] Target size resizing...")
            enhanced_image = apply_target_resize(image, target_width, target_height)
            # Apply extreme enhancement after resize
            enhanced_image = apply_extreme_final_enhancement(enhanced_image, intensity)
        elif upscale_factor > 1:
            logger.info(f"📏 [5/6] EXTREME {upscale_factor}x upscale...")
            enhanced_image = apply_advanced_upscale(image, upscale_factor)
            
            # Return to original size after upscaling
            logger.info(f"📐 Returning to original size {original_size[0]}x{original_size[1]}...")
            enhanced_image = enhanced_image.resize(original_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
            # Apply extreme sharpening after downscale
            sharpness = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = sharpness.enhance(1.3 * intensity)
        else:
            logger.info("⏭️ [5/6] Upscale/Resize (skipped)")
            enhanced_image = image
        
        # 6. EXTREME Final Enhancement
        logger.info("✨ [6/6] EXTREME Final Enhancement...")
        enhanced_image = apply_extreme_final_enhancement(enhanced_image, intensity)
        
        # Final encoding
        logger.info("📤 Encoding result...")
        output_base64 = image_to_base64(enhanced_image)
        
        # Statistics
        cubic_mask, _, _, _ = detect_cubic_regions_extreme(enhanced_image)
        cubic_pixel_count = int(np.sum(cubic_mask))
        total_pixels = enhanced_image.size[0] * enhanced_image.size[1]
        cubic_percentage = (cubic_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
        
        # Response
        logger.info("="*30)
        logger.info("✅ EXTREME PROCESSING COMPLETE")
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
                "processing_mode": "EXTREME",
                "cubic_statistics": {
                    "cubic_pixels": cubic_pixel_count,
                    "cubic_percentage": round(cubic_percentage, 2),
                    "has_cubics": cubic_pixel_count > 0,
                    "total_pixels": total_pixels
                },
                "corrections_applied": [
                    "extreme_white_balance",
                    f"extreme_pattern_{pattern_type}" if apply_pattern else "pattern_skipped",
                    "extreme_ring_hole_detection",
                    f"extreme_cubic_enhancement_{intensity}",
                    f"resize_to_{enhanced_image.width}x{enhanced_image.height}" if (target_width or target_height) else
                    f"extreme_upscale_{upscale_factor}x_return_to_original" if upscale_factor > 1 else "no_resize",
                    "extreme_final_enhancement"
                ],
                "upscale_factor": upscale_factor,
                "processing_resolution": f"{upscale_factor}x ({int(original_size[0]*upscale_factor)}x{int(original_size[1]*upscale_factor)})" if upscale_factor > 1 else "1x",
                "target_dimensions": {
                    "width": target_width,
                    "height": target_height
                } if (target_width or target_height) else None,
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
        logger.info(f"⚡ EXTREME Mode Enabled - Lightweight Edition")
        logger.info(f"📥 Event type: {type(event)}")
        
        # Critical: Log raw event for debugging
        if isinstance(event, dict):
            logger.info(f"📋 Event keys: {list(event.keys())}")
            if len(event) < 10:
                logger.info(f"📄 Full event: {event}")
        
        # RunPod ALWAYS sends data in {"input": {...}} format
        if not isinstance(event, dict):
            raise ValueError(f"Event must be a dict, got {type(event)}")
        
        if 'input' not in event:
            logger.error("❌ CRITICAL: No 'input' field in event!")
            logger.error(f"Available keys: {list(event.keys())}")
            logger.error("RunPod requires {'input': {...}} structure")
            
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
                "expected_structure": {"input": {"enhanced_image": "base64_string", "filename": "optional", "intensity": 1.5, "upscale_factor": 2, "target_width": 2048, "target_height": 2048}}
            }
        }

# RunPod entry point
if __name__ == "__main__":
    logger.info(f"🚀 Starting Cubic Detail Enhancement v{VERSION}")
    logger.info(f"⚡ EXTREME Mode - Maximum Enhancement")
    logger.info(f"📦 Lightweight Edition - No External Dependencies")
    logger.info("⚠️ CRITICAL: RunPod requires {'input': {...}} structure")
    logger.info("💎 EXTREME Processing: Maximum detail preservation and enhancement")
    
    # FIXED: Pass handler directly, not wrapped in dict
    runpod.serverless.start(handler)
