import runpod
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import logging
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################
# CUBIC DETAIL ENHANCEMENT HANDLER
# VERSION: Cubic-Sparkle-V1
################################

VERSION = "Cubic-Sparkle-V1"

def decode_base64_fast(base64_str: str) -> bytes:
    """Fast base64 decode with padding handling"""
    try:
        if not base64_str or len(base64_str) < 50:
            raise ValueError("Invalid base64 string")
        
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[-1]
        
        # Clean whitespace
        base64_str = ''.join(base64_str.split())
        
        # Keep only valid base64 characters
        valid_chars = set(string.ascii_letters + string.digits + '+/=')
        base64_str = ''.join(c for c in base64_str if c in valid_chars)
        
        # Try with padding first
        try:
            decoded = base64.b64decode(base64_str, validate=True)
            return decoded
        except Exception:
            # If fails, try to add proper padding
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
    
    # Force RGBA and save as PNG
    if image.mode != 'RGBA':
        logger.warning(f"⚠️ Converting {image.mode} to RGBA for transparency")
        image = image.convert('RGBA')
    
    logger.info("💎 Saving RGBA image as PNG with compression level 3")
    image.save(buffered, format='PNG', compress_level=3, optimize=True)
    
    buffered.seek(0)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_str  # WITH padding

def find_input_data(data):
    """Extract input image data from various formats - includes thumbnail support"""
    if isinstance(data, str) and len(data) > 50:
        return data
    
    if isinstance(data, dict):
        # Priority keys - thumbnail 추가
        priority_keys = ['image', 'enhanced_image', 'thumbnail', 'image_base64', 'base64', 'img']
        
        for key in priority_keys:
            if key in data and isinstance(data[key], str) and len(data[key]) > 50:
                return data[key]
        
        # Check nested structures
        for key in ['input', 'data', 'output']:
            if key in data:
                if isinstance(data[key], str) and len(data[key]) > 50:
                    return data[key]
                elif isinstance(data[key], dict):
                    result = find_input_data(data[key])
                    if result:
                        return result
    
    return None

def detect_cubic_regions(image: Image.Image, sensitivity=1.0):
    """큐빅/보석 영역 감지"""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    rgb_array = np.array(image.convert('RGB'), dtype=np.uint8)
    alpha_array = np.array(image.split()[3], dtype=np.uint8)
    
    # 색상 공간 변환
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    
    h, s, v = cv2.split(hsv)
    l, a_chan, b_chan = cv2.split(lab)
    
    # 큐빅 마스크 생성 - 다양한 조건
    # 1. 다이아몬드/화이트 큐빅 (높은 명도, 낮은 채도)
    white_cubic = (
        (l > 240 * sensitivity) & 
        (s < 30) & 
        (alpha_array > 200)
    )
    
    # 2. 컬러 큐빅 (높은 명도, 높은 채도)
    color_cubic = (
        (l > 200 * sensitivity) & 
        (s > 100) & 
        (v > 200 * sensitivity) &
        (alpha_array > 200)
    )
    
    # 3. 반사광 영역 (극도로 밝은 부분)
    highlights = (
        (l > 250) & 
        (v > 250) &
        (alpha_array > 200)
    )
    
    # 전체 큐빅 마스크
    cubic_mask = white_cubic | color_cubic | highlights
    
    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cubic_mask = cv2.morphologyEx(cubic_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cubic_mask = cv2.morphologyEx(cubic_mask, cv2.MORPH_CLOSE, kernel)
    
    return cubic_mask.astype(bool), white_cubic, color_cubic, highlights

def enhance_cubic_sparkle(image: Image.Image, intensity=1.0) -> Image.Image:
    """큐빅의 반짝임과 디테일 강화 - 메인 함수"""
    logger.info("💎 Starting cubic detail enhancement...")
    
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    r, g, b, a = image.split()
    rgb_array = np.array(image.convert('RGB'), dtype=np.float32)
    alpha_array = np.array(a, dtype=np.uint8)
    
    # 큐빅 영역 감지
    cubic_mask, white_cubic, color_cubic, highlights = detect_cubic_regions(image, intensity)
    
    cubic_count = np.sum(cubic_mask)
    logger.info(f"✨ Detected {cubic_count} cubic pixels")
    
    if cubic_count == 0:
        logger.info("No cubic regions detected, returning original")
        return image
    
    # 1. 엣지 강화 (큐빅의 컷팅면)
    logger.info("🔷 Enhancing cubic edges...")
    gray = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 다중 스케일 엣지 검출
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 100, 200)
    edges3 = cv2.Canny(gray, 150, 250)
    
    # 엣지 결합
    all_edges = edges1 | edges2 | edges3
    cubic_edges = all_edges & cubic_mask
    
    # 엣지 영역 확장
    edge_dilated = cv2.dilate(cubic_edges.astype(np.uint8), np.ones((3,3)), iterations=1)
    
    # 2. 하이라이트 부스팅
    logger.info("✨ Boosting highlights...")
    lab = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    
    # 밝은 영역 강화
    bright_mask = (l_channel > 240) & cubic_mask
    if np.any(bright_mask):
        boost_factor = 1.1 * intensity
        rgb_array[bright_mask] = np.minimum(rgb_array[bright_mask] * boost_factor, 255)
    
    # 3. 스페큘러 반사 효과 (반짝임)
    logger.info("💫 Adding specular reflections...")
    
    # 큐빅별로 개별 처리
    num_labels, labels = cv2.connectedComponents(cubic_mask.astype(np.uint8))
    
    for i in range(1, min(num_labels, 500)):  # 최대 500개 큐빅 처리
        cubic_region = (labels == i)
        region_size = np.sum(cubic_region)
        
        if region_size < 10 or region_size > 10000:  # 너무 작거나 큰 영역 제외
            continue
        
        # 각 큐빅의 중심과 가장 밝은 점 찾기
        coords = np.where(cubic_region)
        if len(coords[0]) == 0:
            continue
            
        center_y = int(np.mean(coords[0]))
        center_x = int(np.mean(coords[1]))
        
        # 큐빅 영역의 밝기
        region_brightness = l_channel[cubic_region]
        if len(region_brightness) == 0:
            continue
            
        max_bright_idx = np.argmax(region_brightness)
        max_y = coords[0][max_bright_idx]
        max_x = coords[1][max_bright_idx]
        
        # 스파클 효과 추가
        sparkle_radius = max(3, int(np.sqrt(region_size) * 0.3))
        
        # 방사형 그라디언트 스파클
        for dy in range(-sparkle_radius, sparkle_radius + 1):
            for dx in range(-sparkle_radius, sparkle_radius + 1):
                y, x = max_y + dy, max_x + dx
                
                if 0 <= y < rgb_array.shape[0] and 0 <= x < rgb_array.shape[1]:
                    dist = np.sqrt(dy**2 + dx**2)
                    if dist <= sparkle_radius:
                        # 거리에 따른 강도 감소
                        sparkle_intensity = (1 - (dist / sparkle_radius)) * intensity * 0.5
                        
                        # 원본 색상을 유지하면서 밝기만 증가
                        current_color = rgb_array[y, x]
                        brightness_boost = (255 - current_color) * sparkle_intensity
                        rgb_array[y, x] = np.minimum(current_color + brightness_boost, 255)
    
    # 4. 선택적 샤프닝 (큐빅 영역만)
    logger.info("🔪 Sharpening cubic areas...")
    if np.any(cubic_mask):
        # 언샤프 마스크
        blurred = cv2.GaussianBlur(rgb_array, (5, 5), 1.0)
        sharpened = rgb_array + (rgb_array - blurred) * (1.5 * intensity)
        
        # 큐빅 영역에만 적용
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                np.clip(sharpened[:,:,c], 0, 255),
                rgb_array[:,:,c]
            )
    
    # 5. 색상 팝 (컬러 큐빅용)
    logger.info("🌈 Enhancing color cubics...")
    if np.any(color_cubic):
        hsv = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_float = hsv.astype(np.float32)
        
        # 채도 증가
        saturation_boost = 1.3 * intensity
        hsv_float[:,:,1] = np.where(
            color_cubic,
            np.minimum(hsv_float[:,:,1] * saturation_boost, 255),
            hsv_float[:,:,1]
        )
        
        # 명도 약간 증가
        value_boost = 1.05 * intensity
        hsv_float[:,:,2] = np.where(
            color_cubic,
            np.minimum(hsv_float[:,:,2] * value_boost, 255),
            hsv_float[:,:,2]
        )
        
        rgb_array = cv2.cvtColor(hsv_float.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
    # 6. 엣지 하이라이트
    logger.info("✨ Highlighting edges...")
    if np.any(edge_dilated):
        edge_highlight_strength = 0.3 * intensity
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                edge_dilated,
                np.minimum(rgb_array[:,:,c] + (255 - rgb_array[:,:,c]) * edge_highlight_strength, 255),
                rgb_array[:,:,c]
            )
    
    # 7. 최종 미세 조정
    logger.info("🎨 Final adjustments...")
    
    # 전체적인 대비 증가 (큐빅 영역만)
    if np.any(cubic_mask):
        mean_val = np.mean(rgb_array[cubic_mask])
        contrast_factor = 1.1 * intensity
        
        for c in range(3):
            rgb_array[:,:,c] = np.where(
                cubic_mask,
                np.clip((rgb_array[:,:,c] - mean_val) * contrast_factor + mean_val, 0, 255),
                rgb_array[:,:,c]
            )
    
    # RGBA로 재조합
    rgb_enhanced = Image.fromarray(np.clip(rgb_array, 0, 255).astype(np.uint8))
    r2, g2, b2 = rgb_enhanced.split()
    result = Image.merge('RGBA', (r2, g2, b2, a))
    
    # 최종 엣지 샤프닝 (매우 약하게)
    sharpness = ImageEnhance.Sharpness(result)
    result = sharpness.enhance(1.0 + (0.2 * intensity))
    
    logger.info("✅ Cubic enhancement complete!")
    
    return result

def handler(event):
    """RunPod handler for cubic detail enhancement"""
    try:
        logger.info(f"=== Cubic Detail Enhancement {VERSION} Started ===")
        logger.info("💎 큐빅/보석 디테일 강화 전용 모듈")
        logger.info("🔄 Compatible with Enhancement Handler & Thumbnail Handler outputs")
        logger.info("✨ Features:")
        logger.info("  - 다이아몬드/화이트 큐빅 감지")
        logger.info("  - 컬러 큐빅 감지")
        logger.info("  - 엣지 강화 (컷팅면)")
        logger.info("  - 스페큘러 반사 효과")
        logger.info("  - 선택적 샤프닝")
        logger.info("  - 색상 팝 효과")
        
        # 입력 데이터 추출 - thumbnail 키도 체크
        image_data_str = find_input_data(event)
        
        # Thumbnail Handler의 출력도 처리
        if not image_data_str and isinstance(event, dict):
            if 'thumbnail' in event:
                image_data_str = event['thumbnail']
            elif 'output' in event and isinstance(event['output'], dict):
                if 'thumbnail' in event['output']:
                    image_data_str = event['output']['thumbnail']
        
        if not image_data_str:
            raise ValueError("No input image data found")
        
        # 강도 설정 (기본값 1.0)
        intensity = float(event.get('intensity', 1.0))
        intensity = max(0.1, min(2.0, intensity))  # 0.1 ~ 2.0 사이로 제한
        
        logger.info(f"Enhancement intensity: {intensity}")
        
        # 이미지 디코딩
        image_bytes = decode_base64_fast(image_data_str)
        image = Image.open(BytesIO(image_bytes))
        
        # RGBA 변환
        if image.mode != 'RGBA':
            logger.info(f"Converting {image.mode} to RGBA")
            image = image.convert('RGBA')
        
        original_size = image.size
        logger.info(f"Input image size: {original_size}")
        
        # 큐빅 디테일 강화 적용
        enhanced_image = enhance_cubic_sparkle(image, intensity)
        
        # Base64로 인코딩 (padding 포함)
        output_base64 = image_to_base64(enhanced_image)
        
        # 통계 정보
        cubic_mask, _, _, _ = detect_cubic_regions(image)
        cubic_pixel_count = np.sum(cubic_mask)
        cubic_percentage = (cubic_pixel_count / (image.size[0] * image.size[1])) * 100
        
        return {
            "output": {
                "enhanced_image": output_base64,
                "thumbnail": output_base64,  # Thumbnail Handler와 호환성을 위해 추가
                "enhanced_image_with_prefix": f"data:image/png;base64,{output_base64}",
                "size": list(enhanced_image.size),
                "version": VERSION,
                "status": "success",
                "format": "PNG",
                "mode": "RGBA",
                "intensity": intensity,
                "cubic_statistics": {
                    "cubic_pixels": int(cubic_pixel_count),
                    "cubic_percentage": round(cubic_percentage, 2),
                    "has_cubics": cubic_pixel_count > 0
                },
                "enhancements_applied": [
                    "edge_enhancement",
                    "highlight_boosting",
                    "specular_reflections",
                    "selective_sharpening",
                    "color_pop",
                    "edge_highlighting",
                    "contrast_adjustment"
                ],
                "base64_padding": "INCLUDED",
                "compression": "level_3",
                "compatible_with": ["enhancement_handler", "thumbnail_handler"],
                "input_accepted": ["enhanced_image", "thumbnail", "image"]
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
