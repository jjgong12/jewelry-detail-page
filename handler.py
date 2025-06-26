import json
import base64
import io
import os
from PIL import Image, ImageDraw, ImageFont
import requests
from typing import List, Dict

def handler(event):
    """
    주얼리 상세페이지 생성 핸들러
    30,000px 길이의 상세페이지를 자동으로 만듭니다.
    6720×4480 이미지를 적절히 크롭하여 디테일 보존
    """
    try:
        # 입력 받기
        images = event.get('images', [])  # 보정된 이미지들 (base64)
        texts = event.get('texts', [])     # Claude가 생성한 문구들
        
        # 디자인 설정값
        PAGE_WIDTH = 860          # 상세페이지 너비
        TARGET_HEIGHT = 30000     # 목표 높이
        MARGIN = 50              # 좌우 여백
        IMAGE_SPACING = 200      # 이미지 간격 (여유있게)
        TEXT_SPACING = 150       # 텍스트 간격
        TOP_MARGIN = 150         # 상단 여백
        
        # 크롭 설정
        DISPLAY_HEIGHT = 1200    # 각 이미지 표시 높이
        CONTENT_WIDTH = PAGE_WIDTH - (MARGIN * 2)  # 760px
        
        # 빈 캔버스 생성 (배경색: 흰색)
        detail_page = Image.new('RGB', (PAGE_WIDTH, TARGET_HEIGHT), '#FFFFFF')
        
        # 현재 위치 추적
        current_y = TOP_MARGIN
        
        # 각 이미지와 텍스트 처리
        for i, img_base64 in enumerate(images):
            # Base64 이미지 디코딩
            img_data = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_data))
            
            # 6720×4480 이미지를 크롭하여 디테일 보존
            # 1단계: 높이를 DISPLAY_HEIGHT로 맞춤
            height_ratio = DISPLAY_HEIGHT / img.height
            temp_width = int(img.width * height_ratio)
            temp_height = DISPLAY_HEIGHT
            
            # 리사이징 (디테일 보존을 위해 LANCZOS 사용)
            img_resized = img.resize((temp_width, temp_height), Image.Resampling.LANCZOS)
            
            # 2단계: 중앙에서 CONTENT_WIDTH만큼 크롭
            if temp_width > CONTENT_WIDTH:
                # 중앙 크롭
                left = (temp_width - CONTENT_WIDTH) // 2
                right = left + CONTENT_WIDTH
                img_cropped = img_resized.crop((left, 0, right, temp_height))
            else:
                # 이미지가 작으면 그대로 사용
                img_cropped = img_resized
            
            # 이미지를 중앙 정렬하여 붙이기
            x_position = (PAGE_WIDTH - img_cropped.width) // 2
            detail_page.paste(img_cropped, (x_position, current_y))
            current_y += img_cropped.height + IMAGE_SPACING
            
            # 텍스트 추가 (있는 경우)
            if i < len(texts) and texts[i]:
                text_img = create_text_block(texts[i], CONTENT_WIDTH)
                # 텍스트도 중앙 정렬
                text_x = (PAGE_WIDTH - text_img.width) // 2
                # 투명 배경 처리를 위해 mask 사용
                if text_img.mode == 'RGBA':
                    detail_page.paste(text_img, (text_x, current_y), text_img)
                else:
                    detail_page.paste(text_img, (text_x, current_y))
                current_y += text_img.height + TEXT_SPACING
            
            # 높이 체크 (30,000px 초과 방지)
            if current_y > TARGET_HEIGHT - 1000:
                print(f"높이 제한 도달: {i+1}개 이미지 처리 완료")
                break
        
        # 하단 여백 추가
        current_y += 200
        
        # 최종 이미지를 base64로 변환
        buffer = io.BytesIO()
        # 파일 크기 최적화를 위해 quality 조정
        detail_page.save(buffer, format='JPEG', quality=90, optimize=True)
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Make.com용으로 padding 제거
        result_base64 = result_base64.rstrip('=')
        
        return {
            "output": {
                "detail_page": result_base64,
                "dimensions": {
                    "width": PAGE_WIDTH,
                    "height": TARGET_HEIGHT
                },
                "actual_content_height": current_y,
                "images_processed": len(images),
                "image_display_size": {
                    "width": CONTENT_WIDTH,
                    "height": DISPLAY_HEIGHT
                }
            }
        }
        
    except Exception as e:
        return {
            "output": {
                "error": str(e),
                "error_type": type(e).__name__
            }
        }


def create_text_block(text, width):
    """
    텍스트 블록 이미지 생성 (한글 지원, 감성적 디자인)
    """
    try:
        # 한글 폰트 다운로드
        font_path = "/tmp/NanumMyeongjo.ttf"
        if not os.path.exists(font_path):
            print("한글 폰트 다운로드 중...")
            # 나눔명조 - 더 세련된 폰트
            font_url = "https://github.com/google/fonts/raw/main/ofl/nanummyeongjo/NanumMyeongjo-Regular.ttf"
            response = requests.get(font_url)
            with open(font_path, 'wb') as f:
                f.write(response.content)
        
        # 폰트 설정 (더 작고 세련되게)
        font_size = 22
        line_height = int(font_size * 1.8)  # 줄간격 넓게
        font = ImageFont.truetype(font_path, font_size)
        
        # 텍스트를 줄바꿈 처리
        lines = text.split('\n')
        
        # 텍스트 영역 높이 계산
        text_height = len(lines) * line_height + 60
        
        # 텍스트 이미지 생성 (투명 배경)
        text_img = Image.new('RGBA', (width, text_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(text_img)
        
        # 각 줄 그리기 (중앙 정렬)
        y = 30
        for line in lines:
            line = line.strip()
            if line:  # 빈 줄이 아닌 경우만
                # 텍스트 크기 계산
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                
                # 중앙 정렬
                x = (width - text_width) // 2
                
                # 텍스트 색상: 고급스러운 진회색
                draw.text((x, y), line, fill=(68, 68, 68, 255), font=font)
            y += line_height
        
        return text_img
        
    except Exception as e:
        # 폰트 에러 시 기본 텍스트 이미지 생성
        print(f"텍스트 생성 에러: {e}")
        fallback_img = Image.new('RGBA', (width, 100), (255, 255, 255, 0))
        draw = ImageDraw.Draw(fallback_img)
        draw.text((20, 20), text[:100], fill=(68, 68, 68, 255))
        return fallback_img


def runpod_handler(event):
    """RunPod 서버리스 진입점"""
    return handler(event["input"])

# RunPod 서버리스 환경에서 실행
if __name__ == "__runpod__":
    import runpod
    runpod.serverless.start({"handler": runpod_handler})
