import os
import warnings
import torch
import gc
import json
import re
import math
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# [설정]
# ---------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT_IMAGE_PATH = "test2.png"
OUTPUT_DIR = "final_menu_project"

# 모델 ID
OCR_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_ADAPTER_PATH = "output/qwen25_vl_ocr_lora_20eps_new"
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
IMAGE_MODEL_ID = "SG161222/RealVisXL_V3.0_Turbo"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# [폰트 경로]
# ---------------------------------------------------------
JP_FONT_PATH = "C:/Users/JONGWOONG/Downloads/SourceHanSansJP/SourceHanSansJP-VF.otf"
KR_FONT_PATH = "C:/Windows/Fonts/malgun.ttf"

from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )
    except Exception as e:
        raise ImportError(
            "Qwen2.5-VL 모델 클래스를 찾을 수 없습니다. "
            "`transformers`를 Qwen2.5-VL 지원 버전으로 업그레이드하세요."
        ) from e

from qwen_vl_utils import process_vision_info
from peft import PeftModel
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler


# ==========================================
# 메모리 청소
# ==========================================
def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================
# 유틸 함수
# ==========================================
def _parse_menu_line(line: str):
    """OCR 라인에서 이름과 가격 추출"""
    raw = (line or "").strip()
    if not raw:
        return {"name": "", "price": None}

    if "|" in raw:
        left, right = raw.split("|", 1)
        name = left.strip()
        price_str = re.sub(r"[^\d]", "", right)
        price = int(price_str) if price_str else None
        return {"name": name, "price": price}

    # 가격이 섞여 있는 경우
    price_match = re.search(r"(?:[¥￥]\s*)?(\d{2,4})\s*(?:円)?", raw)
    if price_match:
        price_str = price_match.group(1)
        price = int(price_str) if price_str else None
        name = re.sub(r"[¥￥]?\s*\d{2,4}\s*円?", "", raw).strip()
        return {"name": name, "price": price}

    return {"name": raw, "price": None}


# ==========================================
# 폰트 로더
# ==========================================
def get_jp_font(size):
    """일본어 전용 폰트"""
    try:
        return ImageFont.truetype(JP_FONT_PATH, size)
    except Exception as e:
        print(f"⚠️ JP 폰트 로딩 실패: {e}")
        return ImageFont.load_default()


def get_kr_font(size):
    """한국어 전용 폰트"""
    try:
        return ImageFont.truetype(KR_FONT_PATH, size)
    except Exception as e:
        print(f"⚠️ KR 폰트 로딩 실패: {e}")
        return ImageFont.load_default()


# ==========================================
# 메뉴 항목 추출
# ==========================================
def extract_menu_items(raw_text):
    """OCR 텍스트에서 메뉴 항목만 추출"""
    price_pattern = r'([ぁ-んァ-ン一-龯a-zA-Z\s]+?)\s*[¥￥]?\s*(\d{2,4})\s*[円]?'
    matches = re.finditer(price_pattern, raw_text)

    menu_items = []
    seen = set()

    for match in matches:
        name = match.group(1).strip()
        price = match.group(2)

        skip_patterns = [
            r'^\d+',
            r'[!！？?]',
            r'(一押|迷った|アナタ|人気|オリジナル|限定|おすすめ|新登場)',
            r'^[a-zA-Z\s]{1,2}$',
            r'(です|ます|から|、|。)',
        ]

        should_skip = any(re.search(pattern, name) for pattern in skip_patterns)

        if should_skip or len(name) < 2 or len(name) > 30:
            continue

        if not re.search(r'[ぁ-んァ-ン一-龯]', name):
            continue

        key = name + price
        if key in seen:
            continue
        seen.add(key)

        menu_items.append(f"{name} | ¥{price}")

    # 가격 없는 메뉴
    lines = raw_text.split('\n')
    for line in lines:
        line = line.strip()

        if any(line in item for item in menu_items):
            continue

        if re.search(r'[¥￥]\s*\d+|\d{2,4}\s*円', line):
            continue

        if not line or len(line) < 2 or len(line) > 25:
            continue

        if not re.search(r'[ぁ-んァ-ン一-龯]', line):
            continue

        skip_patterns = [
            r'[!！？?]',
            r'(一押|迷った|アナタ|人気|オリジナル|限定|おすすめ)',
            r'(です|ます|から|、|。)',
            r'^\d{4}',
        ]

        should_skip = any(re.search(pattern, line) for pattern in skip_patterns)

        if not should_skip and line not in seen:
            menu_items.append(line)
            seen.add(line)

    return menu_items


# ==========================================
# [Step 1] OCR: 손글씨 읽기
# ==========================================
def run_pure_ocr(image_path):
    print("\n👀 [Step 1] OCR: 손글씨 읽기...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            OCR_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        processor = AutoProcessor.from_pretrained(
            OCR_MODEL_ID,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28
        )

        prompt = "Read all text in this image."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1200)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"   📝 [OCR 원본]\n{raw_text[:300]}...\n")

        menu_items = extract_menu_items(raw_text)
        formatted_text = '\n'.join(menu_items)

        print(f"   ✅ 메뉴 추출 ({len(menu_items)}개)\n{formatted_text}\n")

        del model, processor
        flush_memory()
        return formatted_text

    except Exception as e:
        print(f"❌ OCR 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# [Step 2] LLM: 번역 및 설명 생성 (1개씩)
# ==========================================
def run_llm_logic(raw_text):
    print("\n🧠 [Step 2] LLM: 메뉴 정보 생성 중...")

    menu_lines = [line for line in raw_text.strip().split('\n') if line.strip()]
    all_menu_data = []

    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=bnb_config,
            device_map="cuda"
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # ✨ 시스템 프롬프트
        system_prompt = """You are a Japanese→Korean menu translator.
Output ONLY valid JSON. name_ko and description MUST be in Korean (한글)."""

        # ✨ 용어집
        glossary = """
[Japanese Food Glossary]
イクラ=이쿠라(연어알) | お造り=오츠쿠리(회) | 丼=동(덮밥)
納豆=낫토(청국장) | 梅干し=우메보시(매실장아찌) | 冷奴=히야얏코(냉두부)
なめろう=나메로(생선회무침) | 海鮮=카이센(해산물) | 定食=테이쇼쿠(정식)
カマ=카마(턱살) | 鮮魚=센교(신선한생선) | 焼き=야키(구이)
玉子=타마고(계란) | 卵=타마고(알) | 煮卵=니타마고(반숙란)
サバ=사바(고등어) | 明太=멘타이(명란) | カンパチ=칸파치(방어)
キムチ=김치 | ごはん=고항(밥) | おかわり=오카와리(리필)
のり=노리(김) | 胡麻=고마(참깨) | 追加=츠이카(추가)
一品=이핀(한그릇) | 増し=마시(추가) | ネタ=네타(재료)
トロ=토로(참치뱃살) | サーモン=사몬(연어) | まぐろ=마구로(참치)
うに=우니(성게) | えび=에비(새우) | たこ=타코(문어)
いか=이카(오징어) | ほたて=호타테(가리비) | 刺身=사시미(회)
生=나마(생) | 揚げ=아게(튀김) | 煮=니(조림)
炒め=이타메(볶음) | 蒸し=무시(찜) | 枝豆=에다마메(풋콩)
"""

        total = len(menu_lines)

        # ✨ 1개씩 처리
        for idx, line in enumerate(menu_lines):
            print(f"   ⚙️ [{idx + 1}/{total}] 처리 중: {line[:40]}...")

            parsed = _parse_menu_line(line)
            name = parsed["name"]
            price = parsed["price"]

            if not name:
                print(f"      ⚠️ 건너뜀 (이름 없음)")
                continue

            user_prompt = f"""
{glossary}

[Menu Item]
Name: "{name}"
Price: {price if price else "null"}

[Task]
Translate to Korean using the glossary above.
Output ONE JSON object:

{{
  "name": "{name}",
  "price": {price if price else "null"},
  "name_ko": "accurate Korean translation",
  "description": "Korean description (1-2 sentences)",
  "category": "meat|seafood|vegetable|drink|dessert|food",
  "t2i_prompt": "English food photography prompt"
}}

CRITICAL:
- name_ko MUST be in Korean (한글)
- description MUST be in Korean (한글)
- Use glossary for accurate translation
- Output ONLY JSON, no markdown, no commentary

Translate:
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=2000,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output_text = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # ✨ JSON 파싱 (실패하면 스킵)
            try:
                # 코드펜스 제거
                clean = re.sub(r'```json|```', '', output_text).strip()

                # 첫 { 부터 마지막 } 까지
                start = clean.find('{')
                end = clean.rfind('}')

                if start == -1 or end == -1:
                    raise ValueError("No JSON found")

                json_str = clean[start:end + 1]
                item = json.loads(json_str)

                # 원본 보존
                item["name"] = name
                item["price"] = price

                all_menu_data.append(item)
                print(f"      ✅ {name} → {item.get('name_ko', '?')}")

            except Exception as e:
                print(f"      ❌ 파싱 실패: {str(e)[:50]}")
                # Fallback: 기본값으로 추가
                all_menu_data.append({
                    "name": name,
                    "price": price,
                    "name_ko": name,
                    "description": "메뉴 설명",
                    "category": "food",
                    "t2i_prompt": "Japanese food on plate"
                })
                print(f"      ⚠️ 기본값 사용")
                continue

        del model, tokenizer
        flush_memory()

        print(f"\n   ✅ 총 {len(all_menu_data)}개 완료 (실패: {total - len(all_menu_data)}개)\n")

        if all_menu_data:
            print("   📋 생성된 메뉴:")
            for item in all_menu_data[:10]:
                print(f"      {item.get('name', '?')} → {item.get('name_ko', '?')}")
            if len(all_menu_data) > 10:
                print(f"      ... 외 {len(all_menu_data) - 10}개")
        print()

        return all_menu_data


    except Exception as e:

        print(f"      ❌ 파싱 실패: {str(e)[:50]}")

        # ✨ 카테고리 추측

        category = 'food'

        if re.search(r'(肉|牛|豚|鶏|チキン)', name):

            category = 'meat'

        elif re.search(r'(魚|刺身|寿司|海鮮|イカ|タコ|エビ)', name):

            category = 'seafood'

        elif re.search(r'(野菜|サラダ|キャベツ)', name):

            category = 'vegetable'

        elif re.search(r'(ビール|酒|ドリンク|ジュース)', name):

            category = 'drink'

        elif re.search(r'(デザート|ケーキ|アイス)', name):

            category = 'dessert'

        all_menu_data.append({

            "name": name,

            "price": price,

            "name_ko": name,

            "description": "메뉴 설명",

            "category": category,

            "t2i_prompt": f"Japanese {category} dish on plate, izakaya style"

        })

        print(f"      ⚠️ 기본값 사용 (category: {category})")


# ==========================================
# [Step 3] 이미지 생성
# ==========================================
def run_image_gen(menu_data):
    print("\n🎨 [Step 3] 이미지 생성 중...")
    if not menu_data:
        return []

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        print(f"❌ 이미지 모델 로딩 실패: {e}")
        return menu_data

    for idx, item in enumerate(menu_data):
        name = item.get('name', 'Unknown')
        name_ko = item.get('name_ko', '')
        base_prompt = item.get('t2i_prompt', 'delicious food on plate')

        full_prompt = (
            f"{base_prompt}, "
            "(masterpiece:1.3), (best quality:1.2), (photorealistic:1.4), "
            "professional food photography, 8k uhd, sharp focus, "
            "appetizing, detailed texture, shallow depth of field"
        )

        negative_prompt = (
            "text, watermark, logo, blurry, cartoon, anime, drawing, "
            "illustration, ugly, deformed, low quality, worst quality"
        )

        print(f"   🍱 [{idx + 1}/{len(menu_data)}] {name} ({name_ko})")

        try:
            image = pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=8,
                guidance_scale=2.5,
                height=512,
                width=512
            ).images[0]

            safe_name = re.sub(r'[<>:"/\\|?*]', '_', name[:30])
            path = os.path.join(OUTPUT_DIR, f"menu_{idx:02d}_{safe_name}.png")
            image.save(path)
            item['image_path'] = path
            print(f"      ✅ 저장 완료")

        except Exception as e:
            print(f"      ⚠️ 생성 실패: {str(e)[:60]}")

            placeholder = Image.new('RGB', (512, 512), (50, 50, 50))
            draw_ph = ImageDraw.Draw(placeholder)

            font = get_kr_font(24)
            text = f"{name_ko}\n\n이미지 생성\n실패"
            draw_ph.text(
                (256, 256),
                text,
                fill="white",
                anchor="mm",
                font=font,
                align="center"
            )

            safe_name = re.sub(r'[<>:"/\\|?*]', '_', name[:30])
            path = os.path.join(OUTPUT_DIR, f"menu_{idx:02d}_{safe_name}.png")
            placeholder.save(path)
            item['image_path'] = path

    del pipe
    flush_memory()
    return menu_data


# ==========================================
# [Step 4] 최종 메뉴판 조립
# ==========================================
def create_board(menu_items):
    print("\n🍱 [Step 4] 최종 메뉴판 생성 중...")
    if not menu_items:
        print("❌ 메뉴 항목이 없습니다")
        return

    cols = 3
    rows = math.ceil(len(menu_items) / cols)
    img_size = 420
    card_width = img_size + 40
    card_height = img_size + 200

    board_width = cols * card_width + 80
    board_height = rows * card_height + 140

    board = Image.new("RGB", (board_width, board_height), (20, 20, 20))
    draw = ImageDraw.Draw(board)

    # 폰트 로드
    title_font_jp = get_jp_font(48)
    font_name_jp = get_jp_font(24)
    font_ko = get_kr_font(22)
    font_desc_kr = get_kr_font(16)
    font_price_kr = get_kr_font(26)

    # 타이틀
    draw.text(
        (board_width // 2, 50),
        "🍶 メニュー 🍶",
        fill=(255, 255, 255),
        anchor="mm",
        font=title_font_jp
    )

    for idx, item in enumerate(menu_items):
        col = idx % cols
        row = idx // cols

        x = 40 + col * card_width
        y = 100 + row * card_height

        # 그림자
        shadow_offset = 4
        draw.rectangle(
            [x + shadow_offset, y + shadow_offset,
             x + card_width - 20 + shadow_offset, y + card_height - 20 + shadow_offset],
            fill=(10, 10, 10)
        )

        # 카드 배경
        draw.rectangle(
            [x, y, x + card_width - 20, y + card_height - 20],
            fill=(40, 40, 40),
            outline=(100, 100, 100),
            width=2
        )

        # 이미지
        img_x, img_y = x + 10, y + 10
        if 'image_path' in item and os.path.exists(item['image_path']):
            try:
                menu_img = Image.open(item['image_path']).resize((img_size, img_size))
                board.paste(menu_img, (img_x, img_y))
            except:
                draw.rectangle(
                    [img_x, img_y, img_x + img_size, img_y + img_size],
                    outline=(80, 80, 80),
                    width=2
                )
        else:
            draw.rectangle(
                [img_x, img_y, img_x + img_size, img_y + img_size],
                outline=(80, 80, 80),
                width=2
            )

        # 텍스트
        text_y = img_y + img_size + 20
        text_x = img_x + 10

        # 일본어 이름
        name = item.get('name', 'Unknown')
        if len(name) > 25:
            name = name[:23] + "..."
        draw.text(
            (text_x, text_y),
            name,
            fill=(255, 255, 255),
            font=font_name_jp
        )

        # 한국어 번역
        name_ko = item.get('name_ko', '')
        if len(name_ko) > 28:
            name_ko = name_ko[:26] + "..."
        draw.text(
            (text_x, text_y + 35),
            name_ko,
            fill=(255, 215, 0),
            font=font_ko
        )

        # 설명
        desc = item.get('description', '')
        if len(desc) > 35:
            if ' ' in desc[:35]:
                desc = desc[:35].rsplit(' ', 1)[0] + "..."
            else:
                desc = desc[:33] + "..."
        draw.text(
            (text_x, text_y + 70),
            desc,
            fill=(190, 190, 190),
            font=font_desc_kr
        )

        # 가격
        price = item.get('price')
        if price is not None:
            draw.text(
                (img_x + img_size - 10, text_y + 110),
                f"¥{int(price)}",
                fill=(100, 255, 100),
                font=font_price_kr,
                anchor="ra"
            )

    final_path = os.path.join(OUTPUT_DIR, "FINAL_MENU_BOARD.png")
    board.save(final_path, quality=95)

    print(f"\n✅ 완료!")
    print(f"   📁 저장 경로: {final_path}")
    print(f"   📊 총 {len(menu_items)}개 메뉴")


# ==========================================
# [메인 실행]
# ==========================================
if __name__ == "__main__":
    flush_memory()
    print("=" * 60)
    print("🍶 메뉴판 생성 시스템 (Simple Version)")
    print("=" * 60)

    raw_text = run_pure_ocr(INPUT_IMAGE_PATH)

    if raw_text:
        menu_data = run_llm_logic(raw_text)

        if menu_data:
            menu_data = run_image_gen(menu_data)
            create_board(menu_data)
        else:
            print("\n❌ 메뉴 데이터 생성 실패")
    else:
        print("\n❌ OCR 실패")

    print("\n" + "=" * 60)