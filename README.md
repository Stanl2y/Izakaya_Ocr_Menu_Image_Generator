# Izakaya OCR Menu Image Generator

손글씨/이미지 메뉴판에서 **OCR로 메뉴를 추출**하고, 각 메뉴를 **일본어→한국어 번역 + 설명 생성**한 뒤, **메뉴 이미지(T2I) 생성** 및 **최종 메뉴판 이미지로 조립**하는 파이프라인입니다.

![Final Menu Board](final_menu_project/FINAL_MENU_BOARD.png)

## What it does

`main.py` 한 파일로 아래 과정을 순서대로 수행합니다.

1. **OCR**: `Qwen/Qwen2.5-VL-3B-Instruct` + LoRA 어댑터로 이미지 내 텍스트를 읽고 메뉴 라인만 정제
2. **LLM**: `Qwen/Qwen2.5-7B-Instruct`로 메뉴명/가격을 기반으로 `name_ko`, `description`, `category`, `t2i_prompt` JSON 생성
3. **Image Gen**: `SG161222/RealVisXL_V3.0_Turbo`로 메뉴 이미지 생성
4. **Board**: 생성된 이미지/텍스트를 카드 형태로 배치해 `FINAL_MENU_BOARD.png` 출력

## Prerequisites

- Python `3.10+` (WSL 기준 이 레포에서는 `python3` 사용)
- NVIDIA GPU + CUDA (코드가 기본적으로 `device_map="cuda"`를 사용)
- Hugging Face 모델 다운로드가 가능한 환경 (첫 실행 시 수 GB 다운로드 발생)
- 폰트 파일
  - 일본어 폰트: `JP_FONT_PATH`
  - 한국어 폰트: `KR_FONT_PATH`

## Installation

### 1) 가상환경 생성 (권장)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2) PyTorch 설치

CUDA 버전에 맞는 PyTorch를 설치하세요. (환경마다 달라서 본 README에 고정 명령을 넣지 않았습니다.)

### 3) 나머지 의존성 설치

```bash
pip install -U transformers diffusers peft accelerate bitsandbytes qwen-vl-utils pillow
```

### 4) (선택) Hugging Face 로그인

일부 모델은 약관 동의/로그인이 필요할 수 있습니다.

```bash
huggingface-cli login
```

## Quickstart

1) `main.py` 상단의 설정을 본인 환경에 맞게 수정합니다.

- `INPUT_IMAGE_PATH`: 입력 메뉴 이미지 경로
- `OUTPUT_DIR`: 결과 저장 폴더 (기본: `final_menu_project`)
- `LORA_ADAPTER_PATH`: OCR LoRA 어댑터 경로 (기본: `output/qwen25_vl_ocr_lora_20eps_new`)
- `JP_FONT_PATH`, `KR_FONT_PATH`: 폰트 파일 경로 (현재 값은 작성자 PC 경로)

2) 실행

```bash
python3 main.py
```

## Outputs

실행이 완료되면 기본적으로 `final_menu_project/` 아래에 결과가 생성됩니다.

- `final_menu_project/FINAL_MENU_BOARD.png`: 최종 메뉴판 이미지
- `final_menu_project/menu_XX_<menu_name>.png`: 메뉴별 생성 이미지 (실패 시 플레이스홀더 이미지 저장)

## Project structure

- `main.py`: 전체 파이프라인
- `data/`
  - `data/images/`: (학습/실험용) 이미지들
  - `data/dataset.jsonl`: (학습/실험용) 이미지-텍스트(JSONL) 데이터
- `output/`: OCR LoRA 어댑터/토크나이저 등 산출물
- `final_menu_project/`: 샘플 출력물

## Notes / Troubleshooting

- **Windows 네이티브 환경**에서는 `bitsandbytes`/CUDA 세팅이 까다로울 수 있습니다. 가능하면 **WSL2 또는 Linux** 환경을 권장합니다.
- **레포 용량**: `output/`(LoRA 어댑터)와 이미지 폴더는 커질 수 있어, GitHub에 올릴 때는 **Git LFS 사용** 또는 **불필요한 산출물 제외**를 권장합니다.
- **VRAM 부족(OOM)**: 이미지 생성(SDXL 계열) + LLM/OCR까지 포함되어 VRAM 요구가 큽니다. 부족하면
  - `run_image_gen()`의 `height/width`를 줄이거나
  - 이미지 생성을 건너뛰고(함수 호출 제거) OCR/번역만 먼저 확인하세요.
- **폰트 깨짐/□ 표시**: `JP_FONT_PATH`, `KR_FONT_PATH`가 유효한지 확인하세요.
- **LoRA 어댑터 없이 OCR 실행**: `run_pure_ocr()`에서 `PeftModel.from_pretrained(...)` 부분을 주석 처리하면 베이스 모델로 OCR을 시도할 수 있습니다.

## License

이 레포에 별도 라이선스 파일이 없습니다. 공개/배포 목적이라면 `LICENSE` 추가를 권장합니다.  
또한 사용한 모델들의 라이선스는 각 Hugging Face 모델 페이지를 따릅니다.
