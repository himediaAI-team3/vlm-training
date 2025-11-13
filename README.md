# VLM Training

Vision-Language 모델 학습을 위한 데이터셋 전처리 및 파인튜닝 파이프라인입니다.

이 레포지토리는 Vision-Language 모델 학습의 전체 워크플로우를 제공합니다:
- **데이터셋 전처리**: 이미지 파싱부터 GPT 기반 설명 생성까지
- **파인튜닝**: 데이터셋 학습, 모델 병합, vLLM 배포

---

## 목차

1. [데이터셋 전처리 파이프라인](#데이터셋-전처리-파이프라인)
2. [파인튜닝 가이드](#파인튜닝-가이드)

---

# 데이터셋 전처리 파이프라인

피부 질환 진단을 위한 Vision-Language 모델 학습용 데이터셋 구축 파이프라인입니다.

---

## 작업 환경 준비 (필수)

**중요**: 이 데이터셋 전처리 작업은 **별도의 작업 폴더**에서 진행하는 것을 강력히 권장합니다.

### 왜 별도 폴더가 필요한가요?

데이터 전처리 과정에서 다음과 같은 대용량 파일들이 생성됩니다:
- `extracted_data/` - 원본 이미지 (10,800장, 수 GB)
- `skin_dataset/` - 중간 데이터셋 (수 GB)
- `skin_dataset_fixed/` - 최종 데이터셋 (수 GB)
- `skin_dataset_temp/` - 임시 파일

**메인 프로젝트 폴더에서 작업하면**:
- 프로젝트가 매우 지저분해짐
- Git 관리가 복잡해짐 (용량 큰 파일 추적)
- 불필요한 파일로 인한 혼란

### 권장 작업 방법

#### 방법 1: 별도 프로젝트 폴더 생성 (추천)

```bash
# 데이터셋 전처리 전용 폴더 생성
mkdir dataset-workspace
cd dataset-workspace

# 이 레포지토리 클론
git clone https://github.com/himediaAI-team3/vlm-training.git .

# 이미지 다운로드 및 배치
# extracted_data/ 폴더에 원본 이미지 배치

# 작업 진행
python pipeline/step1_parse_dataset.py
python pipeline/step2_add_descriptions.py
python pipeline/step3_check_issues.py
# config_postprocess.py 수정
python pipeline/step4_fix_dataset.py
python pipeline/step5_upload_to_hub.py
```

---

## 데이터셋 정보

- **대상 질환**: 건선, 아토피, 여드름, 주사, 지루, 정상 (6개 클래스)
- **데이터 규모**: Train 9,600개 / Test 1,200개
- **이미지 해상도**: 1024x1024 PNG
- **설명 생성**: GPT-4o-mini 기반 의학적 분석

---

## 전처리 워크플로우

```
Step 1: 이미지 파싱 + label 정리
  ↓
./skin_dataset/ (label 정리됨, output 비어있음)
  ↓
Step 2: GPT-4 설명 생성 (순수 출력)
  ↓
./skin_dataset/ (GPT 원본 출력)
  ↓
Step 3: 문제점 자동 탐지
  ↓
리포트 확인 → config_postprocess.py 수정
  ↓
Step 4: 후처리 적용
  ↓
./skin_dataset_fixed/ (최종본)
  ↓
Step 5: HuggingFace Hub 업로드 (선택)
  ↓
HuggingFace Hub
```

---

## 빠른 시작

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp env.example .env
# .env 파일을 열어서 OpenAI API 키를 입력하세요
```

### 2. 데이터 준비

원본 이미지를 다음 구조로 배치하세요:

**데이터 출처**: [AI Hub - 안면부 피부질환 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&&srchDataRealmCode=REALM006&aihubDataSe=data&dataSetSn=71863)

```
extracted_data/
├── Training/
│   └── images/
│       ├── TS_건선_정면/    (800장)
│       ├── TS_건선_측면/    (800장)
│       ├── TS_아토피_정면/  (800장)
│       ├── TS_아토피_측면/  (800장)
│       ├── TS_여드름_정면/  (800장)
│       ├── TS_여드름_측면/  (800장)
│       ├── TS_주사_정면/    (800장)
│       ├── TS_주사_측면/    (800장)
│       ├── TS_지루_정면/    (800장)
│       ├── TS_지루_측면/    (800장)
│       ├── TS_정상_정면/    (800장)
│       └── TS_정상_측면/    (800장)
└── Validation/
    └── images/
        ├── VS_건선_정면/    (100장)
        ├── VS_건선_측면/    (100장)
        ├── VS_아토피_정면/  (100장)
        ├── VS_아토피_측면/  (100장)
        ├── VS_여드름_정면/  (100장)
        ├── VS_여드름_측면/  (100장)
        ├── VS_주사_정면/    (100장)
        ├── VS_주사_측면/    (100장)
        ├── VS_지루_정면/    (100장)
        ├── VS_지루_측면/    (100장)
        ├── VS_정상_정면/    (100장)
        └── VS_정상_측면/    (100장)
```

**중요**: 
- 폴더명은 정확히 위와 같아야 합니다
- 이미지 파일은 `.png`, `.jpg`, `.jpeg` 형식 지원
- 각 폴더에 정확한 수량의 이미지가 있어야 합니다

---

## 5단계 실행 가이드

### Step 1: 이미지 파싱 + label 정리

```bash
python pipeline/step1_parse_dataset.py
```

**기능**:
- `extracted_data/` 폴더의 이미지 재귀 탐색
- HuggingFace Dataset 형식으로 변환
- **자동 label 정리**: `건선_정면` → `건선`

**출력**: `./skin_dataset/` (초기 데이터셋)

---

### Step 2: GPT-4 설명 생성

```bash
python pipeline/step2_add_descriptions.py
```

**기능**:
- GPT-4o-mini로 각 이미지의 의학적 분석 생성
- **순수 GPT 출력만 저장** (후처리 없음)
- 자동 중간 저장 (150개 단위)

**출력**: `./skin_dataset/` (GPT 원본 출력)

**중요**: 
- **이 단계는 매우 오래 걸립니다** (10,800회 API 호출)
- 중간 저장(150개 단위)이 자동으로 되므로, 중단되어도 이어서 실행 가능
- 이 단계에서는 영어 번역이나 라벨 통일을 하지 않습니다
- GPT의 원본 출력을 그대로 저장합니다

---

### Step 3: 문제점 자동 탐지

```bash
python pipeline/step3_check_issues.py
```

**기능**:
- 영어 단어 자동 탐지 (빈도 포함)
- 라벨 불일치 체크 (`label` 필드 vs `<label>` 태그)
- 통계 리포트 출력

**다음 작업**:
1. 리포트를 확인
2. `config_postprocess.py` 파일을 열기
3. 발견된 영어 용어를 `MEDICAL_TERMS`에 추가
   - **주의**: 형용사나 부사인 경우 단순 단어 번역이 아닌 **문장 전체를 문맥에 맞게 수정**해야 합니다
4. 발견된 라벨 변형을 `LABEL_MAPPING`에 추가

---

### Step 4: 후처리 적용

**작업 전 필수**: `config_postprocess.py`를 수정하여 Step 3에서 발견된 문제점 반영

```bash
python pipeline/step4_fix_dataset.py
```

**기능**:
- `config_postprocess.py`의 설정을 로드
- 영어 의학 용어 → 한글 번역
- output의 `<label>` 태그 통일

**출력**: `./skin_dataset_fixed/` (최종 데이터셋)

---

### Step 5: HuggingFace Hub 업로드 (선택)

```bash
python pipeline/step5_upload_to_hub.py
```

**기능**:
- `./skin_dataset_fixed/` → HuggingFace Hub 업로드
- `.env`에서 `HF_TOKEN` 읽기
- 메타데이터 자동 생성

**필수 설정** (`.env` 파일):
```
HF_TOKEN=your_huggingface_token
HF_REPO_NAME=username/dataset-name
```

---

## 파일 구조

### 실행 스크립트 (pipeline/)

| 파일                         | 설명                      | 입력                      | 출력                       |
|------------------------------|---------------------------|---------------------------|--------------------------|
| `step1_parse_dataset.py`     | 이미지 파싱 + label 정리   | `extracted_data/`         | `./skin_dataset/`         |
| `step2_add_descriptions.py`  | GPT-4 설명 생성            | `./skin_dataset/`         | `./skin_dataset/`        |
| `step3_check_issues.py`      | 문제점 자동 탐지           | `./skin_dataset/`         | 리포트 출력                 |
| `step4_fix_dataset.py`       | 후처리 적용                | `./skin_dataset/`         | `./skin_dataset_fixed/`  |
| `step5_upload_to_hub.py`     | HuggingFace Hub 업로드     | `./skin_dataset_fixed/`   | Hub                      |

### 설정 파일

| 파일                     | 설명                                                       |
|--------------------------|-----------------------------------------------------------|
| `config_postprocess.py`  | 영어 용어 + 라벨 매핑 (예시, Step 3 결과에 맞춰 수정)       |
| `env.example`            | 환경변수 템플릿                                             |
| `requirements.txt`       | 필요한 패키지 목록                                          |

---

## 문제 해결

### Q1. Step 2에서 중단되었어요
**A**: `step2_add_descriptions.py`를 다시 실행하세요. 이미 처리된 데이터는 자동으로 건너뜁니다.

### Q2. Step 3에서 영어 단어가 많이 발견되었어요
**A**: 정상입니다. `config_postprocess.py`에 해당 용어를 추가한 후 Step 4를 실행하세요.

### Q3. config_postprocess.py를 어떻게 수정하나요?
**A**: 
```python
# Step 3 리포트 확인
발견된 영어 단어
  - erythema (125회)
  - papule (89회)

# config_postprocess.py의 MEDICAL_TERMS에 추가
MEDICAL_TERMS = {
    "erythema": "홍반",    # Step 3에서 발견됨
    "papule": "구진",      # Step 3에서 발견됨
    # 기존 예시는 참고용, 실제 발견된 단어만 추가하면 됨
}
```

**주의**: 
- config_postprocess.py에 이미 작성된 내용은 예시입니다. Step 3 결과를 보고 실제로 필요한 항목만 추가/수정하세요.
- **형용사나 부사인 경우**: 단순히 단어만 번역하는 것이 아니라, 해당 단어가 포함된 **문장 전체를 문맥에 맞게 수정**해야 합니다. 예를 들어 "severe erythema"가 발견되면 단순히 "severe": "심한"으로 매핑하는 것보다, 해당 문장을 "심한 홍반"처럼 자연스러운 한글로 재작성하는 것이 좋습니다.

### Q4. Step 4 후에도 문제가 남아있어요
**A**: 
1. `python pipeline/step3_check_issues.py` 재실행
2. 새로 발견된 항목을 `config_postprocess.py`에 추가
3. `python pipeline/step4_fix_dataset.py` 재실행

### Q5. API 키 오류가 발생해요
**A**: `.env` 파일에 올바른 OpenAI API 키가 입력되었는지 확인하세요.

---

# 파인튜닝 가이드

Vision-Language 모델 파인튜닝 및 배포 파이프라인입니다.

자세한 내용은 [`finetuning/README.md`](finetuning/README.md)를 참고하세요.

## 빠른 시작

1. **RunPod Pod 생성** (GPU VRAM 48GB 이상 권장)
2. **Step 1: 데이터셋 학습** - `finetuning/notebooks/dataset_study.ipynb` 실행
3. **Step 2: 모델 병합** - `finetuning/notebooks/model_merge.ipynb` 실행
4. **Step 3: vLLM 배포** - Pod 터미널에서 vLLM 서버 실행
5. **Step 4: 모델 테스트** - 배치 테스트 또는 혼동행렬 평가

자세한 가이드는 [`finetuning/README.md`](finetuning/README.md)를 확인하세요.

---

## 라이선스 및 저작권

### 이미지 데이터 저작권

본 프로젝트에서 활용한 **이미지 데이터**는 [AI Hub - 안면부 피부질환 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&&srchDataRealmCode=REALM006&aihubDataSe=data&dataSetSn=71863)에서 제공하는 데이터를 사용하였습니다.

**AI Hub 데이터 사용 조건**:
- AI Hub의 데이터 이용 약관을 준수해야 합니다
- 상업적 이용 시 별도의 승인이 필요할 수 있습니다
- 데이터 출처를 반드시 명시해야 합니다

**중요**: 
- AI Hub 데이터를 사용할 경우, [AI Hub 이용약관](https://www.aihub.or.kr/)을 반드시 확인하고 준수하세요
- 이 저장소는 데이터 전처리 **스크립트만** 제공합니다
- 원본 이미지 데이터는 포함되어 있지 않으며, 사용자가 직접 AI Hub에서 다운로드해야 합니다

### 자체 데이터 사용

AI Hub 데이터 대신 **본인이 직접 수집하거나 저작권 문제가 없는 이미지**를 사용할 경우:
- 이 전처리 파이프라인을 자유롭게 사용할 수 있습니다
- 동일한 폴더 구조에 맞춰 이미지를 배치하세요
- 본인 데이터에 대한 저작권 관리는 사용자의 책임입니다

### 코드 라이선스

이 저장소의 **전처리 스크립트 및 코드**는 연구 및 교육 목적으로 자유롭게 사용 가능합니다.

