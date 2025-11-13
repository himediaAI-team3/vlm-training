# VLM Training

Vision-Language 모델 학습을 위한 데이터셋 전처리 및 파인튜닝 파이프라인입니다.

이 레포지토리는 Vision-Language 모델 학습의 전체 워크플로우를 제공합니다:
- **데이터셋 전처리**: 이미지 파싱부터 GPT 기반 설명 생성까지
- **파인튜닝**: 데이터셋 학습, 모델 병합, vLLM 배포

---

## 📁 레포지토리 구조

```
vlm-training/
├── dataset/              # 데이터셋 전처리 파이프라인
│   ├── pipeline/        # 5단계 전처리 스크립트
│   ├── config_postprocess.py
│   ├── requirements.txt
│   └── README.md         # 상세 가이드
│
└── finetuning/          # 파인튜닝 코드
    ├── notebooks/       # 학습, 병합, 평가 노트북
    ├── vllm_langchain_test.py
    └── README.md        # 상세 가이드
```

---

## 🚀 빠른 시작

### 데이터셋 전처리

1. **환경 설정**
   ```bash
   cd dataset
   pip install -r requirements.txt
   cp env.example .env
   # .env 파일에 OpenAI API 키 입력
   ```

2. **5단계 실행**
   ```bash
   python pipeline/step1_parse_dataset.py    # 이미지 파싱
   python pipeline/step2_add_descriptions.py  # GPT 설명 생성
   python pipeline/step3_check_issues.py      # 문제점 탐지
   # config_postprocess.py 수정
   python pipeline/step4_fix_dataset.py      # 후처리 적용
   python pipeline/step5_upload_to_hub.py     # HuggingFace 업로드 (선택)
   ```

자세한 내용은 [`dataset/README.md`](dataset/README.md)를 참고하세요.

---

### 파인튜닝

1. **RunPod Pod 생성** (GPU VRAM 48GB 이상 권장)
2. **Step 1: 데이터셋 학습** - `finetuning/notebooks/dataset_study.ipynb` 실행
3. **Step 2: 모델 병합** - `finetuning/notebooks/model_merge.ipynb` 실행
4. **Step 3: vLLM 배포** - Pod 터미널에서 vLLM 서버 실행
5. **Step 4: 모델 테스트** - 배치 테스트 또는 혼동행렬 평가

자세한 내용은 [`finetuning/README.md`](finetuning/README.md)를 참고하세요.

---

## 📋 워크플로우 개요

### 데이터셋 전처리 파이프라인

```
원본 이미지
  ↓
Step 1: 이미지 파싱 + label 정리
  ↓
Step 2: GPT-4 설명 생성 (10,800회 API 호출)
  ↓
Step 3: 문제점 자동 탐지
  ↓
Step 4: 후처리 적용 (영어 용어 번역, 라벨 통일)
  ↓
Step 5: HuggingFace Hub 업로드 (선택)
  ↓
최종 데이터셋
```

### 파인튜닝 파이프라인

```
HuggingFace 데이터셋
  ↓
Step 1: 데이터셋 학습 (RunPod Pod)
  ↓
Step 2: 모델 병합
  ↓
Step 3: vLLM 배포
  ↓
Step 4: 모델 테스트
  ↓
배포 완료
```

---

## 📚 상세 문서

- **데이터셋 전처리**: [`dataset/README.md`](dataset/README.md)
  - 5단계 실행 가이드
  - 문제 해결 가이드
  - 프롬프트 언어 선택
  - 이미지 해상도 최적화

- **파인튜닝**: [`finetuning/README.md`](finetuning/README.md)
  - RunPod Pod 설정
  - 학습 하이퍼파라미터 설정
  - vLLM 배포 가이드
  - 문제 해결 팁

---

## ⚙️ 주요 특징

- **명확한 단계 분리**: 각 단계가 독립적으로 작동
- **자동 문제 탐지**: 영어 단어, 라벨 불일치 자동 발견
- **유연한 설정**: GPT 출력에 따라 설정 파일만 조정
- **반복 가능**: Step 3-4를 여러 번 반복하여 완벽하게 수정 가능
- **품질 보증**: 최종 검증된 데이터만 HuggingFace에 업로드

---

## 📝 라이선스

이 저장소의 **전처리 스크립트 및 코드**는 연구 및 교육 목적으로 자유롭게 사용 가능합니다.

**이미지 데이터 저작권**: 
- 본 프로젝트에서 활용한 이미지 데이터는 [AI Hub - 안면부 피부질환 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&&srchDataRealmCode=REALM006&aihubDataSe=data&dataSetSn=71863)에서 제공하는 데이터를 사용하였습니다.
- AI Hub 데이터 사용 시 [AI Hub 이용약관](https://www.aihub.or.kr/)을 준수해야 합니다.

---

## 🔗 참고 자료

- [Unsloth 공식 문서](https://github.com/unslothai/unsloth)
- [vLLM 공식 문서](https://docs.vllm.ai/)
- [RunPod 문서](https://docs.runpod.io/)
