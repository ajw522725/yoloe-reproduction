# 참고: 공식 YOLOE 구현 (설명 보조용)

이 폴더는 **팀의 재현 실험을 이해하는 데 필요한 배경 설명**만 제공합니다. 공식 코드 원본은 포함하지 않으며, 실제 구현은 아래 링크를 통해 확인할 수 있습니다.

- 논문: [YOLOE: Real-Time Seeing Anything](https://arxiv.org/abs/2503.07465) (Wang, Liu, Chen, Lin, Han, Ding — ICCV 2025)
- 공식 구현: [github.com/THU-MIG/yoloe](https://github.com/THU-MIG/yoloe)
- 사전학습 가중치: [HuggingFace — jameslahm/yoloe](https://huggingface.co/collections/jameslahm/yoloe-67d5110aabaefbe129c15917)

## 전체 구조

```
Image → Backbone → PAN → Head → {Detection, Segmentation}
                          ├─ Text prompt      → RepRTA
                          ├─ Visual prompt     → SAVPE
                          └─ Prompt-free       → LRPC
```

## RepRTA — 텍스트 프롬프트

**문제:** 이미지 특징은 픽셀에서, 텍스트 특징은 언어모델에서 나오기 때문에 raw dot product로는 잘 정렬되지 않습니다. 기존 방법들은 매 입력 이미지마다 이미지-텍스트 특징을 실시간으로 섞는 cross-modal 연산(카테고리 수 × 이미지 위치 수)을 수행해 비용이 큽니다.

**아이디어:** 텍스트 → 이미지 공간으로 가는 매핑 "규칙"을 경량 **보조 네트워크(Auxiliary Network, translation dictionary 역할)**로 학습해두고, 학습되지 않은 단어에도 재사용합니다.

- Transform path: 텍스트 특징을 공통 표현 공간으로 인코딩
- Gate path: 프롬프트별로 어떤 차원을 통과시킬지 결정 (예: "dog, cat" → animal 차원)
- 학습 후에는 고정된 fθ(P)를 컨볼루션 커널에 재파라미터화(re-parameterize)해 흡수시키므로, 추론 시에는 원래 YOLO head와 동일한 연산만 수행 → **추론/전이 오버헤드 0**

## SAVPE — 비전 프롬프트

박스/포인트 등 비전 프롬프트를 입력받아, Activation branch(프롬프트에 따라 달라지는 가중치)와 Semantic branch(프롬프트와 무관한 특징)를 분리해 집계함으로써 적은 연산으로 프롬프트 임베딩을 만들고, contrastive matching으로 탐지에 사용합니다.

## LRPC — 프롬프트-프리

입력 없이 모든 객체를 식별해야 하는 상황에서, "8,400개 앵커 × 4,585개 vocabulary"를 매번 전부 대조하는 대신 특화된 임베딩으로 먼저 앵커를 필터링(scan)한 뒤, 남은 앵커만 내재된 vocabulary와 매칭합니다. 대형 언어모델에 의존하지 않아 메모리·지연시간 부담이 적습니다.

## 팀 재현과의 연결

- `code/table1_zero_shot_eval/`은 위 세 메커니즘이 실제로 학습 없이도(zero-shot) open-vocabulary 성능을 내는지를 LVIS minival로 검증합니다.
- `code/table4_coco_finetune/`은 COCO에 대해 Linear Probing → Full Fine-tuning으로 전이했을 때 성능이 어떻게 변화하는지(FT > LP 경향)를 직접 학습해 확인합니다.

자세한 재현 결과와 한계는 [저장소 루트 README](../README.md)를 참고하세요.
