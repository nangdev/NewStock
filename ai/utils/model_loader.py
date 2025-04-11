import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple
import logging

# 로깅 설정
logger = logging.getLogger("model_loader")
logging.basicConfig(level=logging.INFO)

CATEGORY_LIST = ["finance", "strategy", "govern", "tech", "external"]
MODEL_DIR = "models"


import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_all_models_and_tokenizers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = {}

    for category in CATEGORY_LIST:
        try:
            # 1. 모델 구조 불러오기
            model = AutoModelForSequenceClassification.from_pretrained(
                "snunlp/KR-FinBert-SC",
                num_labels=4,
                ignore_mismatched_sizes=True
            ).to(device)

            # 2. 토크나이저 로딩
            tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")

            # 3. 가중치 파일 경로
            weight_path = os.path.join(MODEL_DIR, f"{category}_model.pt")

            # 4. 가중치 로드
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"[{category}] 모델 가중치 로드 완료")
            else:
                logger.warning(f"[{category}] 가중치 파일 없음 → 기본 모델 사용")

            model.eval()
            model_dict[category] = (model, tokenizer, device)

        except Exception as e:
            logger.exception(f"[{category}] 모델 로딩 실패: {e}")

    return model_dict

