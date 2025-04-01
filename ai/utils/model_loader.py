import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple
import logging

# 로깅 설정
logger = logging.getLogger("model_loader")
logging.basicConfig(level=logging.INFO)


def load_model_and_tokenizer():
    """사전학습된 모델과 토크나이저를 로드"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"디바이스 설정: {device}")

    try:
        # 모델 및 토크나이저 로드
        model = AutoModelForSequenceClassification.from_pretrained(
            "snunlp/KR-FinBert-SC",
            num_labels=4,
            ignore_mismatched_sizes=True
            ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")

        # 추가 가중치 로딩 (fine-tuned 모델)
        model_path = 'models/finance_model.pt'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)

            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint, strict=False)
            elif hasattr(checkpoint, 'state_dict'):
                model.load_state_dict(checkpoint.state_dict(), strict=False)
            else:
                logger.warning("체크포인트 형식이 예상과 다릅니다.")

            logger.info("모델 가중치를 성공적으로 로드했습니다.")
        else:
            logger.warning("로컬 가중치 파일이 존재하지 않습니다. 원본 모델만 사용됩니다.")

        model.eval()
        return model, tokenizer

    except Exception as e:
        logger.exception("모델 로딩 중 오류 발생")
        raise RuntimeError(f"모델 로드 실패: {str(e)}")


def predict(model: torch.nn.Module, tokenizer: AutoTokenizer, sentence: str) -> Dict[str, float]:
    """문장에 대한 감성 예측 확률 반환"""
    encoded = tokenizer(sentence,
                        max_length=512,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt')

    device = next(model.parameters()).device
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probabilities = F.softmax(logits, dim=1).cpu().numpy()

    return {
        'negative': float(probabilities[0][0]),
        'positive': float(probabilities[0][1]),
        'neutral': float(probabilities[0][2]),
        'composite': float(probabilities[0][3])
    }


def calculate_sentiment_score(prediction_result, neutral_alpha=0.5, composite_beta=0.7):
    """
    감성 점수를 계산하는 함수

    종합점수 = [(긍정확률−부정확률)×(1−중립확률×α)]×복합조정계수
    복합조정계수 = (1 - 복합확률) × β

    Args:
        prediction_result: 예측 결과 딕셔너리
        neutral_alpha: 중립 영향력 계수 (기본값 0.5)
        composite_beta: 복합 영향력 계수 (기본값 0.7)

    Returns:
        float: -1.0 ~ 1.0 사이의 감성 점수
    """
    pos = prediction_result['positive']
    neg = prediction_result['negative']
    neu = prediction_result['neutral']
    comp = prediction_result['composite']

    # 복합 조정 계수 계산
    composite_adjustment = (1 - comp) * composite_beta

    # 종합 점수 계산
    sentiment_score = (pos - neg) * (1 - neu * neutral_alpha) * composite_adjustment

    return sentiment_score


def calculate_adjusted_sentiment_score(prediction_result, neutral_alpha=0.5, composite_beta=0.8):
    """
    복합 감성이 높을 때 조정된 감성 점수를 계산하는 함수
    
    Args:
        prediction_result: 예측 결과 딕셔너리
        neutral_alpha: 중립 영향력 계수 (기본값 0.5)
        composite_beta: 복합 영향력 계수 (기본값 0.8)
    
    Returns:
        float: -1.0 ~ 1.0 사이의 조정된 감성 점수
    """
    pos = prediction_result['positive']
    neg = prediction_result['negative']
    neu = prediction_result['neutral']
    comp = prediction_result['composite']
    
    # 복합 감성이 높은 경우
    if comp > 0.7:
        # 긍정과 부정의 상대적 비율 계산 (0에 가까울수록 부정, 1에 가까울수록 긍정)
        pos_neg_ratio = pos / (pos + neg + 0.0001)  # 0으로 나누기 방지
        
        # 상대적 비율을 -1~1 범위로 변환 (-1: 완전 부정, 1: 완전 긍정)
        sentiment_direction = 2 * pos_neg_ratio - 1
        
        # 복합성과 중립성을 고려한 최종 점수 계산
        adjusted_score = sentiment_direction * (1 - comp * 0.5) * (1 - neu * neutral_alpha)
        return adjusted_score
    
    # 복합 감성이 낮은 경우 기존 방식 사용
    else:
        return calculate_sentiment_score(prediction_result, neutral_alpha, composite_beta)

def finance_score(prediction_result, neutral_alpha=0.5, composite_beta=0.8):
    adjusted_score = calculate_adjusted_sentiment_score(prediction_result, neutral_alpha, composite_beta)
    adjusted_score_10 = 10 * adjusted_score
    return adjusted_score, adjusted_score_10