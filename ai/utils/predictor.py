import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple, List
import torch.nn.functional as F

# predict + scoring 관련 함수

def predict(model: torch.nn.Module, tokenizer: AutoTokenizer, sentence: str, device: torch.device) -> Dict[str, float]:
    """문장에 대한 감성 예측 확률 반환"""
    encoded = tokenizer(sentence,
                        max_length=512,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt')

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

# def calculate_adjusted_sentiment_score(prediction_result, neutral_alpha=0.6, composite_beta=0.6):
#     pos = prediction_result['positive']
#     neg = prediction_result['negative']
#     neu = prediction_result['neutral']
#     comp = prediction_result['composite']

#     diff = pos - neg

#     # 🎯 감정 차이 완화 (덜 극단적으로)
#     if diff >= 0:
#         diff_scaled = diff ** 1.1
#     else:
#         diff_scaled = -((-diff) ** 1.1)

#     # 🎯 조정 계수 강화
#     adjusted_score = diff_scaled * (1 - neu * neutral_alpha) * (1 - comp * composite_beta)

#     return round(adjusted_score, 2)


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

def compute_article_score(prediction_result, neutral_alpha=0.5, composite_beta=0.8):
    adjusted_score = calculate_adjusted_sentiment_score(prediction_result, neutral_alpha, composite_beta)
    adjusted_score_10 = 10 * adjusted_score
    return adjusted_score, adjusted_score_10

def calculate_weighted_article_score(aspect_scores: Dict[str, List[float]]) -> float:
    total_weighted_score = 0.0
    total_count = 0

    for scores in aspect_scores.values():
        count = len(scores)
        if count > 0:
            avg_score = sum(scores) / count
            total_weighted_score += avg_score * count
            total_count += count

    return round(total_weighted_score / total_count, 2) if total_count > 0 else 0.0
