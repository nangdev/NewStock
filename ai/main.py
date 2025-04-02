import nltk

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch.nn as nn
from utils.model_loader import load_all_models_and_tokenizers
from utils.predictor import predict, compute_article_score, calculate_weighted_article_score
from utils.preprocessor import preprocessing_single_news
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("score_article")

app = FastAPI(title="News AI API")

# 글로벌 executor 선언 (필요시 FastAPI startup 이벤트에서 선언 가능)
executor = ThreadPoolExecutor(max_workers=5)

# === 모델 및 요약기 초기화 ===
try:
    model_dict = load_all_models_and_tokenizers()
    summarizer = pipeline("summarization", model="noahkim/KoT5_news_summarization")
    logger.info("모든 모델과 요약기 로딩 완료")
except Exception as e:
    logger.exception("모델 로딩 실패")
    raise RuntimeError(f"모델 로딩 실패: {str(e)}")


# === 요청/응답 스키마 ===
class ScoreRequest(BaseModel):
    title: str
    content: str

class ScoreResponse(BaseModel):
    content: str
    aspect_scores: Dict[str, float]
    score: float

class SummarizationRequest(BaseModel):
    content: str
    max_length: int = 300
    min_length: int = 40
    do_sample: bool = False

class SummarizationResponse(BaseModel):
    summary_content: str


# === API 엔드포인트 정의 ===
@app.get("/")
async def home():
    return {"message": "Welcome to the News AI API!"}


@app.post("/score", response_model=ScoreResponse)
async def score_article(input_data: ScoreRequest):
    def predict_score_for_category(category, model, tokenizer, device, sentence: str):
        try:
            result = predict(model, tokenizer, sentence, device)
            score = compute_article_score(result)[1]
            return category, score
        except Exception as e:
            logger.warning(f"[{category}] 예측 실패: {e}")
            return category, 0.0

    try:
        news_dict = {"title": input_data.title, "content": input_data.content}
        processed = preprocessing_single_news(news_dict)

        if not processed:
            return ScoreResponse(content="", aspect_scores={}, score=0.0)

        sentences = processed["filtered_sentences"]
        cleaned_content = processed["cleaned_content"]
        logger.info("유효 문장 수: %d", len(sentences))

        aspect_scores: Dict[str, List[float]] = {cat: [] for cat in model_dict}
        futures = []

        # 각 문장에 대해 관련 카테고리만 병렬 처리
        for s in sentences:
            sentence = s["sentence"]
            if not sentence:
                continue

            for category in [cat for cat in model_dict if s.get(cat, 0) == 1]:
                model, tokenizer, device = model_dict[category]
                future = executor.submit(predict_score_for_category, category, model, tokenizer, device, sentence)
                futures.append(future)

        # 결과 수집
        for f in futures:
            category, score = f.result()
            aspect_scores[category].append(score)

        # 평균 점수 계산
        average_scores = {
            category: round(sum(scores) / len(scores), 3) if scores else 0.0
            for category, scores in aspect_scores.items()
        }

        article_score = calculate_weighted_article_score(aspect_scores)

        return ScoreResponse(
            content=cleaned_content,
            aspect_scores=average_scores,
            score=article_score
        )

    except Exception as e:
        logger.exception("Error in /score endpoint")
        raise HTTPException(status_code=400, detail="문서 분석 중 오류가 발생했습니다.")


@app.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest):
    try:
        result = summarizer(
            request.content,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample
        )
        summary_text = result[0]["summary_text"]
        return SummarizationResponse(summary_content=summary_text)
    except Exception as e:
        logger.exception("Error in /summarize endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# === 서버 실행 ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)