# 표준 라이브러리
import logging
from collections import Counter
from typing import Tuple, Dict, List

# 외부 라이브러리
import nltk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from konlpy.tag import Okt
import torch.nn as nn
import uvicorn

# 사용자 정의 모듈
from utils.model_loader import load_all_models_and_tokenizers
from utils.predictor import predict, compute_article_score, calculate_weighted_article_score
from utils.preprocessor import preprocessing_single_news


# === 로깅 설정 ===
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("score_article")

# === FastAPI 앱 및 글로벌 객체 초기화 ===
app = FastAPI(title="News AI API")
okt = Okt()

# === 모델 및 요약기 초기화 ===
try:
    model_dict = load_all_models_and_tokenizers()
    summarizer = pipeline("summarization", model="noahkim/KoT5_news_summarization")
    logger.info("모든 모델과 요약기 로딩 완료")
except Exception as e:
    logger.exception("모델 로딩 실패")
    raise RuntimeError(f"모델 로딩 실패: {str(e)}")

# === 요청/응답 스키마 정의 ===
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

class KeywordItem(BaseModel):
    word: str
    count: int

class KeywordResponse(BaseModel):
    keywords: List[KeywordItem]

class Article(BaseModel):
    content: str

class KeywordRequest(BaseModel):
    articles: List[Article]

# === API 엔드포인트 정의 ===
@app.get("/")
async def home():
    return {"message": "Welcome to the News AI API!"}


@app.post("/score", response_model=ScoreResponse)
async def score_article(input_data: ScoreRequest):
    try:
        news_dict = {"title": input_data.title, "content": input_data.content}
        processed = preprocessing_single_news(news_dict)

        if not processed:
            return ScoreResponse(content="", aspect_scores={}, score=0.0)


        sentences = processed["filtered_sentences"]
        cleaned_content = processed["cleaned_content"]
        logger.info("문장 리스트: %s", sentences)

        aspect_scores: Dict[str, List[float]] = {cat: [] for cat in model_dict.keys()}

        # 각 문장에 대해 카테고리별 점수 계산
        for s in sentences:
            sentence = s["sentence"]
            if not sentence:
                continue

            for category in [cat for cat in model_dict if s.get(cat, 0) == 1]:
                model, tokenizer, device = model_dict[category]
                score = compute_article_score(predict(model, tokenizer, sentence, device))[1]
                aspect_scores[category].append(score)

        logger.info("카테고리별 점수 리스트: %s", aspect_scores)

        # 평균 점수 계산
        average_scores = {
            category: round(sum(scores) / len(scores), 3) if scores else 0.0
            for category, scores in aspect_scores.items()
        }

        # 종합 점수 계산 (가중 평균 방식)
        article_score = calculate_weighted_article_score(aspect_scores)

        return ScoreResponse(content=cleaned_content, aspect_scores=average_scores, score=article_score)

    except Exception as e:
        logger.exception("Error in /score endpoint")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest):
    try:
        result = summarizer(
            request.content,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample
        )
        return SummarizationResponse(summary_content=result[0]["summary_text"])
    except Exception as e:
        logger.exception("Error in /summarize endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/keywords", response_model=KeywordResponse)
def get_keywords(request: KeywordRequest):
    try:
        if not request.articles:
            raise HTTPException(status_code=400, detail="기사 하나 이상 제공해야 합니다.")

        docs = [article.content for article in request.articles]
        allowed_pos = {"Noun"}
        stopwords = {
            "전년", "지난해", "올해", "동월", "현대차", "계획", "모비스", "퍼센트", "대다", "하다", "기아", "대비", "가장", "있다"
        }

        keywords = [
            word for doc in docs
            for word, tag in okt.pos(doc, stem=True)
            if tag in allowed_pos and len(word) > 1 and word not in stopwords
        ]

        counter = Counter(keywords)
        keyword_items = [KeywordItem(word=word, count=count) for word, count in counter.most_common(10)]
        return KeywordResponse(keywords=keyword_items)

    except Exception as e:
        logger.exception("Error in /keywords endpoint with Okt + POS filtering")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)