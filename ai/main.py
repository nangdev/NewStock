import nltk
nltk.download('punkt')  # punkt 리소스 다운로드

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch.nn as nn
from utils.model_loader import load_model_and_tokenizer, predict, finance_score
from utils.preprocessor import preprocessing_single_news
from typing import Tuple, List
import logging
import uvicorn

# 키워드 추출을 위한 라이브러리 (krwordrank 사용)
from krwordrank.word import KRWordRank
from collections import Counter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("score_article")

app = FastAPI(title="News AI API")

# === 모델 및 토크나이저, 요약기 초기화 ===
def initialize_models() -> Tuple:
    try:
        model, tokenizer = load_model_and_tokenizer()
        model.eval()
        summarizer = pipeline("summarization", model="noahkim/KoT5_news_summarization")
        logger.info("모든 모델이 정상적으로 로딩되었습니다.")
        return model, tokenizer, summarizer
    except Exception as e:
        logger.exception("모델 로딩 실패")
        raise RuntimeError(f"모델 로딩 실패: {str(e)}")

model, tokenizer, summarizer = initialize_models()

# === 요청/응답 스키마 ===
class ScoreRequest(BaseModel):
    title: str
    content: str

class ScoreResponse(BaseModel):
    content: str
    score: int

class SummarizationRequest(BaseModel):
    content: str
    max_length: int = 300
    min_length: int = 40
    do_sample: bool = False

class SummarizationResponse(BaseModel):
    summary_content: str

class KeywordResponse(BaseModel):
    keywords: List[str]

# 여러 기사를 받아서 전체 키워드를 집계하는 요청
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
            return ScoreResponse(content="", score=0)

        sentences = processed["filtered_sentences"]
        cleaned_content = processed["cleaned_content"]

        logger.info("문장 리스트: %s", sentences)

        # 재무적성과가 1인 문장만 필터링
        finance_sentences = [s for s in sentences if s["재무적성과"] == 1 and s["sentence"].strip()]

        if not finance_sentences:
            return ScoreResponse(content=cleaned_content, score=0)

        # 점수 예측
        scores = [finance_score(predict(model, tokenizer, s["sentence"].strip()))[1] for s in finance_sentences]

        logger.info("점수 리스트: %s", scores)

        average_score = round(sum(scores) / len(scores))
        logger.info(f"재무적성과 점수: {average_score}")

        return ScoreResponse(content=cleaned_content, score=average_score)

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
        summary_text = result[0]["summary_text"]
        return SummarizationResponse(summary_content=summary_text)
    except Exception as e:
        logger.exception("Error in /summarize endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/keywords", response_model=KeywordResponse)
def get_keywords(request: KeywordRequest):
    try:
        if not request.articles:
            raise HTTPException(status_code=400, detail="기사 하나 이상 제공해야 합니다.")

        # 여러 기사에서 텍스트만 추출
        docs = [article.content for article in request.articles]

        # 불용어 집합 정의 (필요에 따라 단어들을 추가/수정)
        stopwords = {
            "국내","해외","판매","글로벌","대비","전년","시장","증가","기아","투자"
        }

        # KRWordRank 초기화 (불용어 집합 포함)
        wordrank_extractor = KRWordRank(
            min_count=5,    # 후보 단어가 등장해야 하는 최소 빈도수
            max_length=10,  # 후보 단어의 최대 길이
            verbose=True,
            stopwords=stopwords
        )

        beta = 0.85
        max_iter = 10

        # extract 메서드 사용 (전체 결과 추출 후 직접 상위 10개 선택)
        keywords, ranks, graph = wordrank_extractor.extract(docs, beta, max_iter)

        # 추출된 키워드 딕셔너리를 점수 기준 내림차순 정렬 후 상위 10개 선택
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        top_keywords = [word for word, score in top_keywords]

        return KeywordResponse(keywords=top_keywords)

    except Exception as e:
        logger.exception("Error in /keywords endpoint with krwordrank")
        raise HTTPException(status_code=500, detail=str(e))



# === 서버 실행 ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
