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

# 키워드 추출을 위한 라이브러리
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter
from konlpy.tag import Mecab

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("score_article")

app = FastAPI(title="News AI API")

# === 모델 및 토크나이저, 요약기, 키워드 모델 초기화 ===
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

# 키워드 모델 초기화 (다국어 임베딩 모델 사용)
embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
keyword_model = KeyBERT(model=embedding_model)
logger.info("키워드 모델이 정상적으로 로딩되었습니다.")

# Mecab 형태소 분석기 초기화
mecab = Mecab()

def extract_nouns(text: str) -> List[str]:
    return mecab.nouns(text)

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
def aggregate_keywords(request: KeywordRequest):
    try:
        if not request.articles:
            raise HTTPException(status_code=400, detail="기사를 하나 이상 제공해야 합니다.")

        all_keywords = []
        for article in request.articles:
            nouns = extract_nouns(article.content)
            if not nouns:
                continue
            text = " ".join(nouns)
            keywords = keyword_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 1),
                stop_words=None,
                top_n=10
            )
            extracted = [kw for kw, score in keywords]
            all_keywords.extend(extracted)

        keyword_counter = Counter(all_keywords)
        top_keywords = [keyword for keyword, count in keyword_counter.most_common(10)]

        return KeywordResponse(keywords=top_keywords)

    except Exception as e:
        logger.exception("Error in /aggregate_keywords endpoint")
        raise HTTPException(status_code=500, detail=str(e))


# === 서버 실행 ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
