import nltk
nltk.download('punkt_tab')  # "punkt_tab" 리소스 다운로드

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch.nn as nn
from utils.model_loader import load_model_and_tokenizer, predict, finance_score
from utils.preprocessor import preprocessing_single_news
from typing import Tuple
import logging
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
            raise HTTPException(status_code=400, detail="재무적성과가 1인 문장이 없습니다.")

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


# === 서버 실행 ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)