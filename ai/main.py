import nltk
nltk.download('punkt_tab')  # "punkt_tab" 리소스 다운로드

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from transformers import pipeline, AutoModel
from utils.model_loader import load_model_and_tokenizer, predict
from utils.tools import article_cleanser, tokenize_sentences  # 전처리 함수들 임포트

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("score_article")

app = FastAPI(title="News AI API")

# === BERTClassifier 클래스 (필요시 사용) ===
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=3, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device)
        )
        out = self.dropout(pooler) if self.dr_rate else pooler
        return self.classifier(out)

# === 모델 로딩 ===
try:
    # 감성 예측 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    # 뉴스 요약 모델 로드 (서비스 시작 시 한 번만 로드)
    summarizer = pipeline("summarization", model="noahkim/KoT5_news_summarization")
except Exception as e:
    raise Exception(f"모델 로딩 실패: {str(e)}")

# === 요청/응답 모델 정의 (Score 및 Summarization용) ===

class ScoreRequest(BaseModel):
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

# === API 엔드포인트 ===

@app.get("/")
async def home():
    return {"message": "Welcome to the News AI API!"}

@app.post("/score", response_model=ScoreResponse)
async def score_article(input_data: ScoreRequest):
    """
    뉴스 기사 본문(content)을 받아 전처리(article_cleanser, tokenize_sentences)한 후,
    각 문장별 (positive - negative) * 100 값을 계산하여 모든 문장의 평균 점수를 반올림하고,
    전처리된 본문(content)과 감성 score를 함께 반환합니다.
    """
    try:
        logger.debug("Received content: %s", input_data.content)
        # 본문 정제: 불필요한 구문 및 특수문자 제거
        cleaned_content = article_cleanser(input_data.content)
        logger.debug("Cleaned content: %s", cleaned_content)
        # 문장 단위로 분할
        sentences = tokenize_sentences(cleaned_content)
        logger.debug("Tokenized sentences: %s", sentences)
        # 전처리된 본문: 각 문장을 줄바꿈 문자로 결합
        processed_content = "\n".join(s.strip() for s in sentences if s.strip())
        logger.debug("Processed content: %s", processed_content)

        scores = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # 빈 문장은 건너뜁니다.
                neg, neu, pos = predict(model, tokenizer, sentence)
                logger.debug("Sentence: '%s' | neg: %f, neu: %f, pos: %f", sentence, neg, neu, pos)
                # 각 문장의 감성 점수 계산 (positive - negative) * 100
                score = (pos - neg) * 100
                logger.debug("Calculated score for sentence: %f", score)
                scores.append(score)
        if not scores:
            logger.error("No valid sentences found after processing.")
            raise HTTPException(status_code=400, detail="유효한 문장이 없습니다.")
        average_score = round(sum(scores) / len(scores))
        logger.debug("Average score: %d", average_score)
        return ScoreResponse(content=processed_content, score=average_score)
    except Exception as e:
        logger.exception("Error in score_article endpoint")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest):
    """
    입력된 뉴스 텍스트(content)를 요약하여 반환합니다.
    """
    try:
        logger.debug("Summarization request content: %s", request.content)
        result = summarizer(
            request.content,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample
        )
        summary_text = result[0]["summary_text"]
        logger.debug("Summary text: %s", summary_text)
        return SummarizationResponse(summary_content=summary_text)
    except Exception as e:
        logger.exception("Error in summarize endpoint")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
