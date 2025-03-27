from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from transformers import pipeline, AutoModel
from utils.model_loader import load_model_and_tokenizer, predict
from utils.tools import tokenize_sentences, article_cleanser  # preprocessor 함수들 사용

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

# === 요청/응답 모델 정의 ===

# 클라이언트가 보내는 "newsText" 키를 그대로 사용하도록 alias 처리
class ArticleInput(BaseModel):
    article_text: str = Field(..., alias="newsText")

# 뉴스 요약 관련 모델
class SummarizationRequest(BaseModel):
    news_text: str
    max_length: int = 300
    min_length: int = 40
    do_sample: bool = False

class SummarizationResponse(BaseModel):
    summary_text: str

# === API 엔드포인트 ===

@app.get("/")
async def home():
    return {"message": "Welcome to the News AI API!"}

@app.post("/score")
async def score_article(input_data: ArticleInput):
    """
    클라이언트가 전달한 뉴스 텍스트를 preprocessor 과정을 거쳐 정제한 후,
    tokenize_sentences 함수를 통해 문장 단위로 분할하고 각 문장별 (positive - negative) * 100 값을 계산하여
    모든 문장의 평균 score를 반올림하여 반환합니다.
    """
    try:
        # 1. 기사 정제: article_cleanser 함수를 적용하여 불필요한 구문 제거
        cleaned_text = article_cleanser(input_data.article_text)
        
        # 2. 문장 단위 분할: tokenize_sentences 함수를 사용
        sentences = tokenize_sentences(cleaned_text)
        
        scores = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # 빈 문장은 건너뜁니다.
                neg, neu, pos = predict(model, tokenizer, sentence)
                # (positive - negative) * 100 값 계산
                score = (pos - neg) * 100
                scores.append(score)
        if not scores:
            raise HTTPException(status_code=400, detail="유효한 문장이 없습니다.")
        average_score = round(sum(scores) / len(scores))
        return {"score": average_score}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest):
    """
    입력된 뉴스 텍스트를 요약하여 반환합니다.
    """
    try:
        result = summarizer(
            request.news_text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample
        )
        summary_text = result[0]["summary_text"]
        return SummarizationResponse(summary_text=summary_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
