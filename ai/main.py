from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch
import torch.nn as nn
from utils.model_loader import load_model_and_tokenizer, predict
from utils.preprocessor import preprocessing_single_news

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
    model, tokenizer = load_model_and_tokenizer()
    model.eval()
    summarizer = pipeline("summarization", model="noahkim/KoT5_news_summarization")
except Exception as e:
    raise Exception(f"모델 로딩 실패: {str(e)}")

# === 요청/응답 모델 정의 ===
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

# === API 엔드포인트 ===
@app.get("/")
async def home():
    return {"message": "Welcome to the News AI API!"}

@app.post("/score", response_model=ScoreResponse)
async def score_article(input_data: ScoreRequest):
    try:
        news_dict = {
            "title": input_data.title,
            "content": input_data.content
        }

        processed = preprocessing_single_news(news_dict)

        if not processed:
            return ScoreResponse(content="", score=0)

        sentences = processed["filtered_sentences"]
        cleaned_content = processed["cleaned_content"]

        logger.debug("sentences: %s", sentences)
        logger.debug("cleaned_content: %s", cleaned_content)  

        scores = []
        for sent_obj in sentences:
            sentence_text = sent_obj["sentence"].strip()
            if sentence_text:
                neg, neu, pos = predict(model, tokenizer, sentence_text)
                score = (pos - neg) * 100
                scores.append(score)


        if not scores:
            raise HTTPException(status_code=400, detail="유효한 문장이 없습니다.")

        average_score = round(sum(scores) / len(scores))
        return ScoreResponse(content=cleaned_content, score=average_score)

    except Exception as e:
        logger.exception("Error in score_article endpoint")
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
        logger.exception("Error in summarize endpoint")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)