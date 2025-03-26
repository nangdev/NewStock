from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoModel
from utils.model_loader import load_model_and_tokenizer, predict

# BERTClassifier 클래스 정의를 main.py에 복사
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
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

app = FastAPI()

# 모델과 토크나이저 로드
model, tokenizer = load_model_and_tokenizer()

# 입력 데이터 모델 정의
class SentenceInput(BaseModel):
    sentence: str

@app.get("/")
async def home():
    return {"message": "Welcome to the prediction API!"}

# API 엔드포인트 정의
@app.post("/predict")
async def predict_sentence(input_data: SentenceInput):
    '''
    문장을 입력하면 부정, 중립, 긍정 점수를 반환합니다.
    '''
    try:
        # 예측 수행
        neg, neu, pos = predict(model, tokenizer, input_data.sentence)
        
        # 예측 결과 반환
        return {
            "negative": f"{neg:.3f}",
            "neutral": f"{neu:.3f}",
            "positive": f"{pos:.3f}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
