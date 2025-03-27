import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

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

def load_model_and_tokenizer():
    try:
        # 토크나이저와 BERT 모델 초기화
        tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
        bertmodel = AutoModel.from_pretrained("skt/kobert-base-v1", return_dict=False)
        
        # 새 모델 인스턴스 생성
        model = BERTClassifier(bertmodel, dr_rate=0.5)
        
        # state_dict 로드 시도
        try:
            model_path = 'models/E_score_model.pt'

            import torch.serialization
            torch.serialization.add_safe_globals([BERTClassifier])
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                # 전체 모델이 저장된 경우, 가중치만 추출하여 로드
                model = checkpoint
        except Exception as e:
            print(f"{str(e)}")
            print()
            print("모델의 state_dict 정보가 없으므로 모델 전체 로드를 시도합니다.")
            # 다른 방법으로 시도
        model.eval()
        print("모델을 성공적으로 로드했습니다.")
        return model, tokenizer
    except Exception as e:
        print(f"오류 상세 내용: {str(e)}")
        raise Exception(f"모델 로드 중 오류 발생: {str(e)}")


def predict(model, tokenizer, predict_sentence):
    max_len = 64
    
    encoded = tokenizer(predict_sentence,
                       max_length=max_len,
                       padding='max_length',
                       truncation=True,
                       return_tensors='pt')
    
    token_ids = encoded['input_ids']
    valid_length = (token_ids != 0).sum(dim=1)
    
    # segment_ids를 0으로 초기화 (단일 문장이므로)
    segment_ids = torch.zeros_like(token_ids)

    model.eval()
    with torch.no_grad():
        out = model(token_ids, valid_length, segment_ids)
        probabilities = F.softmax(out[0], dim=0).numpy()
    
    return probabilities[0], probabilities[1], probabilities[2]

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    
    # 테스트 문장으로 예측 테스트
    test_sentence = "최근 일본정부가 후쿠시마 제1원전 오염수의 바다 방출을 사실상 확정하면서 우진의 방사능 제염사업이 연일 부각되고 있다."
    neg, neu, pos = predict(model, tokenizer, test_sentence)
    print(f"부정: {neg:.3f}, 중립: {neu:.3f}, 긍정: {pos:.3f}")
