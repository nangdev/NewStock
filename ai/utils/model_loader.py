import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model_and_tokenizer():
    try:
        # device 정의
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 및 토크나이저 로드
        model = AutoModelForSequenceClassification.from_pretrained(
            "snunlp/KR-FinBert-SC",
            num_labels=4,
            ignore_mismatched_sizes=True
            ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")

        # state_dict 로드 시도
        try:
            model_path = 'models/finance_model.pt'

            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            
            # 체크포인트 구조에 따라 적절히 로드
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint, strict=False)
            else:
                # 전체 모델이 저장된 경우
                if hasattr(checkpoint, 'state_dict'):
                    model.load_state_dict(checkpoint.state_dict(), strict=False)
                else:
                    print("체크포인트 형식이 예상과 다릅니다.")            
            print("모델 가중치를 성공적으로 로드했습니다")
        
        except Exception as e:
            print(f"state_dict 로드 실패: {str(e)}")

        model.eval()
        print("모델을 성공적으로 로드했습니다")
        return model, tokenizer

    except Exception as e:
        print(f"오류 상세 내용: {str(e)}")
        raise Exception(f"모델 로드 중 오류 발생: {str(e)}")


def predict(model, tokenizer, predict_sentence):
    # 토큰화 및 인코딩
    encoded = tokenizer(predict_sentence,
                        max_length=512,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt')
    
    # 디바이스로 이동
    device = next(model.parameters()).device
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
    
    # 각 클래스별 확률값을 개별적으로 추출
    return {
        'negative': float(probabilities[0][0]),
        'positive': float(probabilities[0][1]),
        'neutral': float(probabilities[0][2]),
        'composite': float(probabilities[0][3])
    }
