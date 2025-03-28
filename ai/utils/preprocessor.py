import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)


# 상수 정의
TITLE_KEYWORDS = ['부고', '인사', '연예']
TITLE_START_PATTERNS = ['[인사]', '[사사건건]', '[MBN 토요포커스]']
CONTENT_KEYWORDS = ['자이언츠', '랜더스', 'KT위즈', 'KT 위즈', '트윈스', '타이거즈', '라이온즈', '다이노스', '이글스', '베어스', '히어로즈']
CONTENT_START_PATTERNS = ['[SBS 김성준의 시사전망대]', '신청해 주셨던 분들']

ASPECT_DEF = {
    '재무적성과': ['매출', '이익', '수익', 'EBITDA', 'ROE', '손실'],
    '전략적성장': ['M&A', '합병', '신사업', '투자', '글로벌'],
    '기술혁신': ['특허', 'R&D', 'AI', '디지털', '자동화'],
    '자본구조': ['부채', '자본', '증자', '배당', '주식'],
    '외부환경': ['환율', '금리', '규제', '정책', '원자재']
}

def apply_initial_filters(news):
    """초기 필터링 조건 검증"""
    if any(kw in news.get('title', '') for kw in TITLE_KEYWORDS):
        return False
    if any(news['title'].startswith(p) for p in TITLE_START_PATTERNS):
        return False
    if any(kw in news.get('content', '') for kw in CONTENT_KEYWORDS):
        return False
    if len(news.get('content', '')) < 30:
        return False
    if any(news['content'].startswith(p) for p in CONTENT_START_PATTERNS):
        return False
    return True


def clean_text(text):
    """텍스트 클렌징 통합 처리 함수"""
    text = text.replace('韓', '한국')
    
    # [1단계] 기본 전처리
    patterns = [
        (r'\(서울=연합뉴스\)(\s*[가-힣]+\s*기자\s*=)?\s*', ''),
        (r'\[.*?\]', ''),  # [ ] 제거
        (r'\<.*?\>', ''),  # < > 제거
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ''),  # 이메일 제거
        (r'[가-힣]{2,4}\s?(?:인턴)?기자', ''),  # 기자 이름 제거
        (r'[▶◆■▲△▲◇▷ⓒ#]', ' '),  # 특수문자 처리
    
        # [2단계] 콘텐츠 필터링
        (r'네이버.*?구독하기', ''),  # 네이버 관련 문구 제거
        (r'[a-z]*\.com', ''),  # URL 제거
    
        # [3단계] 문자 정제
        (r'([一-鿕]|[㐀-䶵]|[豈-龎])+', ''),  # 한자 제거
        (r'\(※.*?\)', ''),  # ※ 주석 제거
    
        # [4단계] 주식 코드 보존
        (r'\((\d{6})\)', r'\1'),  # 6자리 주식코드 괄호 제거
    
        # [5단계] 특수 기호 처리
        (r'[\"\'\u201c\u201d\u2018\u2019]', ''),  # 따옴표 제거
        (r'[()]', ''),  # 잔여 괄호 제거
    
        # [7단계] 최종 정제
        (r'GoodNews.*재배포금지', ''),
        (r'제보는 카카오톡.*?송고\s*|<저작권자.*?송고\s*', ''),
        (r'/서경뉴스봇', ''),
    ]
    
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    # [6단계] 숫자 포맷팅
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1퍼센트', text)  # % → 퍼센트 변환
    text = re.sub(r'(?<=\d),(?=\d{3})', '', text)  # 3자리 콤마 패턴 제거

    # 일괄 치환
    replacements = {
        '\xa0': ' ', '\t': ' ', '/사진제공=dl': '',
        '@': ' ', '㈜': '', '\n': ' ', '\r': ' ', 
        '-': ' ', '·': ',',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # 최종 정제
    return text.lower().strip()


def split_sentences(text):
    """문장 분할 유틸리티"""
    text = re.sub(r'(\d+)\.(\d+)', r'\1<dot>\2', text)
    text = text.replace('.', '. ')
    sentences = sent_tokenize(text)
    return [s.replace('<dot>', '.') for s in sentences]


def analyze_sentences(sentences):
    """문장 분석 및 필터링"""
    valid_sentences = []
    for idx, sentence in enumerate(sentences):
        if not 20 <= len(sentence) <= 200:
            continue
        
        aspect_flags = {
            aspect: int(any(kw in sentence for kw in keywords))
            for aspect, keywords in ASPECT_DEF.items()
        }
        
        if sum(aspect_flags.values()) > 0:
            valid_sentences.append({
                'id': idx,
                'sentence': sentence,
                'length': len(sentence),
                **aspect_flags
            })
    return valid_sentences


def preprocessing_single_news(news):
    """
    단일 뉴스 데이터를 전처리하는 함수
    - 입력: news 딕셔너리 (title, content필수)
    - 출력: 처리된 news 딕셔너리 or None (필터링시)
    """
    if not apply_initial_filters(news):
        return None

    news['cleaned_content'] = clean_text(news['content'])
    news['sentences'] = split_sentences(news['cleaned_content'])

    if len(news['sentences']) >= 200:
        return None

    valid_sentences = analyze_sentences(news['sentences'])

    if not valid_sentences:
        return None

    news.update({
        'filtered_sentences': valid_sentences,
        'num_valid_sentence': len(valid_sentences),
        'aspect_counts': {k: sum(s[k] for s in valid_sentences) for k in ASPECT_DEF}
    })
    
    return news
