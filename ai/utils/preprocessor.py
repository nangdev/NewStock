import re
import nltk
import logging
from typing import Optional
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

# 로그 설정
logger = logging.getLogger("preprocessor")
logging.basicConfig(level=logging.INFO)


# 상수 정의
TITLE_KEYWORDS = ['부고', '인사', '연예']
TITLE_START_PATTERNS = ['[인사]', '[사사건건]', '[MBN 토요포커스]']
CONTENT_KEYWORDS = ['자이언츠', '랜더스', 'KT위즈', 'KT 위즈', '트윈스', '타이거즈', '라이온즈', '다이노스', '이글스', '베어스', '히어로즈']
CONTENT_START_PATTERNS = ['[SBS 김성준의 시사전망대]', '신청해 주셨던 분들']

ASPECT_DEF = {
    '재무적성과': ['매출', '이익', '수익', 'EBITDA', 'ROE', '손실', '영업이익', 'GP율', '재무', '실적'],
    '전략적성장': ['M&A', '합병', '신사업', '투자', '글로벌'],
    '기술혁신': ['특허', 'R&D', 'AI', '디지털', '자동화'],
    '자본구조': ['부채', '자본', '증자', '배당', '주식'],
    '외부환경': ['환율', '금리', '규제', '정책', '원자재']
}


def apply_initial_filters(news: dict) -> bool:
    """뉴스 기사 필터링 조건 검사"""
    title = news.get('title', '')
    content = news.get('content', '')

    if any(kw in title for kw in TITLE_KEYWORDS):
        return False
    if any(title.startswith(p) for p in TITLE_START_PATTERNS):
        return False
    if any(kw in content for kw in CONTENT_KEYWORDS):
        return False
    if len(content) < 30:
        return False
    if any(content.startswith(p) for p in CONTENT_START_PATTERNS):
        return False

    logger.info('모든 조건을 통과하였습니다.')
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
    
    text = text.lower().strip()
    # 최종 정제
    logger.info('텍스트 정제가 완료되었습니다.')
    return text


def split_sentences(text: str) -> list:
    """문장 단위 분리"""
    text = re.sub(r'(\d+)\.(\d+)', r'\1<dot>\2', text)
    text = text.replace('.', '. ')
    sentences = sent_tokenize(text)
    sentences = [s.replace('<dot>', '.') for s in sentences]

    logger.info('문장 분할이 완료되었습니다.')
    return sentences


def analyze_sentences(sentences: list, min_len: int = 20, max_len: int = 200) -> list:
    """문장 필터링 및 관점 분석"""
    valid_sentences = []

    for idx, sentence in enumerate(sentences):
        if not (min_len <= len(sentence) <= max_len):
            continue

        aspect_flags = {
            aspect: int(any(kw in sentence for kw in keywords))
            for aspect, keywords in ASPECT_DEF.items()
        }

        if any(aspect_flags.values()):
            valid_sentences.append({
                'id': idx,
                'sentence': sentence,
                'length': len(sentence),
                **aspect_flags
            })

    logger.info(f'유효한 문장 수: {len(valid_sentences)}')
    return valid_sentences


def preprocessing_single_news(news: dict) -> Optional[dict]:
    """
    단일 뉴스 기사 전처리
    - 필터 조건 통과한 경우 정제 및 분석
    """

    if not apply_initial_filters(news):
        return None

    cleaned = clean_text(news['content'])
    sentences = split_sentences(cleaned)
    valid = analyze_sentences(sentences)

    if not valid:
        return None

    news.update({
        'cleaned_content': cleaned,
        'sentences': sentences,
        'filtered_sentences': valid,
        'num_valid_sentence': len(valid),
        'aspect_counts': {k: sum(s[k] for s in valid) for k in ASPECT_DEF}
    })

    logger.info('기사 모든 전처리 완료')
    return news