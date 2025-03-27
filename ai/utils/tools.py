# 러이브러리 import
import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

# 함수 정의

def exclude_by_keywords(df, column_name, keywords):
    return df[~df[column_name].str.contains('|'.join(keywords))]


def exclude_by_categories(df, column_name, categories):
    return df[~df[column_name].isin(categories)]


def exclude_by_start_patterns(df, column_name, patterns):
    return df[~df[column_name].str.startswith(tuple(patterns))]


def filter_news(df):
    # 제외할 카테고리 및 키워드 정의
    categories = ['연예', '오피니언', '사람속으로', '뉴스광장 1부', '서경스타', 'News Today', '통합뉴스룸ET']
    categories_keywords = ['농구', '프로야구', '스포츠', '골프', '야구', '축구', '배구', '연예', '방송', '사설', '엔터']
    title_keywords = ['부고', '인사']
    article_keywords = ['자이언츠', '랜더스', 'KT위즈', 'KT 위즈', '트윈스', '타이거즈', '라이온즈', '다이노스', '이글스', '베어스', '히어로즈']
    title_start_patterns = ['[인사]', '[사사건건]', '[MBN 토요포커스]']
    article_start_patterns = ['[SBS 김성준의 시사전망대]', '신청해 주셨던 분들']

    # 필터링 적용
    df = exclude_by_categories(df, 'category', categories)
    df = exclude_by_keywords(df, 'category', categories_keywords)
    df = exclude_by_keywords(df, 'title', title_keywords)
    df = exclude_by_keywords(df, 'article', article_keywords)

    # 추가 조건 필터링
    df = df[~((df['category'] == '증권') & (df['category_str'] == '홈>증권>증권정보'))]
    
    # 본문 길이 기준 필터링
    df = df[df['article'].apply(len) >= 30]

    # 제목과 본문 시작 문구 기준 필터링
    df = exclude_by_start_patterns(df, 'title', title_start_patterns)
    df = exclude_by_start_patterns(df, 'article', article_start_patterns)

    return df    


def replace_category_by_keyword(df, column_name, category_map):
    """
    카테고리 키워드에 따라 카테고리를 변경하는 함수.

    Parameters:
        df (pd.DataFrame): 데이터프레임
        column_name (str): 카테고리 열 이름
        category_map (dict): 키워드와 대체 카테고리 매핑 딕셔너리

    Returns:
        pd.DataFrame: 카테고리가 변경된 데이터프레임
    """
    for keywords, replacement in category_map.items():
        df.loc[df[column_name].str.contains(keywords, na=False), column_name] = replacement
    return df


def filter_categories_by_count(df, column_name, min_count):
    """
    특정 카테고리에 속한 기사 개수가 최소 개수 이상인 경우만 필터링.

    Parameters:
        df (pd.DataFrame): 데이터프레임
        column_name (str): 카테고리 열 이름
        min_count (int): 최소 개수 기준

    Returns:
        pd.DataFrame: 필터링된 데이터프레임
    """
    category_counts = df[column_name].value_counts()
    filtered_categories = category_counts[category_counts >= min_count].index
    return df[df[column_name].isin(filtered_categories)].reset_index(drop=True)


def article_cleanser(text):
    result = result1 = re.sub(r'\(서울=연합뉴스\)\s*', '', text)

    # [1단계] 기본 전처리
    result = re.sub(r'\[.*?\]', '', result)  # [ ] 제거
    result = re.sub(r'\<.*?\>', '', result)  # < > 제거
    result = re.sub(r'[a-zA-Z0-9]+@[a-zA-Z]+(\.[a-z]{2,4}){1,2}', '', result)  # 이메일 제거
    result = re.sub(r'[가-힣]{2,4}\s?(?:인턴)?기자', '', result)  # 기자 이름 제거
    result = re.sub(r'[▶◆■▲△▲◇▷]', ' ', result)  # 특수문자 처리
    
    # [2단계] 콘텐츠 필터링
    result = re.sub(r'네이버.*?구독하기', '', result)  # 네이버 관련 문구 제거
    result = re.sub(r'[a-z]*\.com', '', result)  # URL 제거
    result = result.replace('\xa0', ' ').replace('\t', '').replace('/사진제공=dl', '')
    
    # [3단계] 문자 정제
    result = re.sub(r'([一-鿕]|[㐀-䶵]|[豈-龎])+', '', result)  # 한자 제거
    result = re.sub(r'\(※.*?\)', '', result)  # ※ 주석 제거
    
    # [4단계] 주식 코드 보존
    result = re.sub(r'\((\d{6})\)', r'\1', result)  # 6자리 주식코드 괄호 제거 (예: (034220) → 034220)
    
    # [5단계] 특수 기호 처리
    result = re.sub(r'[\"\'\u201c\u201d\u2018\u2019]', '', result)  # 따옴표 제거
    result = re.sub(r'[()]', '', result)  # 잔여 괄호 제거
    
    # [6단계] 숫자 포맷팅
    result = result.replace('\n', ' ').replace('\r', ' ').replace('-', ' ')
    result = re.sub(r'(\d+(?:\.\d+)?)%', r'\1퍼센트', result)  # % → 퍼센트 변환 (공백 문제 해결)
    result = re.sub(r'(?<=\d),(?=\d{3})', '', result)  # 3자리 콤마 패턴 제거 (예: 357,815,700 → 357815700)
    
    # [7단계] 최종 정제
    result = re.sub(r'제보는 카카오톡.*?송고\s*|<저작권자.*?송고\s*', '', result, flags=re.DOTALL)
    result = re.sub(r'/서경뉴스봇', '', result)
    result = result.strip()

    return result

def preprocess_for_tokenization(text):
    """소수점을 임시 문자열로 대체"""
    return re.sub(r'(\d+)\.(\d+)', r'\1<dot>\2', text)

def postprocess_after_tokenization(sentences):
    """임시 문자열을 다시 소수점으로 복원"""
    return [re.sub(r'<dot>', '.', sentence) for sentence in sentences]


def tokenize_sentences(text):
    # 소수점을 임시 문자열로 대체
    protected_text = preprocess_for_tokenization(text)

    # 마침표 -> 마침표+공백
    protected_text = protected_text.replace('.', '. ')

    # 문장 분리
    tokenized_sentences = sent_tokenize(protected_text)
    
    # 임시 문자열을 다시 소수점으로 복원
    final_sentences = postprocess_after_tokenization(tokenized_sentences)
    return final_sentences
