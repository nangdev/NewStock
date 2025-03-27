# 패키지 import
import pandas as pd
import re

# 함수 import
from utils.tools import filter_news
from utils.tools import replace_category_by_keyword
from utils.tools import filter_categories_by_count
from utils.tools import article_cleanser
from utils.tools import tokenize_sentences


# 1. 데이터 불러오기
## 1.1. 뉴스 데이터
news = pd.DataFrame()
news = news.drop_duplicates('title')  # 제목 기준 중복 기사 제거
news = news.fillna('article').reset_index(drop=True)
news.rename(columns={'company': 'media'}, inplace=True)  # 뉴스 회사 컬럼명 변경
print('뉴스 확인 : ', news.shape)

## 1.2. 코스피 종목 데이터
kospi200_list = pd.DataFrame()['종목명']

# 2. 기사 필터링

## 2.1. 종목에 코스피200 종목 기업명이 포함된 기사만 필터링

# 코스피 200 종목 리스트를 길이 순으로 정렬 (긴 이름이 먼저 검색되도록)
sorted_companies = sorted(kospi200_list, key=len, reverse=True)

# 정확한 단어 매칭을 위한 정규표현식 패턴 생성
pattern = r'\b(?:' + '|'.join(map(re.escape, sorted_companies)) + r')\b'

# 정규표현식을 사용하여 제목 필터링
news_kospi_200 = news[news['title'].str.contains(pattern, regex=True)]
print('제목에 코스피200 종목 기업명이 포함된 기사만 필터링 완료 : ', news_kospi_200.shape)

## 2.2. 카테고리 기준 필터링
news_kospi_200 = filter_news(news_kospi_200)

print('카테고리, 제목, 본문 기준 뉴스 필터링 완료 : ', news_kospi_200.shape)

## 2.3. 카테고리 정제, 필터링

# 카테고리 매핑 사전
category_map = {
    '뉴스|취재K|영상K|보도자료|현장뉴스': '뉴스',
    '경제|기업·산업|재계인맥대해부|정책|ESG|비즈브리핑|내국세|고용': '경제',
    '법원': '법원/검찰',
    '기업|제약바이오|헬스|팩플|넘버스|비즈니스|라이프트렌드|맛있는 도전|창간|게임|무선|기타|사랑방|홈|화학·소재|여행레저|운수/교통|가전|consumer': '기업',
    '사회|사건|복지|위안부|노동|과학|교통|교육|보건|코로나': '사회',
    '국제|미국|중국|유럽|우크라이나|일본|세계': '국제',
    '기업|창간': '기업',
    '산업|바이오|보험|유통|반도체|자동차|인터넷|영화|industry|음악': '산업',
    '문화|공연|컬처': '문화',
    'IT|과학|모바일': 'IT/과학',
    '대통령|정치|국방|외교|국회': '정치',
    '부동산|real_estate': '부동산',
    '종목|증시|증권|채권|머니랩|market|공시|업계|증권|이슈|자본시장|금융증권': '증권',
    '금융|은행|finance': '금융',
    '국방|외교|국회|정치': '정치',
    '전국|대전세종충청|광주전남|전북|지방|춘천|청주|창원|포항|안동|순천|진주|지역|대구경북|대구|부산경남|수도권|광주|인천|제주|충주|울산|전주|대전|원주|목포': '전국',
}

# 카테고리 정리
news_kospi_200 = replace_category_by_keyword(news_kospi_200, 'category', category_map)

# 카테고리에 속한 기사 개수가 100개 이상인 경우만 필터링
news_kospi_200 = filter_categories_by_count(news_kospi_200, column_name='category', min_count=100)

print('카테고리 정리 및 필터링 완료:', news_kospi_200.shape)

# 4. 기사 전처리

## 4.1. 불필요한 구문 등 제거 및 정리

news_kospi_200['article2'] = news_kospi_200['article'].apply(article_cleanser)
print('기사 본문 1차 정제 완료')
print('예시: ', news_kospi_200['article2'][3333])

## 4.2. 기사 문장 단위 분할

news_kospi_200['sentences'] = news_kospi_200['article2'].apply(tokenize_sentences)
print('문장 분할 완료')
print('예시: ', news_kospi_200['sentences'][3333])

# 문장 개수 카운트
news_kospi_200['num_of_sentences'] = news_kospi_200['sentences'].apply(len)

# 문장이 200개 이하인 기사만 채택
news_kospi_200 = news_kospi_200[news_kospi_200['num_of_sentences']<200]

# 본문 전처리 및 문장 단위로 나뉜 뉴스 데이터 저장
news_kospi_200.to_csv('news_kospi_200_sentences.csv', encoding='utf-8-sig', index=False)# 

