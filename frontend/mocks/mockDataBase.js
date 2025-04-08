export const mockAllStock = {
  data: {
    stockList: [
      {
        stockId: 1,
        stockCode: '005930',
        stockName: '삼성전자',
        isInterested: true,
      },
      {
        stockId: 2,
        stockCode: '000660',
        stockName: 'SK하이닉스',
        isInterested: false,
      },
      {
        stockId: 3,
        stockCode: '040780',
        stockName: '카카오',
        isInterested: false,
      },
      {
        stockId: 4,
        stockCode: '023660',
        stockName: '삼성생명',
        isInterested: false,
      },
      {
        stockId: 5,
        stockCode: '110660',
        stockName: '테슬라',
        isInterested: false,
      },
      {
        stockId: 6,
        stockCode: '174660',
        stockName: '삼성화재',
        isInterested: false,
      },
      {
        stockId: 7,
        stockCode: '230660',
        stockName: '삼성증권',
        isInterested: false,
      },
    ],
  },
};

export const mockUserStock = {
  data: {
    stockList: [
      {
        stockId: 1,
        stockCode: '005930',
        stockName: '삼성전자',
        isInterested: true,
      },
      {
        stockId: 2,
        stockCode: '000660',
        stockName: 'SK하이닉스',
        isInterested: false,
      },
      {
        stockId: 3,
        stockCode: '040780',
        stockName: '카카오',
        isInterested: false,
      },
    ],
  },
};

export const mockNotificationList = {
  data: {
    notificationList: [
      {
        unId: 1,
        newsInfo: {
          newsId: 101,
          title: '카카오, 신규 서비스 출시 예정',
          publishedDate: '2025-04-06 10:00:00',
        },
        stockInfo: {
          stockId: 201,
          stockCode: '035720',
          stockName: '카카오',
        },
        isRead: true,
      },
      {
        unId: 2,
        newsInfo: {
          newsId: 102,
          title: '삼성전자, 반도체 수출 역대 최대',
          publishedDate: '2025-03-14 12:00:00',
        },
        stockInfo: {
          stockId: 202,
          stockCode: '005930',
          stockName: '삼성전자',
        },
        isRead: true,
      },
      {
        unId: 3,
        newsInfo: {
          newsId: 103,
          title: 'SK하이닉스, AI 서버 수요 증가 수혜',
          publishedDate: '2025-03-14 15:30:00',
        },
        stockInfo: {
          stockId: 203,
          stockCode: '000660',
          stockName: 'SK하이닉스',
        },
        isRead: false,
      },
      {
        unId: 4,
        newsInfo: {
          newsId: 104,
          title: '네이버, 글로벌 검색 점유율 확대',
          publishedDate: '2025-03-14 16:00:00',
        },
        stockInfo: {
          stockId: 204,
          stockCode: '035420',
          stockName: '네이버',
        },
        isRead: false,
      },
      {
        unId: 5,
        newsInfo: {
          newsId: 105,
          title: 'LG에너지솔루션, 배터리 대규모 수주123123',
          publishedDate: '2025-03-14 17:00:00',
        },
        stockInfo: {
          stockId: 205,
          stockCode: '373220',
          stockName: 'LG에너지솔루션',
        },
        isRead: false,
      },
      {
        unId: 6,
        newsInfo: {
          newsId: 106,
          title: '현대차, 전기차 글로벌 판매 1위 도전',
          publishedDate: '2025-03-14 17:30:00',
        },
        stockInfo: {
          stockId: 206,
          stockCode: '005380',
          stockName: '현대차',
        },
        isRead: true,
      },
      {
        unId: 7,
        newsInfo: {
          newsId: 107,
          title: '기아, 친환경차 라인업 강화 발표',
          publishedDate: '2025-03-14 18:00:00',
        },
        stockInfo: {
          stockId: 207,
          stockCode: '000270',
          stockName: '기아',
        },
        isRead: false,
      },
      {
        unId: 8,
        newsInfo: {
          newsId: 108,
          title: '포스코홀딩스, 철강가격 상승 수혜 전망',
          publishedDate: '2025-03-14 18:30:00',
        },
        stockInfo: {
          stockId: 208,
          stockCode: '005490',
          stockName: '포스코홀딩스',
        },
        isRead: false,
      },
      {
        unId: 9,
        newsInfo: {
          newsId: 109,
          title: '셀트리온, 신약 글로벌 임상 성공',
          publishedDate: '2025-03-14 19:00:00',
        },
        stockInfo: {
          stockId: 209,
          stockCode: '068270',
          stockName: '셀트리온',
        },
        isRead: false,
      },
      {
        unId: 10,
        newsInfo: {
          newsId: 110,
          title: 'HMM, 해운 운임 상승 수혜 지속',
          publishedDate: '2025-03-14 19:30:00',
        },
        stockInfo: {
          stockId: 210,
          stockCode: '011200',
          stockName: 'HMM',
        },
        isRead: false,
      },
    ],
  },
};

export const mockNewsletter = {
  data: {
    newsletterList: [
      {
        stockId: '1',
        content: `- **부산대**와 **한화오션**이 방위산업 인재 양성 및 기술 공동연구를 위한 **업무협약**을 체결했다.
- **LG전자 베스트샵 송우점**이 리뉴얼 오픈 기념으로 **가전 세일 행사**를 4월 11일부터 5월 8일까지 진행한다.
- 웨딩 및 입주 고객을 위한 **특별 혜택**과 다품목 구매 시 최대 **750만 원 할인**이 제공된다.
- 카카오톡이 **일시 장애**를 겪었으며, 이는 헌법재판소의 윤석열 전 대통령 파면 직후 **트래픽 폭증**으로 인한 것이다.
- 과학기술정보통신부는 **장애 원인**을 조사 중이며, 카카오는 긴급 대응을 통해 문제를 해결했다고 밝혔다.`,
        keywordList: [
          {
            keyword: '부산',
            count: 8,
          },
          {
            keyword: '서울',
            count: 6,
          },
          {
            keyword: '울산',
            count: 3,
          },
          {
            keyword: '인천',
            count: 1,
          },
        ],
      },
      {
        stockId: '2',
        content: `- **부산대**와 **한화오션**이 방위산업 인재 양성 및 기술 공동연구를 위한 **업무협약**을 체결했다.
- **LG전자 베스트샵 송우점**이 리뉴얼 오픈 기념으로 **가전 세일 행사**를 4월 11일부터 5월 8일까지 진행한다.
- 웨딩 및 입주 고객을 위한 **특별 혜택**과 다품목 구매 시 최대 **750만 원 할인**이 제공된다.
- 카카오톡이 **일시 장애**를 겪었으며, 이는 헌법재판소의 윤석열 전 대통령 파면 직후 **트래픽 폭증**으로 인한 것이다.
- 과학기술정보통신부는 **장애 원인**을 조사 중이며, 카카오는 긴급 대응을 통해 문제를 해결했다고 밝혔다.
- 카카오톡이 **일시 장애**를 겪었으며, 이는 헌법재판소의 윤석열 전 대통령 파면 직후 **트래픽 폭증**으로 인한 것이다.
- 과학기술정보통신부는 **장애 원인**을 조사 중이며, 카카오는 긴급 대응을 통해 문제를 해결했다고 밝혔다.
- 카카오톡이 **일시 장애**를 겪었으며, 이는 헌법재판소의 윤석열 전 대통령 파면 직후 **트래픽 폭증**으로 인한 것이다.
- 과학기술정보통신부는 **장애 원인**을 조사 중이며, 카카오는 긴급 대응을 통해 문제를 해결했다고 밝혔다.
- 카카오톡이 **일시 장애**를 겪었으며, 이는 헌법재판소의 윤석열 전 대통령 파면 직후 **트래픽 폭증**으로 인한 것이다.
- 과학기술정보통신부는 **장애 원인**을 조사 중이며, 카카오는 긴급 대응을 통해 문제를 해결했다고 밝혔다.`,
        keywordList: [
          {
            keyword: '부산',
            count: 3,
          },
          {
            keyword: '서울',
            count: 9,
          },
          {
            keyword: '울산',
            count: 8,
          },
          {
            keyword: '인천',
            count: 2,
          },
        ],
      },
    ],
  },
};

export const mockScrapNewsList = {
  data: {
    totalPage: 3,
    newsList: [
      {
        newsId: 1,
        title: '삼성전자, 반도체 수출 역대 최대',
        description:
          '삼성전자가 반도체 수출 역대 최대치를 기록했다. 이는 글로벌 반도체 시장의 회복세에 기인한다.',
        publishedDate: '2025-03-14 12:00:00',
        score: 10,
      },
      {
        newsId: 2,
        title: 'SK하이닉스, AI 서버 수요 증가 수혜',
        description:
          'SK하이닉스가 AI 서버 수요 증가로 인해 매출 증가가 예상된다. 이는 반도체 시장의 성장과 맞물려 있다.',
        publishedDate: '2025-03-14 15:30:00',
        score: 8,
      },
      {
        newsId: 3,
        title: '카카오, 신규 서비스 출시 예정',
        publishedDate: '2025-03-14 16:00:00',
        description:
          '카카오는 신규 서비스를 출시할 예정이다. 이는 카카오의 서비스 다각화 전략의 일환으로 보인다.',
        score: 7,
      },
      {
        newsId: 4,
        title: 'LG에너지솔루션, 배터리 대규모 수주',
        publishedDate: '2025-03-14 17:00:00',
        description:
          'LG에너지솔루션이 배터리 대규모 수주를 발표했다. 이는 전기차 시장의 성장과 맞물려 있다.',
        score: 9,
      },
      {
        newsId: 5,
        title: '현대차, 전기차 글로벌 판매 1위 도전',
        description: '현대차가 전기차 글로벌 판매 1위 도전을 선언했지만, 쉽지 않아보인다.',
        publishedDate: '2025-03-14 17:30:00',
        score: 3,
      },
      {
        newsId: 6,
        title: '현대차, 전기차 글로벌 판매 1위 도전',
        description: '현대차가 전기차 글로벌 판매 1위 도전을 선언했지만, 쉽지 않아보인다.',
        publishedDate: '2025-03-14 17:30:00',
        score: 3,
      },
      {
        newsId: 7,
        title: '현대차, 전기차 글로벌 판매 1위 도전',
        description: '현대차가 전기차 글로벌 판매 1위 도전을 선언했지만, 쉽지 않아보인다.',
        publishedDate: '2025-03-14 17:30:00',
        score: 3,
      },
      {
        newsId: 8,
        title: '현대차, 전기차 글로벌 판매 1위 도전',
        description: '현대차가 전기차 글로벌 판매 1위 도전을 선언했지만, 쉽지 않아보인다.',
        publishedDate: '2025-03-14 17:30:00',
        score: 3,
      },
    ],
  },
};
